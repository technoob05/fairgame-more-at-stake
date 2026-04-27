"""
FAIRGAME More-at-Stake — Kaggle T4 experiment runner
=====================================================

Self-contained script that:
  1. Clones aira-list/FAIRGAME into the working dir.
  2. Installs HF + bitsandbytes deps.
  3. Registers a local HuggingFace LLM connector (4-bit nf4, fp16 compute)
     into FAIRGAME's MODEL_PROVIDER_MAP via monkey-patch.
  4. Sweeps the experiment matrix:
       models  x  games  x  languages  x  rounds  x  payoff_scales  x  seeds
     loading each model exactly once, freeing GPU before the next.
  5. Persists per-config CSV (resume-safe) + an aggregated summary.

How to run on Kaggle (T4 x2 recommended):
  - Create one notebook PER MODEL (so they run in parallel against your weekly
    quota). GPU T4 x2, Internet ON.
  - For gated HF models add a Kaggle Secret named HF_TOKEN. Models sourced
    from `kagglehub` or unsloth's bnb-4bit mirrors do not need it.
  - Single cell template:
        import os
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            pass

        !wget -q https://raw.githubusercontent.com/technoob05/fairgame-more-at-stake/main/exp/run_kaggle_t4.py
        !python run_kaggle_t4.py --model Llama3_1_8B --preset scale

Available models: see MODEL_REGISTRY below.
Available presets:
  smoke         1 cell      ~8 min    sanity check
  mini          40 cells    ~17h*     5 games x [10,50] rounds x [1,100] scales
  scale         60 cells    ~42h*     5 games x 50 rounds x 4 scales x 3 seeds
  rounds        45 cells    ~19h*     5 games x [10,30,50] rounds x scale=1
  multilingual  50 cells    ~35h*     5 games x 5 langs x 50 rounds x [1,100]
  pd_replication 150 cells  ~21h*    EXACT match of paper's setup: PD, 5 langs,
                                      lambda in {0.1, 1, 10}, 10 rounds, 10 seeds
  pd_paper      100 cells   ~42h*    EXTENSION: PD, 5 langs, 30 rounds,
                                      lambda in {0.1, 1, 10, 100}, 5 seeds
  pd_lite       45 cells    ~19h*    PD, 5 langs, 30 rounds, 3 scales, 3 seeds
  pd_rounds     120 cells   ~50h*    PD, 5 langs, [10,30] rounds, 4 scales, 3 seeds
  full          900 cells   ~375h*    everything (only realistic for 3B models)

  (* estimates assume Llama-8B at ~6.3s/call. Smaller=faster, larger=slower.)

Outputs land in /kaggle/working/results/. CSVs are keyed per (model, game,
lang, rounds, scale, seed) and skipped on rerun, so an interrupted 12h session
resumes seamlessly the next time the notebook is run.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & bootstrap
# ---------------------------------------------------------------------------

ON_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()
WORKDIR = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).resolve().parent / "kaggle_work"
FAIRGAME_REPO_URL = "https://github.com/aira-list/FAIRGAME.git"
FAIRGAME_DIR = WORKDIR / "FAIRGAME"
RESULTS_DIR = WORKDIR / "results"
RAW_DIR = RESULTS_DIR / "raw"


def _pip_install(*pkgs: str, upgrade: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    if upgrade:
        cmd.append("-U")
    subprocess.check_call(cmd + list(pkgs))


def bootstrap() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not FAIRGAME_DIR.exists():
        print(f"[bootstrap] cloning {FAIRGAME_REPO_URL} -> {FAIRGAME_DIR}")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", FAIRGAME_REPO_URL, str(FAIRGAME_DIR)]
        )
    if str(FAIRGAME_DIR) not in sys.path:
        sys.path.insert(0, str(FAIRGAME_DIR))

    # Light deps; transformers/bitsandbytes are usually present on Kaggle GPU images
    # but we pin minimum versions to be safe.
    _pip_install(
        "python-dotenv",
        "pandas",
        "transformers>=4.50,<5",   # Qwen3 / Gemma3 / Phi-4 chat templates
        "accelerate>=0.33",
        "bitsandbytes>=0.43",
        "sentencepiece",
        "protobuf",
        "kagglehub",               # for Gemma-4 / Nemotron via Kaggle model registry
        "striprtf",                # used by FileManager for cn/vn .rtf templates
        "retry",
    )


def _stub_unused_sdks() -> None:
    """Provide minimal stubs for SDKs FAIRGAME imports but we never call.

    FAIRGAME's llm_factory_connector unconditionally imports the anthropic /
    mistralai / openai connectors at module load, and each does `from <pkg>
    import <Class>`. Installing the real SDKs on Kaggle is fragile (different
    Kaggle images ship different pinned versions, and forcing an upgrade
    breaks other preinstalled tooling). Since we route every prompt through
    LocalHFConnector, those classes are never instantiated — a stub is enough
    to make the import chain succeed.
    """
    import types
    for pkg, cls_name in [
        ("anthropic", "Anthropic"),
        ("mistralai", "Mistral"),
        ("openai",    "OpenAI"),
    ]:
        try:
            mod = __import__(pkg)
            if hasattr(mod, cls_name):
                continue
        except Exception:
            mod = None
        if mod is None:
            mod = types.ModuleType(pkg)
            sys.modules[pkg] = mod
        # Stub class — FAIRGAME never instantiates it; if it ever does the
        # call site will get a clear NameError-style failure rather than
        # a silent network call.
        def _stub_init(self, *a, **k):
            raise RuntimeError(
                f"{cls_name} is a stub from run_kaggle_t4.py; commercial "
                "providers are not configured for this run."
            )
        setattr(mod, cls_name, type(cls_name, (), {"__init__": _stub_init}))


# ---------------------------------------------------------------------------
# Model registry & experiment presets — pick via CLI: --model NAME --preset NAME
# ---------------------------------------------------------------------------
# Each model entry supports:
#   source         "hf" (HuggingFace Hub) or "kagglehub" (Kaggle model registry)
#   path           HF repo id  OR  Kaggle model handle, depending on source
#   load_in_4bit   True = apply nf4 quant; False = use checkpoint as-is (already
#                  quantized, e.g. unsloth/*-bnb-4bit) or fp16 small model
#   is_reasoning   True if model emits <think>...</think> traces (R1-distill etc.)
#   sec_per_call   measured wall time per LLM call on 1x T4. Used only for the
#                  pre-run time estimate (not load behaviour). Adjust to taste.
#
# T4 (2x16GB) memory budget with nf4 4-bit + fp16 compute:
#   3B  ~2 GB    | 7-8B ~5 GB    | 12-14B ~9 GB    (all fit one T4)
#   24B ~14 GB   | 30-32B ~18 GB | (use both T4s via device_map="auto")
#
# Gated HF models (Llama / Gemma on HF Hub): set HF_TOKEN env var, OR use the
# unsloth bnb-4bit mirror, OR pull from `source: "kagglehub"`.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # --- Llama family (unsloth ungated 4-bit mirrors) ---
    "Llama3_2_3B":   dict(source="hf", path="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=3.0),
    "Llama3_1_8B":   dict(source="hf", path="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=6.3),

    # --- Qwen3 (April 2025) ---
    "Qwen3_8B":      dict(source="hf", path="Qwen/Qwen3-8B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=7.0),
    "Qwen3_14B":     dict(source="hf", path="Qwen/Qwen3-14B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=11.0),
    "Qwen3_32B":     dict(source="hf", path="Qwen/Qwen3-32B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=18.0),

    # --- Microsoft Phi-4 (Dec 2024) ---
    "Phi4_14B":      dict(source="hf", path="microsoft/phi-4",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=11.0),

    # --- Google Gemma-3 (Mar 2025) ---
    "Gemma3_4B":     dict(source="hf", path="google/gemma-3-4b-it",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=3.5),
    "Gemma3_12B":    dict(source="hf", path="google/gemma-3-12b-it",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=10.0),

    # --- DeepSeek-R1 distilled (Jan 2025) — reasoning model ---
    "R1_Qwen_14B":   dict(source="hf", path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                          load_in_4bit=True,  is_reasoning=True,  sec_per_call=22.0),

    # --- Mistral ---
    "Mistral_7B_v03": dict(source="hf", path="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                           load_in_4bit=False, is_reasoning=False, sec_per_call=6.0),

    # --- Kaggle-only sources (no HF token, no gating) ---
    "Gemma4_e2b":    dict(source="kagglehub",
                          path="google/gemma-4/transformers/gemma-4-e2b-it",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=2.5),
    "Nemotron3_30Ba3b": dict(source="kagglehub",
                             path="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
                             load_in_4bit=True, is_reasoning=False, sec_per_call=6.0),
}

ALL_GAMES = ["prisoner_dilemma", "stag_hunt", "snow_drift", "battle_sexes", "harmony_game"]
ALL_LANGS = ["en", "fr", "ar", "cn", "vn"]

# Each preset defines the full sweep dimensions. The driver iterates the
# Cartesian product. Cells already on disk are skipped, so an interrupted
# session resumes seamlessly on rerun.
PRESETS = {
    # ~8 min on 8B / 1xT4 — verify the pipeline end-to-end.
    "smoke":        dict(games=["prisoner_dilemma"], languages=["en"],
                         rounds=[10], payoff_scales=[1.0], seeds=[0]),

    # Quick statistical preview: paper baseline + extreme stake at 2 round levels.
    # 5 games * 1 lang * 2 rounds * 2 scales * 2 seeds = 40 cells.
    "mini":         dict(games=ALL_GAMES, languages=["en"],
                         rounds=[10, 50], payoff_scales=[1.0, 100.0], seeds=[0, 1]),

    # CORE PAPER CLAIM: payoff-scale effect at the "more rounds" condition.
    # 5 games * 1 lang * 1 rounds * 4 scales * 3 seeds = 60 cells.
    "scale":        dict(games=ALL_GAMES, languages=["en"],
                         rounds=[50], payoff_scales=[1.0, 5.0, 10.0, 100.0],
                         seeds=[0, 1, 2]),

    # Round-length ablation: 10 (paper baseline) vs 30 vs 50 at scale=1.
    # 5 games * 1 lang * 3 rounds * 1 scale * 3 seeds = 45 cells.
    "rounds":       dict(games=ALL_GAMES, languages=["en"],
                         rounds=[10, 30, 50], payoff_scales=[1.0],
                         seeds=[0, 1, 2]),

    # Multilingual robustness at extreme stakes.
    # 5 games * 5 langs * 1 rounds * 2 scales * 1 seed = 50 cells.
    "multilingual": dict(games=ALL_GAMES, languages=ALL_LANGS,
                         rounds=[50], payoff_scales=[1.0, 100.0], seeds=[0]),

    # EXACT REPLICATION of Buscemi et al. + this paper's published setup:
    # PD only, 5 languages, lambda in {0.1, 1.0, 10.0}, 10 rounds, 10 repeats.
    # 1 * 5 * 1 * 3 * 10 = 150 cells, ~21h on 8B (2 Kaggle sessions).
    # Use this to confirm whether the paper's commercial-LLM findings
    # replicate on open-weight models.
    "pd_replication": dict(games=["prisoner_dilemma"], languages=ALL_LANGS,
                           rounds=[10], payoff_scales=[0.1, 1.0, 10.0],
                           seeds=list(range(10))),

    # CORE EXTENSION of paper. Adds super-amplified stakes (lambda=100) and
    # the "more rounds" condition (30) on top of the paper's 5 langs.
    # 5 seeds for tight 95% CIs (paper used 10; we trade a bit of CI tightness
    # for the extra dimensions). 1 * 5 * 1 * 4 * 5 = 100 cells, ~42h on 8B.
    "pd_paper":     dict(games=["prisoner_dilemma"], languages=ALL_LANGS,
                         rounds=[30], payoff_scales=[0.1, 1.0, 10.0, 100.0],
                         seeds=[0, 1, 2, 3, 4]),

    # Lighter PD-focused for a 2-session model: 3 scales x 3 seeds. ~19h on 8B.
    "pd_lite":      dict(games=["prisoner_dilemma"], languages=ALL_LANGS,
                         rounds=[30], payoff_scales=[0.1, 1.0, 10.0],
                         seeds=[0, 1, 2]),

    # PD with both round conditions — lets us plot the round-length effect.
    # 1 game * 5 langs * 2 rounds * 4 scales * 3 seeds = 120 cells, ~50h on 8B.
    "pd_rounds":    dict(games=["prisoner_dilemma"], languages=ALL_LANGS,
                         rounds=[10, 30], payoff_scales=[0.1, 1.0, 10.0, 100.0],
                         seeds=[0, 1, 2]),

    # Everything. Only realistic for tiny models (3B). 900 cells.
    "full":         dict(games=ALL_GAMES, languages=ALL_LANGS,
                         rounds=[10, 30, 50], payoff_scales=[1.0, 5.0, 10.0, 100.0],
                         seeds=[0, 1, 2]),
}

# Built per-run from CLI args + selected preset & model. main() populates this.
EXPERIMENT_CONFIG: dict = {
    "max_new_tokens": 64,   # answer is "OptionA"/"OptionB"; 64 is plenty buffer
    "temperature": 1.0,
}

# Map our short game keys to (config filename, template prefix)
GAME_CONFIG_FILES = {
    "prisoner_dilemma": "prisoner_dilemma_nocomm_round_known_mild.json",
    "stag_hunt":        "stag_hunt_nocomm_round_known_conventional.json",
    "snow_drift":       "snow_drift_nocomm_round_known_conventional.json",
    "battle_sexes":     "battle_sexes_nocomm_round_known_conventional.json",
    "harmony_game":     "harmony_game_nocomm_round_known_conventional.json",
}


# ---------------------------------------------------------------------------
# Local HuggingFace connector — patches into FAIRGAME's factory
# ---------------------------------------------------------------------------

_THINK_RE = None  # compiled lazily


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> traces emitted by R1-distill style reasoning models.

    Returns the post-think payload only. If the close tag is missing the model
    ran out of budget while thinking — return whatever follows a sentinel or
    an empty string so the FAIRGAME parser does not pick the trace as a choice.
    """
    global _THINK_RE
    if _THINK_RE is None:
        import re
        _THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
    cleaned = _THINK_RE.sub("", text)
    # If a stray opening <think> remains (no close), keep only what follows it.
    if "<think" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):] if idx >= 0 else ""
    return cleaned.strip()


class LocalHFConnector:
    """LLM connector that runs a HuggingFace / kagglehub causal LM on the local GPU.

    Plugs into FAIRGAME via `MODEL_PROVIDER_MAP[name] = (LocalHFConnector, name)`.
    A class-level registry holds the full model_cfg so the connector can resolve
    source / quantization / reasoning flags from just the short name. Models are
    loaded lazily on first prompt and cached, so a sweep for one model reuses
    the same weights across hundreds of game runs.
    """

    _registry: dict = {}      # short_name -> model_cfg dict
    _model_cache: dict = {}   # short_name -> (tokenizer, model)
    DEFAULT_MAX_NEW_TOKENS = EXPERIMENT_CONFIG["max_new_tokens"]
    REASONING_MAX_NEW_TOKENS = 1024   # give R1-distill room for its trace
    DEFAULT_TEMPERATURE = EXPERIMENT_CONFIG["temperature"]

    @classmethod
    def register(cls, model_cfg: dict) -> None:
        cls._registry[model_cfg["name"]] = model_cfg

    def __init__(self, provider_model: str, temperature: float | None = None):
        # provider_model is the short tag we registered under
        self.short_name = provider_model
        cfg = self._registry.get(provider_model, {})
        self.cfg = cfg
        self.is_reasoning = bool(cfg.get("is_reasoning", False))
        self.temperature = self.DEFAULT_TEMPERATURE if temperature is None else temperature
        self.max_new_tokens = (
            self.REASONING_MAX_NEW_TOKENS if self.is_reasoning else self.DEFAULT_MAX_NEW_TOKENS
        )

    @classmethod
    def _resolve_local_path(cls, cfg: dict) -> str:
        """Return a local filesystem path or HF repo id ready for from_pretrained."""
        src = cfg.get("source", "hf")
        path = cfg["path"]
        if src == "kagglehub":
            import kagglehub
            return kagglehub.model_download(path)
        return path  # HF repo id; from_pretrained handles the download + cache

    @classmethod
    def load(cls, short_name: str):
        if short_name in cls._model_cache:
            return cls._model_cache[short_name]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = cls._registry[short_name]
        load_in_4bit = bool(cfg.get("load_in_4bit", True))
        model_path = cls._resolve_local_path(cfg)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        print(f"[hf] loading {short_name}  src={cfg.get('source','hf')}  4bit={load_in_4bit}  path={cfg['path']}")
        t0 = time.time()
        bnb = None
        if load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,   # T4 has no native bf16
                bnb_4bit_use_double_quant=True,
            )
        tok_kw = {"trust_remote_code": True}
        if hf_token and cfg.get("source", "hf") == "hf":
            tok_kw["token"] = hf_token
        tok = AutoTokenizer.from_pretrained(model_path, **tok_kw)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # Build kwargs carefully: passing quantization_config=None to recent
        # transformers (>=4.50) overrides the checkpoint's saved quant config
        # and then crashes during config.__repr__ on `quant_cfg.to_dict()`.
        # Only pass it when we have a real BitsAndBytesConfig; otherwise let
        # transformers auto-detect from the (possibly pre-quantized) checkpoint.
        mdl_kw = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        # transformers 4.50+ renamed torch_dtype -> dtype (old name still works
        # but emits a deprecation warning and may eventually break).
        try:
            import transformers as _tf
            major, minor = (int(x) for x in _tf.__version__.split(".")[:2])
            mdl_kw["dtype" if (major, minor) >= (4, 50) else "torch_dtype"] = torch.float16
        except Exception:
            mdl_kw["torch_dtype"] = torch.float16
        if bnb is not None:
            mdl_kw["quantization_config"] = bnb
        if hf_token and cfg.get("source", "hf") == "hf":
            mdl_kw["token"] = hf_token
        mdl = AutoModelForCausalLM.from_pretrained(model_path, **mdl_kw)
        mdl.eval()
        cls._model_cache[short_name] = (tok, mdl)
        print(f"[hf] loaded {short_name} in {time.time() - t0:.1f}s")
        return cls._model_cache[short_name]

    @classmethod
    def free(cls, short_name: str | None = None) -> None:
        import torch
        names = [short_name] if short_name else list(cls._model_cache.keys())
        for n in names:
            cls._model_cache.pop(n, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[hf] freed {names}")

    def send_prompt(self, prompt: str) -> str:
        import torch
        tok, mdl = self.load(self.short_name)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(mdl.device)
        with torch.inference_mode():
            out = mdl.generate(
                ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if self.is_reasoning:
            text = _strip_thinking(text)
        return text


def register_local_models() -> None:
    """Inject our local connectors into FAIRGAME's provider map."""
    _stub_unused_sdks()  # must run before any `src.*` import
    from src.llm_connectors import llm_factory_connector as fc
    for m in EXPERIMENT_CONFIG["models"]:
        LocalHFConnector.register(m)
        fc.MODEL_PROVIDER_MAP[m["name"]] = (LocalHFConnector, m["name"])
    print("[patch] MODEL_PROVIDER_MAP now includes:",
          [m["name"] for m in EXPERIMENT_CONFIG["models"]])


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def build_base_config(game: str, lang: str) -> dict:
    """Load the canonical FAIRGAME config for `game` and normalize types."""
    cfg_path = FAIRGAME_DIR / "resources" / "config" / GAME_CONFIG_FILES[game]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # JSON in repo uses "True"/"False" strings; validator wants Python bool
    cfg["nRoundsIsKnown"] = _coerce_bool(cfg.get("nRoundsIsKnown", True))
    cfg["allAgentPermutations"] = _coerce_bool(cfg.get("allAgentPermutations", True))
    cfg["agentsCommunicate"] = _coerce_bool(cfg.get("agentsCommunicate", False))
    cfg["languages"] = [lang]
    # Drop the plural "llms" used by some configs — we set singular "llm" per run
    cfg.pop("llms", None)
    return cfg


def scale_payoff(cfg: dict, scale: float) -> dict:
    """Return a deep copy of cfg with payoff weights multiplied by `scale`."""
    out = copy.deepcopy(cfg)
    weights = out["payoffMatrix"]["weights"]
    for k in list(weights.keys()):
        weights[k] = float(weights[k]) * float(scale)
    return out


def attach_template(cfg: dict, game: str, lang: str) -> dict:
    """Inline the prompt template into the config so FAIRGAME doesn't try to read disk."""
    tpl_path = FAIRGAME_DIR / "resources" / "game_templates" / f"{game}_{lang}.txt"
    cfg = copy.deepcopy(cfg)
    cfg["promptTemplate"] = {lang: tpl_path.read_text(encoding="utf-8")}
    return cfg


# ---------------------------------------------------------------------------
# Single experiment cell
# ---------------------------------------------------------------------------

def run_single(model_cfg, game, lang, rounds, scale, seed) -> Path | None:
    name = (
        f"{model_cfg['name']}__{game}__{lang}__r{rounds}__s{scale:g}__seed{seed}"
    )
    out_csv = RAW_DIR / f"{name}.csv"
    if out_csv.exists():
        print(f"[skip] {name}")
        return out_csv

    # Seed RNGs so seed actually has an effect (transformers samples from CUDA RNG)
    import torch
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = build_base_config(game, lang)
    base["nRounds"] = int(rounds)
    base["llm"] = model_cfg["name"]
    cfg = scale_payoff(base, scale)
    cfg = attach_template(cfg, game, lang)

    from src.fairgame_factory import FairGameFactory
    from src.results_processing.results_processor import ResultsProcessor

    print(f"[run ] {name}")
    t0 = time.time()
    factory = FairGameFactory()
    results = factory.create_and_run_games(cfg)
    df = ResultsProcessor().process(results)
    df["meta_model"] = model_cfg["name"]
    df["meta_source"] = model_cfg.get("source", "hf")
    df["meta_path"] = model_cfg["path"]
    df["meta_is_reasoning"] = bool(model_cfg.get("is_reasoning", False))
    df["meta_game"] = game
    df["meta_language"] = lang
    df["meta_rounds"] = rounds
    df["meta_payoff_scale"] = scale
    df["meta_seed"] = seed
    df.to_csv(out_csv, index=False, sep=";")
    print(f"[done] {name}  ({time.time() - t0:.1f}s, rows={len(df)})")
    return out_csv


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def estimate_workload() -> int:
    cfg = EXPERIMENT_CONFIG
    return (
        len(cfg["models"])
        * len(cfg["games"])
        * len(cfg["languages"])
        * len(cfg["rounds"])
        * len(cfg["payoff_scales"])
        * len(cfg["seeds"])
    )


def aggregate() -> None:
    import pandas as pd
    parts = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        try:
            parts.append(pd.read_csv(p, sep=";"))
        except Exception as e:
            print(f"[agg ] skip {p.name}: {e}")
    if not parts:
        print("[agg ] no result files yet")
        return
    df = pd.concat(parts, ignore_index=True)
    out = RESULTS_DIR / "summary.csv"
    df.to_csv(out, index=False, sep=";")
    print(f"[agg ] {len(parts)} files -> {out}  (rows={len(df)})")


def estimate_hours(model_cfg: dict, preset: dict) -> tuple[int, float]:
    """Return (cells, hours) estimate for the given model x preset sweep."""
    cells = (
        len(preset["games"])
        * len(preset["languages"])
        * len(preset["rounds"])
        * len(preset["payoff_scales"])
        * len(preset["seeds"])
    )
    # Each cell runs 4 personality permutations x rounds x 2 agent calls.
    # Use mean rounds across the round-list to weight cell duration.
    avg_rounds = sum(preset["rounds"]) / len(preset["rounds"])
    calls_per_cell = 8.0 * avg_rounds
    sec_per_call = float(model_cfg.get("sec_per_call", 6.5))
    sec = cells * calls_per_cell * sec_per_call
    return cells, sec / 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAIRGAME More-at-Stake Kaggle T4 sweep runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=os.environ.get("MODEL", "Llama3_1_8B"),
        choices=sorted(MODEL_REGISTRY.keys()),
        help="One model from MODEL_REGISTRY. Run a separate notebook per model.",
    )
    parser.add_argument(
        "--preset", default=os.environ.get("PRESET", "smoke"),
        choices=sorted(PRESETS.keys()),
        help="Sweep preset. Use 'smoke' first to verify, then 'scale' for the "
             "paper core claim or 'mini' for a quick statistical preview.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", 64)),
        help="Generation cap. 64 is plenty for OptionA/OptionB answers.",
    )
    parser.add_argument(
        "--temperature", type=float,
        default=float(os.environ.get("TEMPERATURE", 1.0)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap()

    model_cfg = dict(MODEL_REGISTRY[args.model], name=args.model)
    preset = PRESETS[args.preset]

    EXPERIMENT_CONFIG.update(
        models=[model_cfg],
        games=preset["games"],
        languages=preset["languages"],
        rounds=preset["rounds"],
        payoff_scales=preset["payoff_scales"],
        seeds=preset["seeds"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    LocalHFConnector.DEFAULT_MAX_NEW_TOKENS = args.max_new_tokens
    LocalHFConnector.DEFAULT_TEMPERATURE = args.temperature

    register_local_models()

    cells, hours = estimate_hours(model_cfg, preset)
    print(
        f"\n=== FAIRGAME More-at-Stake sweep ===\n"
        f"  model        : {args.model}  ({model_cfg['path']})\n"
        f"  preset       : {args.preset}\n"
        f"  games        : {preset['games']}\n"
        f"  languages    : {preset['languages']}\n"
        f"  rounds       : {preset['rounds']}\n"
        f"  payoff_scale : {preset['payoff_scales']}\n"
        f"  seeds        : {preset['seeds']}\n"
        f"  cells        : {cells}  (resume-safe — finished CSVs are skipped)\n"
        f"  est. wall    : ~{hours:.1f} h  (sec_per_call={model_cfg.get('sec_per_call', 6.5)})\n"
        f"  output dir   : {RAW_DIR}\n"
    )
    if hours > 11.5 and args.preset != "smoke":
        print(f"  WARNING: estimate exceeds Kaggle's 12h limit. Plan multiple\n"
              f"           sessions; the sweep will resume from where it stops.\n")

    m = model_cfg
    print(f">>> MODEL: {m['name']}  src={m.get('source','hf')}  path={m['path']}")
    try:
        LocalHFConnector.load(m["name"])
    except Exception as e:
        print(f"[error] failed to load {m['name']}: {e}")
        traceback.print_exc()
        return

    for game in preset["games"]:
        for lang in preset["languages"]:
            for rounds in preset["rounds"]:
                for scale in preset["payoff_scales"]:
                    for seed in preset["seeds"]:
                        try:
                            run_single(m, game, lang, rounds, scale, seed)
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            print(f"[error] {m['name']} {game} {lang} r={rounds} s={scale} seed={seed}: {e}")
                            traceback.print_exc()
        # Aggregate progressively so partial results are usable mid-session.
        aggregate()

    LocalHFConnector.free(m["name"])
    aggregate()
    print("\n=== sweep done ===")


if __name__ == "__main__":
    main()
