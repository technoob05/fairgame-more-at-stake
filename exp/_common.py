"""
FAIRGAME More-at-Stake — shared experiment infrastructure
=========================================================

This module is imported by the per-experiment driver files in this folder
(`exp_<MODEL>__s<SCALE>.py`). Each driver pins one (model, scale) pair and
forwards everything else as kwargs to `run_experiment(...)`. That lets each
Kaggle session run a single 7-12h slice of the sweep, while the heavy
lifting — FAIRGAME clone, bnb 4-bit loader, reasoning-token strip, stub of
unused commercial-provider SDKs — lives here in one place.

Public API:
    MODEL_REGISTRY                 dict of supported short_name -> model spec
    run_experiment(model, game, scale, languages, rounds, seeds, ...)
    aggregate()                     concat all per-cell CSVs into summary.csv
"""

from __future__ import annotations

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
    _pip_install(
        "python-dotenv", "pandas",
        "transformers>=4.50,<5", "accelerate>=0.33", "bitsandbytes>=0.43",
        "sentencepiece", "protobuf", "kagglehub", "striprtf", "retry",
    )


def _stub_unused_sdks() -> None:
    """FAIRGAME's llm_factory_connector unconditionally imports anthropic /
    mistralai / openai connectors at module load. We never call those, so a
    stub is enough — and avoids the version conflicts that real installs
    cause on Kaggle's preinstalled stack.
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

        def _stub_init(self, *a, **k):
            raise RuntimeError(
                f"{cls_name} is a stub; commercial providers are not "
                "configured for this run."
            )
        setattr(mod, cls_name, type(cls_name, (), {"__init__": _stub_init}))


# ---------------------------------------------------------------------------
# Model registry  — every model the sweep can target
# ---------------------------------------------------------------------------
# `sec_per_call` is empirical wall-clock per LLM call on 1x T4 with the
# given quantization; used only for the pre-run time estimate.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict = {
    # Llama family — unsloth ungated 4-bit mirrors (no HF_TOKEN needed).
    "Llama3_2_3B":   dict(source="hf", path="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=3.0),
    "Llama3_1_8B":   dict(source="hf", path="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=6.3),

    # Qwen3 (April 2025).
    "Qwen3_8B":      dict(source="hf", path="Qwen/Qwen3-8B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=7.0),
    "Qwen3_14B":     dict(source="hf", path="Qwen/Qwen3-14B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=11.0),
    "Qwen3_32B":     dict(source="hf", path="Qwen/Qwen3-32B",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=18.0),

    # Microsoft Phi-4 (Dec 2024) — 14B dense, MIT.
    "Phi4_14B":      dict(source="hf", path="microsoft/phi-4",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=11.0),

    # Google Gemma-3 (Mar 2025).
    "Gemma3_4B":     dict(source="hf", path="google/gemma-3-4b-it",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=3.5),
    "Gemma3_12B":    dict(source="hf", path="google/gemma-3-12b-it",
                          load_in_4bit=True,  is_reasoning=False, sec_per_call=10.0),

    # DeepSeek-R1 distilled (Jan 2025) — emits <think>...</think> reasoning trace.
    "R1_Qwen_14B":   dict(source="hf", path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                          load_in_4bit=True,  is_reasoning=True,  sec_per_call=22.0),

    # Mistral.
    "Mistral_7B_v03": dict(source="hf", path="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                           load_in_4bit=False, is_reasoning=False, sec_per_call=6.0),

    # Kaggle-only sources (no HF_TOKEN, no gating).
    "Gemma4_e2b":    dict(source="kagglehub",
                          path="google/gemma-4/transformers/gemma-4-e2b-it",
                          load_in_4bit=False, is_reasoning=False, sec_per_call=2.5),
    "Nemotron3_30Ba3b": dict(source="kagglehub",
                             path="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
                             load_in_4bit=True, is_reasoning=False, sec_per_call=6.0),
}

# PD points to *round_not_known* to match the paper-released dataset
# (data_fairgame/*/* CSVs all have nRoundsIsKnown=False). Without this the
# horizon is finite-and-known, agents backward-induct, and you lose all
# cooperation by Round 1 — utterly different game-theoretic regime.
# The other games stick with their 'round_known' variant since the paper
# does not release equivalent datasets for them.
GAME_CONFIG_FILES = {
    "prisoner_dilemma": "prisoner_dilemma_nocomm_round_not_known_mild.json",
    "stag_hunt":        "stag_hunt_nocomm_round_known_conventional.json",
    "snow_drift":       "snow_drift_nocomm_round_known_conventional.json",
    "battle_sexes":     "battle_sexes_nocomm_round_known_conventional.json",
    "harmony_game":     "harmony_game_nocomm_round_known_conventional.json",
}


# ---------------------------------------------------------------------------
# LLM connector — local HuggingFace / kagglehub model, monkey-patched into
# FAIRGAME's MODEL_PROVIDER_MAP at runtime
# ---------------------------------------------------------------------------

_THINK_RE = None  # compiled lazily


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> traces emitted by R1-distill style models.

    Returns post-think payload only. If the close tag is missing the model
    ran out of budget while thinking — return whatever follows the last
    closing tag, or empty string if none, so the FAIRGAME parser does not
    pick the trace as a choice.
    """
    global _THINK_RE
    if _THINK_RE is None:
        import re
        _THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
    cleaned = _THINK_RE.sub("", text)
    if "<think" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):] if idx >= 0 else ""
    return cleaned.strip()


class LocalHFConnector:
    """LLM connector that runs a HuggingFace / kagglehub causal LM on the local GPU.

    Plugs into FAIRGAME via `MODEL_PROVIDER_MAP[name] = (LocalHFConnector, name)`.
    A class-level registry holds the full model_cfg so the connector can
    resolve source / quantization / reasoning flags from just the short name.
    Models load lazily on first prompt and stay cached for the rest of the
    sweep (one model per Kaggle session, by design).
    """

    _registry: dict = {}      # short_name -> model_cfg dict
    _model_cache: dict = {}   # short_name -> (tokenizer, model)
    DEFAULT_MAX_NEW_TOKENS = 64
    REASONING_MAX_NEW_TOKENS = 1024
    DEFAULT_TEMPERATURE = 1.0

    @classmethod
    def register(cls, model_cfg: dict) -> None:
        cls._registry[model_cfg["name"]] = model_cfg

    def __init__(self, provider_model: str, temperature: float | None = None):
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
        src = cfg.get("source", "hf")
        path = cfg["path"]
        if src == "kagglehub":
            import kagglehub
            return kagglehub.model_download(path)
        return path

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

        print(f"[hf] loading {short_name}  src={cfg.get('source','hf')}  "
              f"4bit={load_in_4bit}  path={cfg['path']}")
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

        # Only pass quantization_config when we actually have one. Passing
        # None to transformers >= 4.50 overrides the checkpoint's saved quant
        # config and crashes during config.__repr__ on `.to_dict()`.
        mdl_kw = {"device_map": "auto", "trust_remote_code": True}
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


def register_local_model(model_cfg: dict) -> None:
    """Inject one local connector into FAIRGAME's provider map."""
    _stub_unused_sdks()
    from src.llm_connectors import llm_factory_connector as fc
    LocalHFConnector.register(model_cfg)
    fc.MODEL_PROVIDER_MAP[model_cfg["name"]] = (LocalHFConnector, model_cfg["name"])


# ---------------------------------------------------------------------------
# Config builder + per-cell runner
# ---------------------------------------------------------------------------

def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _build_base_config(game: str, lang: str) -> dict:
    cfg_path = FAIRGAME_DIR / "resources" / "config" / GAME_CONFIG_FILES[game]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["nRoundsIsKnown"]      = _coerce_bool(cfg.get("nRoundsIsKnown", True))
    cfg["allAgentPermutations"] = _coerce_bool(cfg.get("allAgentPermutations", True))
    cfg["agentsCommunicate"]   = _coerce_bool(cfg.get("agentsCommunicate", False))
    cfg["languages"] = [lang]
    cfg.pop("llms", None)
    return cfg


def _scale_payoff(cfg: dict, scale: float) -> dict:
    out = copy.deepcopy(cfg)
    weights = out["payoffMatrix"]["weights"]
    for k in list(weights.keys()):
        weights[k] = float(weights[k]) * float(scale)
    return out


def _attach_template(cfg: dict, game: str, lang: str) -> dict:
    tpl_path = FAIRGAME_DIR / "resources" / "game_templates" / f"{game}_{lang}.txt"
    cfg = copy.deepcopy(cfg)
    cfg["promptTemplate"] = {lang: tpl_path.read_text(encoding="utf-8")}
    return cfg


def _run_single(model_cfg, game, lang, rounds, scale, seed) -> Path | None:
    name = f"{model_cfg['name']}__{game}__{lang}__r{rounds}__s{scale:g}__seed{seed}"
    out_csv = RAW_DIR / f"{name}.csv"
    if out_csv.exists():
        print(f"[skip] {name}")
        return out_csv

    import torch, random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = _build_base_config(game, lang)
    base["nRounds"] = int(rounds)
    base["llm"] = model_cfg["name"]
    cfg = _scale_payoff(base, scale)
    cfg = _attach_template(cfg, game, lang)

    from src.fairgame_factory import FairGameFactory
    from src.results_processing.results_processor import ResultsProcessor

    print(f"[run ] {name}")
    t0 = time.time()
    factory = FairGameFactory()
    results = factory.create_and_run_games(cfg)
    df = ResultsProcessor().process(results)
    df["meta_model"]         = model_cfg["name"]
    df["meta_source"]        = model_cfg.get("source", "hf")
    df["meta_path"]          = model_cfg["path"]
    df["meta_is_reasoning"]  = bool(model_cfg.get("is_reasoning", False))
    df["meta_game"]          = game
    df["meta_language"]      = lang
    df["meta_rounds"]        = rounds
    df["meta_payoff_scale"]  = scale
    df["meta_seed"]          = seed
    df.to_csv(out_csv, index=False, sep=";")
    print(f"[done] {name}  ({time.time() - t0:.1f}s, rows={len(df)})")
    return out_csv


def aggregate() -> None:
    """Concatenate all per-cell CSVs in RAW_DIR into one summary.csv."""
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


def zip_results(tag: str) -> Path | None:
    """Pack RESULTS_DIR into a single zip in WORKDIR for easy Kaggle download.

    The zip lands directly in /kaggle/working/results_<tag>.zip so it shows up
    in the Kaggle notebook's "Output" panel and can be downloaded with one click.
    """
    import shutil
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("[zip ] nothing to zip yet")
        return None
    base = WORKDIR / f"results_{tag}"
    zip_path = base.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(base), "zip", root_dir=str(RESULTS_DIR))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[zip ] {zip_path}  ({size_mb:.2f} MB)")
    return zip_path


# ---------------------------------------------------------------------------
# Public entrypoint — called by exp_*.py drivers
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    model: str,
    scale: float,
    game: str = "prisoner_dilemma",
    languages: list = ("en", "fr", "ar", "cn", "vn"),
    rounds: int = 30,
    seeds: list = (0, 1, 2, 3, 4),
    max_new_tokens: int = 64,
    temperature: float = 1.0,
) -> None:
    """Run the (model, scale) slice across the given languages x seeds.

    Defaults match Strategy B: PD only, 5 paper languages, 30 rounds, 5 seeds.
    Each call writes 1 CSV per (lang, seed) and skips files already on disk
    so a Kaggle session interruption is safe to resume by rerunning the cell.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}; pick one of {sorted(MODEL_REGISTRY)}")
    bootstrap()

    model_cfg = dict(MODEL_REGISTRY[model], name=model)
    LocalHFConnector.DEFAULT_MAX_NEW_TOKENS = int(max_new_tokens)
    LocalHFConnector.DEFAULT_TEMPERATURE   = float(temperature)
    register_local_model(model_cfg)

    cells = len(languages) * len(seeds)
    calls_per_cell = 8 * int(rounds)
    sec = cells * calls_per_cell * float(model_cfg["sec_per_call"])
    print(
        f"\n=== FAIRGAME slice: {model} @ scale={scale} ===\n"
        f"  game         : {game}\n"
        f"  languages    : {list(languages)}\n"
        f"  rounds       : {rounds}\n"
        f"  seeds        : {list(seeds)}\n"
        f"  cells        : {cells}  (resume-safe — finished CSVs are skipped)\n"
        f"  est. wall    : ~{sec/3600:.1f} h  (sec/call ~ {model_cfg['sec_per_call']:.1f})\n"
        f"  output dir   : {RAW_DIR}\n"
    )

    print(f">>> MODEL: {model}  src={model_cfg.get('source','hf')}  path={model_cfg['path']}")
    try:
        LocalHFConnector.load(model)
    except Exception as e:
        print(f"[error] failed to load {model}: {e}")
        traceback.print_exc()
        return

    for lang in languages:
        for seed in seeds:
            try:
                _run_single(model_cfg, game, lang, rounds, scale, seed)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[error] {model} {game} {lang} r={rounds} s={scale} seed={seed}: {e}")
                traceback.print_exc()
        aggregate()  # progressive snapshot

    LocalHFConnector.free(model)
    aggregate()
    zip_results(f"{model}__s{scale:g}")
    print("\n=== slice done ===")
