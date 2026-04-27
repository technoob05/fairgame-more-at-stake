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
  - Create a new notebook, GPU T4 x2, Internet ON.
  - Single cell:
        !wget -q https://raw.githubusercontent.com/technoob05/fairgame-more-at-stake/main/exp/run_kaggle_t4.py
        !python run_kaggle_t4.py
  - Outputs land in /kaggle/working/results/.

Tweak EXPERIMENT_CONFIG below to match your session budget. The script prints
an estimate before running and skips any (model, game, lang, rounds, scale, seed)
whose CSV already exists, so a 12h timeout is safe to resume from.
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


def _pip_install(*pkgs: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs]
    )


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
        "transformers>=4.45,<5",
        "accelerate>=0.33",
        "bitsandbytes>=0.43",
        "sentencepiece",
        "protobuf",
    )


# ---------------------------------------------------------------------------
# Experiment matrix — edit this block to scale up/down
# ---------------------------------------------------------------------------
# Notes on Kaggle T4 (2x16GB) memory budget with nf4 4-bit:
#   Qwen2.5-7B-Instruct  ~5 GB    (fits 1 GPU)
#   Qwen2.5-14B-Instruct ~9 GB    (fits 1 GPU)
#   Qwen2.5-32B-Instruct ~18 GB   (uses both GPUs via device_map="auto")
# A 12h Kaggle session can typically finish ~600-1200 short games depending
# on output token length. Trim `seeds` or `payoff_scales` if you run out.
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG = {
    "models": [
        {"name": "Qwen2_5_7B",  "hf_id": "Qwen/Qwen2.5-7B-Instruct",  "load_in_4bit": True},
        {"name": "Qwen2_5_14B", "hf_id": "Qwen/Qwen2.5-14B-Instruct", "load_in_4bit": True},
        {"name": "Qwen2_5_32B", "hf_id": "Qwen/Qwen2.5-32B-Instruct", "load_in_4bit": True},
    ],
    "games": [
        "prisoner_dilemma",
        "stag_hunt",
        "snow_drift",
        "battle_sexes",
        "harmony_game",
    ],
    "languages": ["en"],          # add "fr","ar","cn","vn" to study language effects
    "rounds": [10, 30, 50],       # 10 = paper baseline; 30/50 = "more rounds" condition
    "payoff_scales": [1.0, 5.0, 10.0, 100.0],  # multiplies all weight* in payoffMatrix
    "seeds": [0, 1, 2],           # repeats; each uses sampling so trajectories differ
    "max_new_tokens": 256,
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

class LocalHFConnector:
    """LLM connector that runs a HuggingFace causal LM on the local GPU.

    Plugs into FAIRGAME via `MODEL_PROVIDER_MAP[name] = (LocalHFConnector, hf_id)`.
    The model is loaded lazily on first prompt and cached as a class-level singleton
    keyed by hf_id, so the entire experiment sweep for one model reuses one load.
    """

    _model_cache: dict = {}   # hf_id -> (tokenizer, model)
    DEFAULT_MAX_NEW_TOKENS = EXPERIMENT_CONFIG["max_new_tokens"]
    DEFAULT_TEMPERATURE = EXPERIMENT_CONFIG["temperature"]

    def __init__(self, provider_model: str, temperature: float | None = None):
        self.hf_id = provider_model
        self.temperature = self.DEFAULT_TEMPERATURE if temperature is None else temperature
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS

    @classmethod
    def load(cls, hf_id: str, load_in_4bit: bool = True):
        if hf_id in cls._model_cache:
            return cls._model_cache[hf_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[hf] loading {hf_id} (4bit={load_in_4bit})")
        t0 = time.time()
        bnb = None
        if load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        mdl.eval()
        cls._model_cache[hf_id] = (tok, mdl)
        print(f"[hf] loaded {hf_id} in {time.time() - t0:.1f}s")
        return cls._model_cache[hf_id]

    @classmethod
    def free(cls, hf_id: str | None = None) -> None:
        import torch
        ids = [hf_id] if hf_id else list(cls._model_cache.keys())
        for i in ids:
            cls._model_cache.pop(i, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[hf] freed {ids}")

    def send_prompt(self, prompt: str) -> str:
        import torch
        tok, mdl = self.load(self.hf_id, load_in_4bit=True)
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
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def register_local_models() -> None:
    """Inject our local connectors into FAIRGAME's provider map."""
    from src.llm_connectors import llm_factory_connector as fc
    for m in EXPERIMENT_CONFIG["models"]:
        fc.MODEL_PROVIDER_MAP[m["name"]] = (LocalHFConnector, m["hf_id"])
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
    df["meta_hf_id"] = model_cfg["hf_id"]
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


def main() -> None:
    bootstrap()
    register_local_models()

    total = estimate_workload()
    print(
        f"\n=== FAIRGAME More-at-Stake sweep ===\n"
        f"  models       : {[m['name'] for m in EXPERIMENT_CONFIG['models']]}\n"
        f"  games        : {EXPERIMENT_CONFIG['games']}\n"
        f"  languages    : {EXPERIMENT_CONFIG['languages']}\n"
        f"  rounds       : {EXPERIMENT_CONFIG['rounds']}\n"
        f"  payoff_scale : {EXPERIMENT_CONFIG['payoff_scales']}\n"
        f"  seeds        : {EXPERIMENT_CONFIG['seeds']}\n"
        f"  total cells  : {total}  (resume-safe)\n"
        f"  output dir   : {RAW_DIR}\n"
    )

    # Outer loop = model so we load each weight set exactly once.
    for m in EXPERIMENT_CONFIG["models"]:
        print(f"\n>>> MODEL: {m['name']} ({m['hf_id']})")
        try:
            LocalHFConnector.load(m["hf_id"], load_in_4bit=m.get("load_in_4bit", True))
        except Exception as e:
            print(f"[error] failed to load {m['hf_id']}: {e}")
            traceback.print_exc()
            continue

        for game in EXPERIMENT_CONFIG["games"]:
            for lang in EXPERIMENT_CONFIG["languages"]:
                for rounds in EXPERIMENT_CONFIG["rounds"]:
                    for scale in EXPERIMENT_CONFIG["payoff_scales"]:
                        for seed in EXPERIMENT_CONFIG["seeds"]:
                            try:
                                run_single(m, game, lang, rounds, scale, seed)
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                print(f"[error] {m['name']} {game} {lang} r={rounds} s={scale} seed={seed}: {e}")
                                traceback.print_exc()
            # Aggregate progressively so partial results are usable mid-session.
            aggregate()

        LocalHFConnector.free(m["hf_id"])

    aggregate()
    print("\n=== sweep done ===")


if __name__ == "__main__":
    main()
