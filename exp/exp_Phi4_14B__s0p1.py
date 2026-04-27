"""
FAIRGAME slice: Phi4_14B @ payoff_scale=0.1.

One Kaggle session, one (model, scale) slice. Resume-safe: rerunning the
notebook skips any (lang, seed) cell whose CSV already exists.

How to run on Kaggle:
  EITHER paste this whole file into a notebook cell and run it
         (the bootstrap below auto-clones the repo for `_common.py`),
  OR use a 2-line cell:
        !git clone -q --depth 1 https://github.com/technoob05/fairgame-more-at-stake.git
        !python fairgame-more-at-stake/exp/exp_Phi4_14B__s0p1.py

For gated HF models prepend in either case:
        import os
        from kaggle_secrets import UserSecretsClient
        try: os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception: pass
"""

# === Hyperparameters (edit if needed; defaults match Strategy B / pd_paper) ===
MODEL          = "Phi4_14B"
SCALE          = 0.1
GAME           = "prisoner_dilemma"
LANGUAGES      = ["en", "fr", "ar", "cn", "vn"]   # paper's 5 languages
ROUNDS         = 30                                 # paper baseline x3, post-endgame
SEEDS          = [0, 1, 2, 3, 4]                    # 5 independent repetitions
MAX_NEW_TOKENS = 64                                 # OptionA / OptionB answers
TEMPERATURE    = 1.0


# --- Bootstrap so this file works whether run as a script (`!python ...`) or
# pasted directly into a Kaggle cell (no `__file__`, no sibling `_common.py`).
import os, sys, subprocess
from pathlib import Path

_REPO_URL = "https://github.com/technoob05/fairgame-more-at-stake.git"
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

if not (_HERE / "_common.py").exists():
    _CLONE = Path.cwd() / "fairgame-more-at-stake"
    if not _CLONE.exists():
        print(f"[bootstrap] cloning {_REPO_URL} -> {_CLONE}")
        subprocess.check_call(
            ["git", "clone", "-q", "--depth", "1", _REPO_URL, str(_CLONE)]
        )
    _HERE = _CLONE / "exp"

sys.path.insert(0, str(_HERE))
from _common import run_experiment


if __name__ == "__main__" or "ipykernel" in sys.modules:
    run_experiment(
        model=MODEL,
        scale=SCALE,
        game=GAME,
        languages=LANGUAGES,
        rounds=ROUNDS,
        seeds=SEEDS,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
