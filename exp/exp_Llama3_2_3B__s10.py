"""
FAIRGAME slice: Llama3_2_3B @ payoff_scale=10.0.

One Kaggle session, one (model, scale) slice. Resume-safe: rerunning the
notebook skips any (lang, seed) cell whose CSV already exists.

Kaggle cell:
    import os
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception: pass

    !rm -rf fairgame-more-at-stake _common.py exp_Llama3_2_3B__s10.py
    !git clone -q --depth 1 https://github.com/technoob05/fairgame-more-at-stake.git
    !python fairgame-more-at-stake/exp/exp_Llama3_2_3B__s10.py
"""

# === Hyperparameters (edit if needed; defaults match Strategy B / pd_paper) ===
MODEL          = "Llama3_2_3B"
SCALE          = 10.0
GAME           = "prisoner_dilemma"
LANGUAGES      = ["en", "fr", "ar", "cn", "vn"]   # paper's 5 languages
ROUNDS         = 30                                 # paper baseline x3, post-endgame
SEEDS          = [0, 1, 2, 3, 4]                    # 5 independent repetitions
MAX_NEW_TOKENS = 64                                 # OptionA / OptionB answers
TEMPERATURE    = 1.0


import sys
from pathlib import Path
# Make sibling _common.py importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run_experiment


if __name__ == "__main__":
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
