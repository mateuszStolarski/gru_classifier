from pathlib import Path

import torch

BET_PATH = Path.cwd().parent / "data/bet.csv"
DROP_PATH = Path.cwd().parent / "data/drop.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 2024
