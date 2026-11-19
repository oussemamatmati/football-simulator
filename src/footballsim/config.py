"""Shared constants and paths."""
from __future__ import annotations

import os
from pathlib import Path

# Special tokens (ids are fixed: 0..3)
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# Pitch grid: StatsBomb pitch is 120 x 80. We use 6 columns (length) x 4 rows (width).
PITCH_L, PITCH_W = 120.0, 80.0
GRID_X, GRID_Y = 6, 4
N_ZONES = GRID_X * GRID_Y

MAX_LEN = 40  # BOS + up to 38 events + EOS
MIN_TOKEN_COUNT = 5

# Event types kept for the possession sequences
KEEP_TYPES = [
    "Pass", "Carry", "Shot", "Dribble", "Duel", "Clearance", "Interception",
    "Ball Recovery", "Miscontrol", "Dispossessed", "Foul Won", "Block",
]

DEFAULT_COMPETITIONS = [("FIFA World Cup", "2018"), ("FIFA World Cup", "2022")]
EVENT_COLUMNS = [
    "match_id", "index", "possession", "team", "possession_team",
    "type", "location", "pass_outcome", "shot_outcome",
]


def home() -> Path:
    """Directory for downloaded data, checkpoints and reports (override with FOOTBALLSIM_HOME)."""
    p = Path(os.environ.get("FOOTBALLSIM_HOME", "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def events_path() -> Path:
    return home() / "events.parquet"


def checkpoint_path() -> Path:
    return home() / "model.pt"


def report_path() -> Path:
    return home() / "report.json"
