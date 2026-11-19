import numpy as np
import pandas as pd
import pytest

from footballsim import config as C


@pytest.fixture(scope="session")
def toy_events() -> pd.DataFrame:
    """Synthetic StatsBomb-like events: 20 matches x 30 possessions of random play."""
    rng = np.random.RandomState(0)
    rows = []
    for m in range(20):
        for poss in range(30):
            team = "A" if poss % 2 == 0 else "B"
            for k in range(rng.randint(2, 12)):
                typ = rng.choice(["Pass", "Pass", "Pass", "Carry", "Shot", "Duel"])
                rows.append(dict(
                    match_id=m, index=poss * 100 + k, possession=poss, team=team, possession_team=team,
                    type=typ, loc_x=rng.uniform(0, 120), loc_y=rng.uniform(0, 80),
                    pass_outcome=("Incomplete" if typ == "Pass" and rng.rand() < 0.2 else None),
                    shot_outcome=(rng.choice(["Goal", "Saved", "Off T"]) if typ == "Shot" else None),
                ))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def toy_seqs(toy_events):
    from footballsim.data import build_possessions

    return build_possessions(toy_events)


@pytest.fixture()
def tiny_ckpt(toy_seqs):
    from footballsim.train import train_model

    return train_model(toy_seqs, epochs=2, cfg_overrides=dict(d_model=32, n_heads=2, n_layers=1),
                       device="cpu", log=lambda *_: None)
