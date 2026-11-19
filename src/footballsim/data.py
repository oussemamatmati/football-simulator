"""Download StatsBomb open data and turn events into tokenized possessions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def zone_of(x: float, y: float) -> int:
    """Map a pitch location (StatsBomb 120x80 coordinates) to a zone id in [0, N_ZONES)."""
    xi = min(max(int(x / C.PITCH_L * C.GRID_X), 0), C.GRID_X - 1)
    yi = min(max(int(y / C.PITCH_W * C.GRID_Y), 0), C.GRID_Y - 1)
    return xi * C.GRID_Y + yi


def make_token(action: str, outcome: str, zone: int) -> str:
    return f"{action.replace(' ', '')}_{outcome}_z{zone}"


def parse_token(tok: str) -> tuple[str, str, int] | None:
    """Inverse of make_token. Returns None for special tokens."""
    if tok in C.SPECIAL_TOKENS or "_z" not in tok:
        return None
    head, zone = tok.rsplit("_z", 1)
    action, outcome = head.split("_", 1)
    return action, outcome, int(zone)


def zone_center(zone: int) -> tuple[float, float]:
    xi, yi = divmod(zone, C.GRID_Y)
    return (xi + 0.5) * C.PITCH_L / C.GRID_X, (yi + 0.5) * C.PITCH_W / C.GRID_Y


def download_events(competitions=None, cache: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Download events for the given (competition_name, season_name) pairs; cache as parquet."""
    cache = cache or C.events_path()
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    from statsbombpy import sb  # imported lazily so the rest of the package works offline
    from tqdm.auto import tqdm

    competitions = competitions or C.DEFAULT_COMPETITIONS
    comps = sb.competitions()
    frames = []
    for name, season in competitions:
        sel = comps[(comps.competition_name == name) & (comps.season_name == season)]
        if sel.empty:
            raise ValueError(f"No StatsBomb competition found for {name!r} {season!r}")
        for _, r in sel.iterrows():
            matches = sb.matches(competition_id=r.competition_id, season_id=r.season_id)
            for mid in tqdm(matches.match_id, desc=f"{name} {season}"):
                ev = sb.events(match_id=mid)
                ev["match_id"] = mid
                frames.append(ev.reindex(columns=C.EVENT_COLUMNS))
    events = pd.concat(frames, ignore_index=True)
    # location is a list per row; store as two float columns for parquet friendliness
    loc = events["location"].apply(lambda v: v if isinstance(v, (list, tuple)) and len(v) >= 2 else [np.nan, np.nan])
    events["loc_x"] = loc.apply(lambda v: v[0]).astype(float)
    events["loc_y"] = loc.apply(lambda v: v[1]).astype(float)
    events = events.drop(columns=["location"])
    for col in ("pass_outcome", "shot_outcome"):
        events[col] = events[col].astype("string")
    cache.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(cache)
    return events


def build_possessions(events: pd.DataFrame, max_len: int = C.MAX_LEN) -> pd.DataFrame:
    """Return one row per possession with columns: match_id, possession, team, toks (list[str])."""
    ev = events.copy()
    if "loc_x" not in ev.columns:  # raw StatsBomb frame with a location list column
        ev["loc_x"] = ev["location"].apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        ev["loc_y"] = ev["location"].apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)

    ev = ev[
        ev["type"].isin(C.KEEP_TYPES)
        & (ev["team"] == ev["possession_team"])
        & ev["loc_x"].notna()
        & ev["loc_y"].notna()
    ].copy()
    ev = ev.sort_values(["match_id", "possession", "index"])
    if ev.empty:
        return pd.DataFrame({"match_id": [], "possession": [], "team": [], "toks": []})

    zones = [zone_of(x, y) for x, y in zip(ev["loc_x"], ev["loc_y"])]
    outcomes = []
    for t, po, so in zip(ev["type"], ev["pass_outcome"], ev["shot_outcome"]):
        if t == "Pass":
            outcomes.append("fail" if pd.notna(po) else "ok")
        elif t == "Shot":
            outcomes.append(str(so).replace(" ", "") if pd.notna(so) else "na")
        else:
            outcomes.append("na")
    ev["tok"] = [make_token(a, o, z) for a, o, z in zip(ev["type"], outcomes, zones)]

    seqs = (
        ev.groupby(["match_id", "possession"])
        .agg(team=("team", "first"), toks=("tok", list))
        .reset_index()
    )
    seqs["n"] = seqs["toks"].apply(len)
    seqs = seqs[(seqs.n >= 2) & (seqs.n <= max_len - 2)].reset_index(drop=True)
    return seqs.drop(columns="n")
