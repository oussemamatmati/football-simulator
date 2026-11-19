"""Held-out evaluation: perplexity vs a Markov baseline, simulation realism, calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config as C
from .simulate import simulate, summarize
from .train import run_epoch


def _pairs(X: np.ndarray):
    a, b = X[:, :-1], X[:, 1:]
    m = b != C.PAD_ID
    return a[m], b[m]


def markov_perplexity(Xtr: np.ndarray, Xte: np.ndarray, V: int, alpha: float = 0.1) -> float:
    """First-order Markov chain with add-alpha smoothing, evaluated on Xte."""
    a, b = _pairs(Xtr)
    counts = np.zeros((V, V))
    np.add.at(counts, (a, b), 1)
    P = (counts + alpha) / (counts + alpha).sum(1, keepdims=True)
    a, b = _pairs(Xte)
    return float(np.exp(-np.log(P[a, b]).mean()))


@torch.no_grad()
def shot_calibration(model, Xte: np.ndarray, vocab, device: str = "cpu") -> pd.DataFrame:
    """Reliability data: predicted P(next event is a shot) vs observed frequency."""
    shot_ids = [i for t, i in vocab.stoi.items() if t.startswith("Shot")]
    model.eval()
    ps = []
    for i in range(0, len(Xte), 256):
        xb = torch.as_tensor(Xte[i : i + 256], dtype=torch.long, device=device)
        ps.append(F.softmax(model(xb[:, :-1]), -1)[:, :, shot_ids].sum(-1).cpu().numpy())
    p = np.concatenate(ps)
    y = np.isin(Xte[:, 1:], shot_ids).astype(float)
    m = Xte[:, 1:] != C.PAD_ID
    p, y = p[m], y[m]
    edges = [0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 1.0]
    bins = pd.cut(p, edges, include_lowest=True)
    g = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
    out = g.mean().reset_index(drop=True)
    out["count"] = g.size().to_numpy()
    return out[out["count"] >= 20].reset_index(drop=True)


def evaluate(model, vocab, ckpt: dict, seqs: pd.DataFrame, n_sim: int = 10000, device: str = "cpu") -> dict:
    """Full evaluation on the held-out test matches stored in the checkpoint."""
    split = ckpt["split"]
    tr = seqs[seqs.match_id.isin(split["train"])]
    te = seqs[seqs.match_id.isin(split["test"])]
    Xtr, Xte = vocab.encode_many(tr.toks), vocab.encode_many(te.toks)

    markov = markov_perplexity(Xtr, Xte, len(vocab))
    tf = float(np.exp(run_epoch(model, Xte, None, device=device)))
    sim = simulate(model, vocab, n_sim, seed=0, device=device)
    real = te.toks.tolist()
    calib = shot_calibration(model, Xte, vocab, device=device)
    ece = float(np.average((calib.p - calib.y).abs(), weights=calib["count"]))

    return {
        "markov_perplexity": markov,
        "transformer_perplexity": tf,
        "improvement_pct": (1 - tf / markov) * 100,
        "n_test_possessions": len(te),
        "n_test_matches": len(split["test"]),
        "real": summarize(real),
        "simulated": summarize(sim),
        "shot_calibration_gap": ece,
        "calibration": calib.to_dict(orient="list"),
        "history": ckpt.get("history", []),
    }


def save_report(report: dict, path: Path | None = None) -> Path:
    path = Path(path or C.report_path())
    path.write_text(json.dumps(report, indent=2))
    return path


def load_report(path: Path | None = None) -> dict:
    return json.loads(Path(path or C.report_path()).read_text())
