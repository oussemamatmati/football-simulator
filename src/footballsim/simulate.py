"""Sample possessions from a trained model, optionally from a given starting state."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import parse_token
from .vocab import Vocab


@torch.no_grad()
def simulate(
    model,
    vocab: Vocab,
    n: int,
    prefix: list[str] | None = None,
    temperature: float = 1.0,
    batch_size: int = 250,
    seed: int | None = None,
    device: str = "cpu",
) -> list[list[str]]:
    """Generate n possessions as lists of tokens (prefix tokens included at the start).

    prefix: tokens that every simulated possession must begin with, e.g. ["Pass_ok_z8"].
    temperature: >1 = more diverse, <1 = more conservative.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    prefix = prefix or []
    unknown = [t for t in prefix if t not in vocab.stoi]
    if unknown:
        raise ValueError(f"Prefix tokens not in vocabulary: {unknown}")
    max_len = model.cfg.max_len
    if len(prefix) + 2 > max_len:
        raise ValueError("prefix too long")

    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    start = torch.tensor([[C.BOS_ID] + [vocab.stoi[t] for t in prefix]], device=device)
    out, remaining = [], n
    while remaining > 0:
        b = min(batch_size, remaining)
        x = start.repeat(b, 1)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        while x.shape[1] < max_len:
            logits = model(x)[:, -1, :] / temperature
            logits[:, [C.PAD_ID, C.BOS_ID, C.UNK_ID]] = -1e9
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
            nxt = torch.where(done, torch.full_like(nxt, C.PAD_ID), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)
            done |= nxt == C.EOS_ID
            if bool(done.all()):
                break
        x = F.pad(x, (0, max_len - x.shape[1]), value=C.PAD_ID)  # keep batches the same width
        out.append(x.cpu().numpy())
        remaining -= b
    arr = np.concatenate(out)
    seqs = [vocab.decode(row) for row in arr]
    return [s for s in seqs if s]


def summarize(seqs: list[list[str]]) -> dict:
    """Headline statistics of a set of possessions."""
    if not seqs:
        return {"possessions": 0}
    parsed = [[parse_token(t) for t in s] for s in seqs]
    has_shot = [any(p and p[0] == "Shot" for p in s) for s in parsed]
    has_goal = [any(p and p[0] == "Shot" and p[1] == "Goal" for p in s) for s in parsed]
    last_zone = [next((p[2] for p in reversed(s) if p), None) for s in parsed]
    final_third = [z is not None and z // C.GRID_Y >= C.GRID_X - 2 for z in last_zone]
    return {
        "possessions": len(seqs),
        "shot_rate": float(np.mean(has_shot)),
        "goal_rate": float(np.mean(has_goal)),
        "avg_length": float(np.mean([len(s) for s in seqs])),
        "ends_in_final_third": float(np.mean(final_third)),
    }


def zone_distribution(seqs: list[list[str]], which: str = "all") -> np.ndarray:
    """Share of events per zone (which='all') or per possession end zone (which='end')."""
    c = np.zeros(C.N_ZONES)
    for s in seqs:
        toks = s if which == "all" else s[-1:]
        for t in toks:
            p = parse_token(t)
            if p:
                c[p[2]] += 1
    return c / c.sum() if c.sum() else c
