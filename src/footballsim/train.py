"""Match-level splitting, training and checkpointing."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config as C
from .model import ModelConfig, SmallGPT
from .vocab import Vocab


def split_matches(match_ids, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15):
    """Split match ids into (train, val, test). Splitting by match prevents leakage."""
    ids = np.sort(np.unique(np.asarray(match_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test, n_val = max(1, int(test_frac * n)), max(1, int(val_frac * n))
    test, val, train = ids[:n_test], ids[n_test : n_test + n_val], ids[n_test + n_val :]
    return train.tolist(), val.tolist(), test.tolist()


def device_auto() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, X, optimizer=None, batch_size: int = 128, device: str = "cpu") -> float:
    """One pass over X. Returns mean cross-entropy per token (trains if optimizer is given)."""
    train = optimizer is not None
    model.train(train)
    idx = np.random.permutation(len(X)) if train else np.arange(len(X))
    total, count = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[idx[i : i + batch_size]], dtype=torch.long, device=device)
        with torch.set_grad_enabled(train):
            logits = model(xb[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1),
                ignore_index=C.PAD_ID, reduction="sum",
            )
        n_tok = int((xb[:, 1:] != C.PAD_ID).sum())
        if train:
            optimizer.zero_grad()
            (loss / max(n_tok, 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach())
        count += n_tok
    return total / max(count, 1)


def train_model(
    seqs: pd.DataFrame,
    epochs: int = 40,
    patience: int = 4,
    lr: float = 5e-4,
    seed: int = 42,
    cfg_overrides: dict | None = None,
    device: str | None = None,
    log=print,
) -> dict:
    """Train and return a checkpoint dict (model weights, vocab, config, split, metrics)."""
    device = device or device_auto()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_m, val_m, test_m = split_matches(seqs.match_id, seed)
    tr = seqs[seqs.match_id.isin(train_m)]
    va = seqs[seqs.match_id.isin(val_m)]

    vocab = Vocab.build(tr.toks)  # vocabulary from training matches only
    Xtr, Xva = vocab.encode_many(tr.toks), vocab.encode_many(va.toks)

    cfg = ModelConfig(vocab_size=len(vocab), **(cfg_overrides or {}))
    model = SmallGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log(f"device={device} | vocab={len(vocab)} | params={model.n_params/1e6:.2f}M | "
        f"train/val possessions={len(Xtr)}/{len(Xva)}")

    best, best_state, wait, history = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(model, Xtr, opt, device=device)
        va_loss = run_epoch(model, Xva, None, device=device)
        history.append({"epoch": ep, "train_ppl": float(np.exp(tr_loss)), "val_ppl": float(np.exp(va_loss))})
        log(f"epoch {ep:02d} | train ppl {np.exp(tr_loss):.2f} | val ppl {np.exp(va_loss):.2f}")
        if va_loss < best:
            best, wait = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                log("early stopping")
                break

    model.load_state_dict(best_state)
    return {
        "model_state": best_state,
        "vocab": vocab.itos,
        "model_cfg": cfg.to_dict(),
        "split": {"train": train_m, "val": val_m, "test": test_m},
        "history": history,
        "seed": seed,
        "train_seconds": time.time() - t0,
    }


def save_checkpoint(ckpt: dict, path: Path | None = None) -> Path:
    path = Path(path or C.checkpoint_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def load_checkpoint(path: Path | None = None, device: str = "cpu"):
    """Load a checkpoint. Returns (model, vocab, ckpt)."""
    path = Path(path or C.checkpoint_path())
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Run `footballsim train` first.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(ckpt["vocab"])
    model = SmallGPT(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt
