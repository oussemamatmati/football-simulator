"""Token vocabulary."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import config as C


class Vocab:
    def __init__(self, tokens: list[str]):
        assert tokens[: len(C.SPECIAL_TOKENS)] == C.SPECIAL_TOKENS, "special tokens must come first"
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, sequences, min_count: int = C.MIN_TOKEN_COUNT) -> "Vocab":
        cnt = Counter(t for s in sequences for t in s)
        return cls(C.SPECIAL_TOKENS + sorted(t for t, c in cnt.items() if c >= min_count))

    def encode(self, toks: list[str], max_len: int = C.MAX_LEN) -> np.ndarray:
        ids = [C.BOS_ID] + [self.stoi.get(t, C.UNK_ID) for t in toks] + [C.EOS_ID]
        out = np.full(max_len, C.PAD_ID, dtype=np.int64)
        out[: len(ids)] = ids[:max_len]
        return out

    def encode_many(self, rows, max_len: int = C.MAX_LEN) -> np.ndarray:
        rows = list(rows)
        if not rows:
            return np.zeros((0, max_len), dtype=np.int64)
        return np.stack([self.encode(r, max_len) for r in rows])

    def decode(self, ids) -> list[str]:
        """Decode ids to tokens, dropping BOS, stopping at EOS/PAD."""
        out = []
        for i in ids:
            i = int(i)
            if i in (C.EOS_ID, C.PAD_ID):
                break
            if i == C.BOS_ID:
                continue
            out.append(self.itos[i])
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))
