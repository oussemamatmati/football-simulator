"""A small decoder-only transformer over possession tokens."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from . import config as C


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.2
    max_len: int = C.MAX_LEN

    def to_dict(self) -> dict:
        return asdict(self)


class SmallGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, 4 * cfg.d_model,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        return self.head(self.ln(self.enc(h, mask=mask)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
