"""Explicit pre-LN causal transformer. Intervene on z before the shared projection."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff: int = 128
    vocab: int = 33
    length: int = 10
    answers: int = 16

    def validate(self):
        if (any(type(v) is not int for v in asdict(self).values())
                or not 1 <= self.layers <= 2 or not 1 <= self.heads <= 4
                or not 4 <= self.width <= 128 or self.width % self.heads
                or not 4 <= self.ff <= 256 or self.vocab != 33
                or self.length != 10 or self.answers != 16):
            raise ValueError("invalid bounded model configuration")


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.d = cfg.heads, cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)

    def forward(self, x, head=None, noop=False):
        b, t, w = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.d).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        pattern = scores.masked_fill(mask, -torch.inf).softmax(-1)
        z = pattern @ v  # B,H,T,D; zero at ALL positions of just one head
        if noop:
            z = z.clone()
        if head is not None:
            if type(head) is not int or not 0 <= head < self.heads:
                raise ValueError("head out of range")
            z = z.clone()
            z[:, head, :, :] = 0
        result = self.out(z.transpose(1, 2).reshape(b, t, w))
        return result, pattern


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, cfg.ff), nn.GELU(),
                                 nn.Linear(cfg.ff, cfg.width))

    def forward(self, x, head=None, noop=False):
        update, pattern = self.attn(self.ln1(x), head, noop)
        x = x + update
        return x + self.mlp(self.ln2(x)), pattern


class Transformer(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab, cfg.width)
        self.position = nn.Embedding(cfg.length, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.readout = nn.Linear(cfg.width, cfg.answers)
        if sum(p.numel() for p in self.parameters()) > 250000:
            raise ValueError("parameter budget exceeded")

    def forward(self, tokens, intervention=None, noop=False):
        if intervention is not None:
            layer, head = intervention
            if type(layer) is not int or not 0 <= layer < self.cfg.layers:
                raise ValueError("layer out of range")
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1]))
        patterns = []
        for layer, block in enumerate(self.blocks):
            head = intervention[1] if intervention is not None and layer == intervention[0] else None
            x, pattern = block(x, head, noop)
            patterns.append(pattern)
        return self.readout(self.norm(x[:, -1])), torch.stack(patterns, dim=1)


def save_weights(model, path):
    # Non-pickle, uncompressed numeric arrays. Metadata is separate bounded JSON.
    np.savez(path, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def load_weights(model, path):
    import zipfile
    expected = model.state_dict()
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if (len(items) != len(expected) or any(i.file_size > 2**20 for i in items)
                or sum(i.file_size for i in items) > 2**21
                or len({i.filename for i in items}) != len(items)):
            raise ValueError("checkpoint archive limits exceeded")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(expected):
            raise ValueError("checkpoint keys differ")
        state = {}
        for key, template in expected.items():
            a = data[key]
            if a.shape != tuple(template.shape) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError("invalid checkpoint tensor")
            state[key] = torch.from_numpy(a.copy())
    model.load_state_dict(state, strict=True)
