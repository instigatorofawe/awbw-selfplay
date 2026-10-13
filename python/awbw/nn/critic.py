"""Saved critics, and valuing raw states with one.

A checkpoint records the encoder it was trained against (version, planes and
global names), and loading refuses one that no longer matches the engine's,
since the network would silently read the wrong features.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import _core
from ..data.loader import bucket_for, encode, state_map_ids
from .model import LOSS, WIN, CriticConfig, CriticNet

FORMAT = "awbw-critic"
VERSION = 1


def pick_device(name: str = "auto") -> torch.device:
    """``name``, or for "auto" the best available: CUDA, then Apple's MPS, then the CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encoder_signature() -> dict[str, Any]:
    return {
        "version": _core.ENCODE_VERSION,
        "planes": [[name, n, cat] for name, n, cat in _core.PLANES],
        "globals": list(_core.GLOBAL_NAMES),
    }


def save_critic(path: Path, net: CriticNet, **info: Any) -> None:
    """Write the network's weights and config; ``info`` is kept alongside (metrics, training setup)."""
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "config": asdict(net.cfg),
        "encoder": encoder_signature(),
        "state_dict": net.state_dict(),
        "info": info,
    }
    tmp = path.with_name(path.name + ".part")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_critic(path: Path, device: torch.device | str = "cpu") -> tuple[CriticNet, dict[str, Any]]:
    """The saved network, in eval mode on ``device``, and its info."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT or ck.get("version") != VERSION:
        raise ValueError(f"{path}: not an {FORMAT} v{VERSION} checkpoint")
    if ck["encoder"] != encoder_signature():
        raise ValueError(
            f"{path}: trained on encoder v{ck['encoder']['version']} features, but the engine's "
            f"encoder (v{_core.ENCODE_VERSION}) differs; retrain"
        )
    net = CriticNet(CriticConfig(**ck["config"]))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck["info"]


class Critic:
    """A trained critic that values raw states [N, STATE_BYTES] for their player to move.

    Each state's map must be registered with the engine (opening a ``Dataset``
    or ``engine.map_data`` does that).
    """

    def __init__(self, net: CriticNet, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    @classmethod
    def load(cls, path: Path | str, device: str = "auto") -> Critic:
        dev = pick_device(device)
        net, _ = load_critic(Path(path), dev)
        return cls(net, dev)

    @torch.no_grad()
    def wdl(self, states: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Probabilities [N, 3] of (loss, draw, win) for each state's player to move."""
        states = np.ascontiguousarray(states, dtype=np.uint8)
        if states.ndim != 2 or states.shape[1] != _core.STATE_BYTES:
            raise ValueError(f"states must be [N, {_core.STATE_BYTES}] bytes")
        ids = state_map_ids(states)
        bucket = {}
        for m in np.unique(ids):
            md = _core.find_map(int(m))
            if md is None:
                raise ValueError(f"map {m} is not registered")
            bucket[int(m)] = bucket_for(md.height, md.width)
        sizes = np.array([bucket[int(m)] for m in ids], dtype=np.int64)
        out = np.empty((len(states), 3), dtype=np.float32)
        for size in np.unique(sizes):
            rows = np.flatnonzero(sizes == size)
            for b in range(0, len(rows), batch_size):
                sel = rows[b : b + batch_size]
                planes, globals_ = encode(states[sel], int(size))
                logits = self.net(torch.from_numpy(planes).to(self.device), torch.from_numpy(globals_).to(self.device))
                out[sel] = logits.float().softmax(-1).cpu().numpy()
        return out

    def value(self, states: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Expected result in [-1, 1] for each state's player to move."""
        p = self.wdl(states, batch_size)
        return p[:, WIN] - p[:, LOSS]
