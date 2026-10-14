"""Saved policy-value networks, and scoring decisions with one.

A checkpoint records the encoders it was trained against (the state
encoder's and the decision encoder's versions, planes and global names, and
the micro-step channels); loading refuses one that no longer matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from .. import _core
from ..data.loader import bucket_for, encode_decisions, state_map_ids
from .critic import encoder_signature, pick_device
from .model import LOSS, WIN, PolicyConfig, PolicyValueNet, flat_policy, legal_mask, masked_log_softmax

FORMAT = "awbw-policy"
VERSION = 1


def policy_encoder_signature(movement: bool = False) -> dict[str, Any]:
    """What a network with (or without) the movement planes reads."""
    sig = encoder_signature() | {
        "selection_version": _core.SELECTION_ENCODE_VERSION,
        "selection_planes": [[name, n, cat] for name, n, cat in _core.SELECTION_PLANES],
        "selection_globals": list(_core.SELECTION_GLOBAL_NAMES),
        "channels": list(_core.CHANNELS),
        "nonspatial": _core.NUM_NONSPATIAL_IDS,
    }
    if movement:
        sig |= {
            "movement_version": _core.MOVEMENT_ENCODE_VERSION,
            "movement_planes": [[name, n, cat] for name, n, cat in _core.MOVEMENT_PLANES],
        }
    return sig


def save_policy(path: Path, net: PolicyValueNet, **info: Any) -> None:
    """Write the network's weights and config; ``info`` is kept alongside."""
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "config": asdict(net.cfg),
        "encoder": policy_encoder_signature(net.cfg.movement),
        "state_dict": net.state_dict(),
        "info": info,
    }
    tmp = path.with_name(path.name + ".part")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_policy(path: Path, device: torch.device | str = "cpu") -> tuple[PolicyValueNet, dict[str, Any]]:
    """The saved network, in eval mode on ``device``, and its info."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT or ck.get("version") != VERSION:
        raise ValueError(f"{path}: not an {FORMAT} v{VERSION} checkpoint")
    cfg = PolicyConfig(**ck["config"])
    if ck["encoder"] != policy_encoder_signature(cfg.movement):
        raise ValueError(f"{path}: trained on other input features than the engine's encoders give; retrain")
    net = PolicyValueNet(cfg)
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck["info"]


def policy_ids(index: np.ndarray, width: int, size: int) -> np.ndarray:
    """The micro-step ids of policy indices over a size x size window, for a map
    ``width`` tiles wide (the inverse of ``_core.policy_index``)."""
    index = np.asarray(index, dtype=np.int64)
    spatial = _core.NUM_CHANNELS * size * size
    channel, cell = np.divmod(index, size * size)
    y, x = np.divmod(cell, size)
    ids = channel * _core.MAX_TILES + y * width + x
    return np.where(index < spatial, ids, _core.ID_END_TURN + index - spatial)


@dataclass(slots=True)
class Scored:
    """A decision's legal steps and the network's view of them."""

    ids: np.ndarray  # int64 [L]: legal micro-step ids, ascending
    prior: np.ndarray  # float32 [L]: the policy's probability of each
    wdl: np.ndarray  # float32 [3]: (loss, draw, win) for the player to move

    @property
    def value(self) -> float:
        return float(self.wdl[WIN] - self.wdl[LOSS])


class Policy:
    """A trained policy-value network that scores decisions given as raw states
    [N, STATE_BYTES] plus the picks made so far (as ``encode_policy_batch``).

    Each state's map must be registered with the engine.
    """

    def __init__(self, net: PolicyValueNet, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    @classmethod
    def load(cls, path: Path | str, device: str = "auto") -> Policy:
        dev = pick_device(device)
        net, _ = load_policy(Path(path), dev)
        return cls(net, dev)

    @torch.no_grad()
    def score(
        self,
        states: np.ndarray,
        picks: np.ndarray | None = None,
        n_picks: np.ndarray | None = None,
        ratings: ArrayLike | None = None,
        batch_size: int = 512,
    ) -> list[Scored]:
        """Each decision's legal steps with their prior probabilities, and the value.

        Decision i is in state i after its first ``n_picks[i]`` picks of
        ``picks[i]`` [N, 3] (default: none, the turn's idle phase). ``ratings``
        [N, 2] or [2]: the player to move's, then the opponent's (-1 unknown);
        None: both unknown.
        """
        states = np.ascontiguousarray(states, dtype=np.uint8)
        if states.ndim != 2 or states.shape[1] != _core.STATE_BYTES:
            raise ValueError(f"states must be [N, {_core.STATE_BYTES}] bytes")
        n = len(states)
        picks = np.zeros((n, 3), np.uint16) if picks is None else np.asarray(picks, np.uint16).reshape(n, 3)
        n_picks = np.zeros(n, np.uint8) if n_picks is None else np.asarray(n_picks, np.uint8).reshape(n)
        rating = None if ratings is None else np.broadcast_to(np.asarray(ratings, dtype=np.int32), (n, 2))
        ids = state_map_ids(states)
        width, sizes = np.empty(n, np.int64), np.empty(n, np.int64)
        for m in np.unique(ids):
            md = _core.find_map(int(m))
            if md is None:
                raise ValueError(f"map {m} is not registered")
            width[ids == m], sizes[ids == m] = md.width, bucket_for(md.height, md.width)
        out: list[Scored | None] = [None] * n
        for size in np.unique(sizes):
            rows = np.flatnonzero(sizes == size)
            for b in range(0, len(rows), batch_size):
                sel = rows[b : b + batch_size]
                planes, globals_, nonspatial, _ = encode_decisions(
                    states[sel], picks[sel], n_picks[sel], int(size), targets=False, movement=self.net.cfg.movement
                )
                p = torch.from_numpy(planes).to(self.device)
                ns = torch.from_numpy(nonspatial).to(self.device)
                r = None if rating is None else torch.from_numpy(np.ascontiguousarray(rating[sel])).to(self.device)
                spatial, nonsp, value = self.net(p, torch.from_numpy(globals_).to(self.device), r)
                legal = legal_mask(p, ns)
                prob = masked_log_softmax(flat_policy(spatial, nonsp), legal).exp().cpu().numpy()
                legal_np = legal.cpu().numpy()
                wdl = value.float().softmax(-1).cpu().numpy()
                for j, i in enumerate(sel):
                    idx = np.flatnonzero(legal_np[j])
                    step_ids = policy_ids(idx, int(width[i]), int(size))
                    order = np.argsort(step_ids)
                    out[i] = Scored(step_ids[order], prob[j, idx[order]], wdl[j])
        return [s for s in out if s is not None]
