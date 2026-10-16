"""Gumbel MCTS with a policy-value network (PLAN.md §8).

``_core.BatchedSearch`` runs the searches; this module gives it a network.
``run`` makes the collect / batch / submit rounds, ``NetEvaluator`` scores a
round's leaves, ``SearchAgent`` searches each decision it is asked for (an
``agents.Agent``, so ``agents.play`` and the arena can use it), and
``self_play`` plays whole games, every decision searched, yielding the
samples and the finished games as they come.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from . import _core
from .agents import Decision, Note
from .data.loader import bucket_for
from .nn.model import PolicyValueNet, flat_policy, value_of
from .nn.policy import Policy

# (planes, globals, legal policy indices, movers) -> (logits over each leaf's legal steps, values)
Evaluate = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class NetEvaluator:
    """Scores search leaves with a policy-value network: the logits of each
    leaf's legal steps and P(win) - P(loss) for the player deciding there.

    ``ratings``: the (mover's, opponent's) ratings the network is told at every
    leaf, e.g. one strong player's for both; None leaves them unknown. On CUDA
    the forward pass runs in bfloat16 unless ``bf16`` is False. ``compile``
    runs it through torch.compile (3x the evaluations on the RTX 5070), with
    batches padded to a power of two so that few shapes get compiled.
    """

    def __init__(
        self,
        net: PolicyValueNet | Policy,
        device: torch.device | str | None = None,
        ratings: tuple[int, int] | None = None,
        bf16: bool = True,
        batch_size: int = 1024,
        compile: bool = False,
    ):
        if isinstance(net, Policy):
            device = net.device if device is None else device
            net = net.net
        self.device = torch.device(device or "cpu")
        self.net = net.to(self.device).eval()
        self.cfg = self.net.cfg
        self.forward = torch.compile(self.net) if compile else self.net
        self.pad = compile
        self.ratings = ratings
        self.bf16 = bf16 and self.device.type == "cuda"
        self.batch_size = batch_size
        self.evals = 0

    @property
    def movement(self) -> bool:
        return self.cfg.movement

    @torch.no_grad()
    def __call__(
        self, planes: np.ndarray, globals_: np.ndarray, legal: np.ndarray, movers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(planes)
        logits = np.empty(legal.shape, np.float32)
        values = np.empty(n, np.float32)
        for b in range(0, n, self.batch_size):
            rows = slice(b, min(n, b + self.batch_size))
            m = rows.stop - rows.start
            size = max(16, 1 << (m - 1).bit_length()) if self.pad else m
            p = torch.zeros((size, *planes.shape[1:]), dtype=torch.uint8)
            g = torch.zeros((size, globals_.shape[1]), dtype=torch.float32)
            p[:m], g[:m] = torch.from_numpy(planes[rows]), torch.from_numpy(globals_[rows])
            p, g = p.to(self.device), g.to(self.device)
            idx = torch.from_numpy(legal[rows]).to(self.device)
            r = None
            if self.ratings is not None:
                r = torch.tensor(self.ratings, dtype=torch.int32, device=self.device).expand(size, 2)
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.bf16):
                spatial, nonspatial, value = self.forward(p, g, r)
            flat = flat_policy(spatial[:m], nonspatial[:m]).float()
            logits[rows] = flat.gather(1, idx.clamp(min=0)).cpu().numpy()
            values[rows] = value_of(value[:m]).cpu().numpy()
        self.evals += n
        return logits, values


def run(search: _core.BatchedSearch, evaluate: Evaluate) -> int:
    """Collect / evaluate / submit until nothing is left to search; returns the
    leaves evaluated."""
    evals = 0
    while search.collect():
        planes, globals_, legal, movers = search.batch()
        logits, values = evaluate(planes, globals_, legal, movers)
        search.submit(logits, values)
        evals += len(planes)
    return evals


def selection_of(state: _core.GameState, picks: Iterable[int]) -> _core.Selection:
    """The selection the picks reach from idle (picks change no state)."""
    sel = _core.Selection()
    s = state.copy()
    for pick in picks:
        if _core.step(s, sel, int(pick)) != _core.OK:
            raise ValueError(f"pick {pick} isn't legal")
    return sel


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """What ``SearchAgent`` and ``self_play`` pass to ``_core.SearchConfig``."""

    sims: int = 64
    considered: int = 16
    gumbel_scale: float = 1.0
    max_days: int = 0  # the tree ends games past this day as a day limit decides them; 0: never
    rescale: bool = False  # SearchConfig.rescale
    value_scale: float = 0.1  # SearchConfig.value_scale
    pick_values: bool = True  # SearchConfig.pick_values
    skip_prior: float = 0.0  # SearchConfig.skip_prior: a root this sure of a step isn't searched

    def config(self, movement: bool, seed: int, **more) -> _core.SearchConfig:
        return _core.SearchConfig(
            sims=self.sims,
            skip_prior=self.skip_prior,
            considered=self.considered,
            gumbel_scale=self.gumbel_scale,
            rescale=self.rescale,
            value_scale=self.value_scale,
            pick_values=self.pick_values,
            max_days=self.max_days,
            movement=movement,
            seed=seed,
            **more,
        )


class SearchAgent:
    """Searches each decision it is asked for with a network and plays the step
    the search chooses (an ``agents.Agent``). ``last`` keeps the latest call's
    ``_core.SearchResult`` per decision, and ``notes`` their values."""

    def __init__(self, evaluator: NetEvaluator, settings: SearchSettings | None = None, seed: int = 0):
        self.evaluator = evaluator
        self.settings = settings or SearchSettings()
        self.seed = seed
        self.calls = 0
        self.decisions = 0  # asked for, over every call
        self.last: list[_core.SearchResult] = []
        self.notes: list[Note] = []

    def choose(self, decisions: list[Decision]) -> list[int]:
        self.calls += 1
        self.decisions += len(decisions)
        results: list[_core.SearchResult | None] = [None] * len(decisions)
        by_size: dict[int, list[int]] = {}
        for i, d in enumerate(decisions):
            md = _core.find_map(d.state.map_id)
            if md is None:
                raise ValueError(f"map {d.state.map_id} is not registered")
            by_size.setdefault(bucket_for(md.height, md.width), []).append(i)
        for size, rows in sorted(by_size.items()):
            cfg = self.settings.config(self.evaluator.movement, self.seed * 1_000_003 + self.calls)
            search = _core.BatchedSearch(cfg, size)
            slots = [
                search.add_root(decisions[i].state, selection_of(decisions[i].state, decisions[i].picks)) for i in rows
            ]
            run(search, self.evaluator)
            for i, slot in zip(rows, slots, strict=True):
                results[i] = search.result(slot)
        self.last = [r for r in results if r is not None]
        self.notes = [Note(r.value, r.net_value, r.sims) for r in self.last]
        return [r.id for r in self.last]


@dataclass(slots=True)
class SelfPlayBatch:
    """What one round of ``self_play`` produced."""

    samples: dict[str, np.ndarray]  # as BatchedSearch.take_samples
    records: list[_core.GameRecord]
    evals: int


def self_play(
    evaluator: Evaluate,
    starts: Iterator[_core.GameState],
    config: _core.SearchConfig,
    size: int,
    parallel: int = 256,
) -> Iterator[SelfPlayBatch]:
    """Play a game from each start, ``parallel`` at a time, every decision with
    a choice searched; yields each round's samples and finished games. The
    starts' maps must fit in ``size``."""
    search = _core.BatchedSearch(config, size)
    more = True
    while True:
        while more and search.active < parallel:
            start = next(starts, None)
            if start is None:
                more = False
                break
            search.add_game(start)
        n = search.collect()
        evals = 0
        if n:
            planes, globals_, legal, movers = search.batch()
            logits, values = evaluator(planes, globals_, legal, movers)
            search.submit(logits, values)
            evals = n
        yield SelfPlayBatch(search.take_samples(), search.take_records(), evals)
        if not n and not more:
            return
