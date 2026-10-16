"""Self-play training (PLAN.md §8, M5): one process alternates rounds of search
with steps of training.

Games start from human games' first states (the v1 dataset's training split,
so the arena's validation starts stay unseen) and are played with
``BatchedSearch.add_game``, one search per board bucket, the network scoring
every round's leaves on the GPU. When a game ends its full searches become
samples, each labelled with the result for the player deciding there, and go
into a sliding window of the latest ``window`` samples. Before they do, the
network's losses on them are logged as "fresh": a game it hasn't trained on.

Training keeps a fixed ratio: ``train_ratio`` samples trained on per sample
added to the window, from the first ``min_window`` on. A self-play sample's
policy loss is the cross-entropy of the improved policy (the search's
target), its value loss that of the result, on a random ``value_fraction``
of the samples (a game's samples share their result). A share
``human_fraction`` of the batches are human decisions as in pre-training
(``train_policy``): the step taken and the result. There is no gating: the
search always uses the network being trained, which is the same module.

Self-play tells the network no ratings, at the leaves and in training; human
batches keep theirs, hidden at random. The run directory holds
``metrics.jsonl``, ``net-NNNNNN.pt`` every ``checkpoint_every`` steps and
``last.pt`` + ``optimizer.pt`` for ``resume`` (the window starts empty again).

Cheaper search: a fast search whose root the network is sure of (``skip_prior``)
plays without simulating, and with a fast network (``train``'s ``fast``) the
fast searches' leaves are scored by it instead. The full searches, which become
samples, always use the main network. The fast network trains on the same
batches as the main one, so it follows the main one's search targets
(``fast_train``); it is kept as ``fast-last.pt`` + ``fast-optimizer.pt``.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .. import _core
from ..data.loader import BUCKETS, DecisionBatch, bucket_for, decision_batches, encode_decisions, in_validation
from ..data.shards import Dataset
from ..search import NetEvaluator, SearchSettings
from .model import PolicyValueNet
from .policy import load_policy, save_policy
from .train_critic import _optimizer, prefetch
from .train_policy import _forward, _to_device

MAX_ENTRIES = 192  # policy entries kept per sample (the most legal steps in v1 is 160)


@dataclass
class SelfPlayConfig:
    # Search
    sims: int = 64  # simulations of a full search, which becomes a sample
    fast_sims: int = 16  # the rest only choose the step
    full_fraction: float = 0.25
    skip_prior: float = 0.0  # a fast search whose root gives one step this much plays it unsearched; 0: never
    considered: int = 16
    gumbel_scale: float = 1.0
    value_scale: float = 0.1
    max_days: int = 30  # then a day limit decides the game
    # Resignation (SearchConfig): a player resigns when its searched value was at most
    # resign_value at the start of each of its last resign_turns turns; a share
    # resign_off of the games play on, to count how often the rule was wrong.
    resign_value: float = -0.9
    resign_turns: int = 2  # 0: never
    resign_off: float = 0.1
    parallel: int = 512  # games at once
    fast_train: bool = True  # train the fast network (train's `fast`) on the main one's batches
    # Window and training
    window: int = 300_000  # self-play samples kept
    min_window: int = 20_000  # samples before training starts
    train_ratio: float = 4.0  # self-play samples trained on per sample added
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    value_weight: float = 1.0
    value_fraction: float = 0.25  # share of self-play samples whose result trains the value head
    human_fraction: float = 0.25  # share of training batches from human games
    human_value_fraction: float = 0.05
    rating_dropout: float = 0.25  # human batches
    augment: bool = True
    compile: bool = False
    bf16: bool = True
    checkpoint_every: int = 2000  # training steps
    log_every: float = 60.0  # seconds
    max_steps: int = 0  # training steps to stop at; 0 = never
    seed: int = 0


# --- The window ----------------------------------------------------------------


class Window:
    """The latest ``capacity`` self-play samples, the oldest dropped first.

    A sample is a decision (its state and the picks so far), the board bucket
    its policy indices are for, the result for the player deciding and the
    improved policy as up to MAX_ENTRIES (policy index, probability) pairs,
    the likeliest kept and renormalized.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.state = np.empty((capacity, _core.STATE_BYTES), np.uint8)
        self.picks = np.empty((capacity, 3), np.uint16)
        self.n_picks = np.empty(capacity, np.uint8)
        self.size = np.zeros(capacity, np.uint8)
        self.z = np.empty(capacity, np.int8)
        self.index = np.empty((capacity, MAX_ENTRIES), np.uint16)
        self.prob = np.empty((capacity, MAX_ENTRIES), np.float16)
        self.count = 0  # samples held
        self.added = 0  # samples ever added
        self._rows: dict[int, np.ndarray] | None = None

    def add(self, s: dict[str, np.ndarray]) -> None:
        """Samples as ``game_samples`` gives them."""
        n = len(s["z"])
        if n > self.capacity:
            s = {k: v[-self.capacity :] for k, v in s.items()}
            n = self.capacity
        at = (self.added + np.arange(n)) % self.capacity
        self.state[at] = s["states"]
        self.picks[at] = s["picks"]
        self.n_picks[at] = s["n_picks"]
        self.size[at] = s["size"]
        self.z[at] = s["z"]
        self.index[at] = s["index"]
        self.prob[at] = s["prob"]
        self.added += n
        self.count = min(self.capacity, self.count + n)
        self._rows = None

    def rows(self) -> dict[int, np.ndarray]:
        """The rows held, by bucket."""
        if self._rows is None:
            held = self.size[: self.count]
            self._rows = {b: r for b in BUCKETS if len(r := np.flatnonzero(held == b))}
        return self._rows

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        """A bucket, chosen by its share of the window, and ``batch_size`` of its rows."""
        rows = self.rows()
        sizes = list(rows)
        counts = np.array([len(rows[b]) for b in sizes], np.float64)
        size = sizes[int(rng.choice(len(sizes), p=counts / counts.sum()))]
        return size, rows[size][rng.integers(len(rows[size]), size=batch_size)]


def game_samples(samples: list[dict[str, np.ndarray]], winner: int, size: int) -> dict[str, np.ndarray]:
    """A finished game's samples (``take_samples`` rows) as ``Window.add`` takes
    them, each labelled with the result for its mover: 1 won, -1 lost, 0 drawn."""
    states = np.concatenate([s["states"] for s in samples])
    mover = np.concatenate([s["mover"] for s in samples])
    n = len(states)
    index = np.zeros((n, MAX_ENTRIES), np.uint16)
    prob = np.zeros((n, MAX_ENTRIES), np.float32)
    row = 0
    for s in samples:
        off = s["offsets"]
        for i in range(len(off) - 1):
            idx, p = s["index"][off[i] : off[i + 1]], s["policy"][off[i] : off[i + 1]]
            if len(p) > MAX_ENTRIES:
                keep = np.argsort(-p, kind="stable")[:MAX_ENTRIES]
                idx, p = idx[keep], p[keep]
            index[row, : len(p)] = idx
            prob[row, : len(p)] = p / p.sum()
            row += 1
    z = np.zeros(n, np.int8) if winner == _core.DRAW else np.where(mover == winner, 1, -1).astype(np.int8)
    return {
        "states": states,
        "picks": np.concatenate([s["picks"] for s in samples]),
        "n_picks": np.concatenate([s["n_picks"] for s in samples]),
        "size": np.full(n, size, np.uint8),
        "z": z,
        "index": index,
        "prob": prob.astype(np.float16),
    }


def _split_by_game(s: dict[str, np.ndarray]) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """``take_samples`` output, game by game, with its own offsets."""
    games = s["game"]
    for g in np.unique(games):
        rows = np.flatnonzero(games == g)
        off = s["offsets"]
        steps = np.concatenate([np.arange(off[r], off[r + 1]) for r in rows])
        lens = off[rows + 1] - off[rows]
        part = {k: s[k][rows] for k in ("states", "picks", "n_picks", "mover")}
        part |= {k: s[k][steps] for k in ("index", "policy")}
        part["offsets"] = np.r_[0, np.cumsum(lens)]
        yield int(g), part


# --- Playing --------------------------------------------------------------------


@dataclass(slots=True)
class FinishedGame:
    samples: dict[str, np.ndarray]  # as game_samples
    record: _core.GameRecord
    size: int
    first: int  # the player who moved first


class Games:
    """Self-play games, ``parallel`` at once, one ``BatchedSearch`` per board bucket.

    ``round`` walks every search to its next leaves, has ``evaluator`` score
    them and hands back the games that ended.
    """

    def __init__(
        self,
        evaluator: NetEvaluator,
        starts: Iterator[_core.GameState],
        config: _core.SearchConfig,
        parallel: int,
        fast_evaluator: NetEvaluator | None = None,
    ):
        self.evaluator = evaluator
        self.fast_evaluator = fast_evaluator  # scores the fast searches' leaves, if given
        self.starts = starts
        self.config = config
        self.parallel = parallel
        self.searches: dict[int, _core.BatchedSearch] = {}
        self.pending: dict[tuple[int, int], list[dict[str, np.ndarray]]] = defaultdict(list)
        self.first: dict[tuple[int, int], int] = {}
        self.evals = 0
        self.fast_evals = 0  # of those, by fast_evaluator

    @property
    def active(self) -> int:
        return sum(bs.active for bs in self.searches.values())

    def _search(self, size: int) -> _core.BatchedSearch:
        if size not in self.searches:
            cfg = _core.SearchConfig(**{k: getattr(self.config, k) for k in _SEARCH_FIELDS})
            cfg.seed = self.config.seed * 31 + size
            self.searches[size] = _core.BatchedSearch(cfg, size)
        return self.searches[size]

    def _evaluate(self, bs: _core.BatchedSearch) -> tuple[np.ndarray, np.ndarray]:
        planes, globals_, legal, movers = bs.batch()
        if self.fast_evaluator is None:
            return self.evaluator(planes, globals_, legal, movers)
        fast = bs.fast_leaves().astype(bool)
        logits = np.empty(legal.shape, np.float32)
        values = np.empty(len(planes), np.float32)
        for evaluator, rows in ((self.evaluator, ~fast), (self.fast_evaluator, fast)):
            if rows.any():
                logits[rows], values[rows] = evaluator(planes[rows], globals_[rows], legal[rows], movers[rows])
        self.fast_evals += int(fast.sum())
        return logits, values

    def round(self) -> list[FinishedGame]:
        while self.active < self.parallel:
            start = next(self.starts)
            md = _core.find_map(start.map_id)
            assert md is not None
            size = bucket_for(md.height, md.width)
            self.first[size, self._search(size).add_game(start)] = start.active
        done = []
        for size, bs in self.searches.items():
            n = bs.collect()
            if n:
                logits, values = self._evaluate(bs)
                bs.submit(logits, values)
                self.evals += n
            samples = bs.take_samples()
            if len(samples["game"]):
                for g, part in _split_by_game(samples):
                    self.pending[size, g].append(part)
            for rec in bs.take_records():
                parts = self.pending.pop((size, rec.game), [])
                first = self.first.pop((size, rec.game))
                if parts:
                    done.append(FinishedGame(game_samples(parts, rec.winner, size), rec, size, first))
        return done


_SEARCH_FIELDS = (
    "sims",
    "fast_sims",
    "full_fraction",
    "skip_prior",
    "considered",
    "gumbel_scale",
    "visit_scale",
    "value_scale",
    "rescale",
    "pick_values",
    "max_days",
    "max_actions",
    "resign_value",
    "resign_turns",
    "resign_off",
    "movement",
    "seed",
)


def human_starts(ds: Dataset, val_fraction: float = 0.05) -> np.ndarray:
    """First states [N, STATE_BYTES] of the dataset's training games."""
    starts = []
    for path in ds.shards:
        with np.load(path) as f:
            train = ~in_validation(f["g_id"], val_fraction)
            starts.append(f["state"][f["g_start"][train]])
    if not starts or not sum(map(len, starts)):
        raise ValueError(f"{ds.root}: no training games")
    return np.concatenate(starts)


def cycle(states: np.ndarray, rng: np.random.Generator) -> Iterator[_core.GameState]:
    """The states in random order, round and round."""
    while True:
        for i in rng.permutation(len(states)):
            yield _core.GameState.from_bytes(states[i].tobytes())


# --- Training -------------------------------------------------------------------


def _decision_tensors(w: Window, rows: np.ndarray, size: int, movement: bool, device: torch.device):
    planes, globals_, nonspatial, _ = encode_decisions(
        w.state[rows], w.picks[rows], w.n_picks[rows], size, targets=False, movement=movement
    )
    return {
        "planes": torch.from_numpy(planes).to(device),
        "globals": torch.from_numpy(globals_).to(device),
        "nonspatial": torch.from_numpy(nonspatial).to(device),
        "index": torch.from_numpy(w.index[rows].astype(np.int64)).to(device),
        "prob": torch.from_numpy(w.prob[rows].astype(np.float32)).to(device),
        "z": torch.from_numpy(w.z[rows].astype(np.int64) + 1).to(device),  # -1/0/+1 -> LOSS/DRAW/WIN
    }


def selfplay_losses(logp: torch.Tensor, value: torch.Tensor, t: dict[str, torch.Tensor]):
    """Per sample: the cross-entropy of the improved policy, its entropy (so
    cross-entropy - entropy is the KL) and the value's cross-entropy."""
    target = t["prob"]
    picked = logp.gather(1, t["index"])
    ce = -torch.where(target > 0, target * picked, 0).sum(1)
    entropy = -torch.where(target > 0, target * target.clamp_min(1e-12).log(), 0).sum(1)
    return ce, entropy, F.cross_entropy(value.float(), t["z"], reduction="none")


@torch.no_grad()
def fresh_losses(net: Any, g: FinishedGame, movement: bool, device: torch.device, batch_size: int = 512):
    """The network's mean policy cross-entropy, KL and value cross-entropy on a
    game it hasn't trained on."""
    w = Window(len(g.samples["z"]))
    w.add(g.samples)
    sums = np.zeros(3)
    for b in range(0, w.count, batch_size):
        rows = np.arange(b, min(w.count, b + batch_size))
        t = _decision_tensors(w, rows, g.size, movement, device)
        logp, value, _ = _forward(net, t, None)
        ce, ent, vce = selfplay_losses(logp, value, t)
        sums += [ce.sum().item(), (ce - ent).sum().item(), vce.sum().item()]
    return [float(x) for x in sums / w.count]


def _human_stream(ds: Dataset, cfg: SelfPlayConfig, movement: bool) -> Iterator[DecisionBatch]:
    """Human training decisions, pass after pass."""
    epoch = 0
    while True:
        yield from decision_batches(
            ds,
            cfg.batch_size,
            split="train",
            seed=cfg.seed * 1000 + epoch,
            fraction=0.1,
            drop_last=True,
            movement=movement,
        )
        epoch += 1


class Stats:
    """Sums since the last log line."""

    def __init__(self) -> None:
        self.v: dict[str, float] = defaultdict(float)

    def add(self, **kv: float) -> None:
        for k, x in kv.items():
            self.v[k] += x

    def mean(self, key: str, by: str) -> float:
        return float(self.v[key] / self.v[by]) if self.v[by] else math.nan

    def clear(self) -> None:
        self.v.clear()


def train(
    init: Path,
    out: Path,
    cfg: SelfPlayConfig,
    *,
    data: Path,
    fast: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Self-play from ``init`` (a policy checkpoint), or with ``resume`` from
    ``out/last.pt``, until ``max_steps`` (forever if 0). ``fast``: a policy
    checkpoint whose network scores the fast searches' leaves (with ``resume``,
    ``out/fast-last.pt`` if there is one)."""
    device = torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset(data)
    step = 0
    if resume and (out / "last.pt").exists():
        net, info = load_policy(out / "last.pt", device)
        step = info["progress"]["step"]
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=device, weights_only=True))
        log(f"resumed from {out / 'last.pt'} at step {step}")
    else:
        net, _ = load_policy(init, device)
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
    assert isinstance(net, PolicyValueNet)
    net.eval()  # no dropout or batch norm: the mode changes nothing, and the compiled search keeps one
    movement = net.cfg.movement
    fast_net = fast_opt = fast_evaluator = None
    if fast is not None:
        resumed = resume and (out / "fast-last.pt").exists()
        fast_net, _ = load_policy(out / "fast-last.pt" if resumed else fast, device)
        assert isinstance(fast_net, PolicyValueNet)
        if fast_net.cfg.movement != movement:
            raise ValueError("the fast network must read the planes the main one does (movement)")
        fast_net.eval()
        if cfg.fast_train:
            fast_opt = _optimizer(fast_net, cfg)  # type: ignore[arg-type]
            if resumed:
                state = torch.load(out / "fast-optimizer.pt", map_location=device, weights_only=True)
                fast_opt.load_state_dict(state)
        fast_evaluator = NetEvaluator(fast_net, device, ratings=None, bf16=cfg.bf16, compile=cfg.compile)
    if cfg.compile:
        # The search's and training's compiled graphs of the networks count together.
        limit = 128 if fast_net is not None else 64
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
    evaluator = NetEvaluator(net, device, ratings=None, bf16=cfg.bf16, compile=cfg.compile)
    forward = torch.compile(net, dynamic=False) if cfg.compile else net
    fast_forward = torch.compile(fast_net, dynamic=False) if cfg.compile and fast_opt is not None else fast_net
    settings = SearchSettings(
        sims=cfg.sims,
        considered=cfg.considered,
        gumbel_scale=cfg.gumbel_scale,
        max_days=cfg.max_days,
        value_scale=cfg.value_scale,
        skip_prior=cfg.skip_prior,
    )
    search_cfg = settings.config(
        movement,
        cfg.seed * 1_000_003 + step,
        fast_sims=cfg.fast_sims,
        full_fraction=cfg.full_fraction,
        resign_value=cfg.resign_value,
        resign_turns=cfg.resign_turns,
        resign_off=cfg.resign_off,
    )
    rng = np.random.default_rng([cfg.seed, step])
    starts = human_starts(ds)
    games = Games(evaluator, cycle(starts, rng), search_cfg, cfg.parallel, fast_evaluator)
    window = Window(cfg.window)
    human = prefetch(_human_stream(ds, cfg, movement)) if cfg.human_fraction > 0 else None
    setup = {"selfplay": asdict(cfg), "init": str(init), "dataset": str(data)}
    if fast is not None:
        setup["fast"] = str(fast)
    params = sum(p.numel() for p in net.parameters())
    log(
        f"{params / 1e6:.2f}M parameters on {device}; {len(starts):,} starts, {cfg.parallel} games at once, "
        f"{cfg.sims} sims ({cfg.full_fraction:.0%} of searches, else {cfg.fast_sims}"
        + (f", unsearched from a prior of {cfg.skip_prior:g}" if cfg.skip_prior > 0 else "")
        + "); "
        f"window {cfg.window:,}, ratio {cfg.train_ratio:g}, human batches {cfg.human_fraction:.0%}"
    )
    if fast_net is not None:
        fast_params = sum(p.numel() for p in fast_net.parameters())
        trained_too = "trained alongside" if fast_opt is not None else "fixed"
        log(f"fast searches scored by a {fast_params / 1e6:.2f}M-parameter network, {trained_too}")
    use_bf16 = cfg.bf16 and device.type == "cuda"
    trained = 0.0  # self-play samples trained on, against window.added * train_ratio
    stats, totals = Stats(), Stats()
    t0 = t_log = time.monotonic()
    evals_log = fast_evals_log = 0

    def batch_loss(fwd: Any, t: dict[str, torch.Tensor], rating, k: int, human_batch: bool, valued, prefix: str):
        """The batch's loss for one network, its parts added to the stats under ``prefix``."""
        with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
            logp, value, _ = _forward(fwd, t, rating, k)
        if human_batch:
            vce = F.cross_entropy(value.float(), t["z"], reduction="none")
            policy_loss = -logp.gather(1, t["target"][:, None]).mean()
            parts = {"h_steps": 1, "h_nll": policy_loss.item(), "h_valued": valued.sum().item()}
            parts["h_vce"] = (vce * valued).sum().item()
        else:
            ce, ent, vce = selfplay_losses(logp, value, t)
            policy_loss = ce.mean()
            parts = {"s_steps": 1, "s_ce": policy_loss.item(), "s_kl": (ce - ent).mean().item()}
            parts |= {"s_vce": (vce * valued).sum().item(), "s_valued": valued.sum().item()}
        stats.add(**{prefix + key: x for key, x in parts.items()})
        value_loss = (vce * valued).sum() / valued.sum().clamp_min(1)
        return policy_loss + cfg.value_weight * value_loss

    def update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def train_step(human_batch: bool) -> None:
        nonlocal step
        warm = min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
        for optimizer in (opt, fast_opt):
            for group in optimizer.param_groups if optimizer is not None else ():
                group["lr"] = cfg.lr * warm
        k = int(rng.integers(8)) if cfg.augment else 0
        if human_batch:
            assert human is not None
            b = next(human)
            t = _to_device(b, device)
            hide = torch.from_numpy(rng.random(len(b.target)) < cfg.rating_dropout).to(device)
            rating = t["rating"].masked_fill(hide[:, None], -1)
            valued = torch.from_numpy(rng.random(len(b.target)) < cfg.human_value_fraction).to(device)
        else:
            size, rows = window.sample(cfg.batch_size, rng)
            t = _decision_tensors(window, rows, size, movement, device)
            rating = None
            valued = torch.from_numpy(rng.random(len(rows)) < cfg.value_fraction).to(device)
        update(net, opt, batch_loss(forward, t, rating, k, human_batch, valued, ""))
        if fast_net is not None and fast_opt is not None:
            update(fast_net, fast_opt, batch_loss(fast_forward, t, rating, k, human_batch, valued, "fast_"))
        step += 1

    def checkpoint(name: str, metrics: dict[str, Any]) -> None:
        progress = {"step": step, "samples": window.added, "games": int(totals.v["games"])}
        # Plain floats: torch.load(weights_only=True) refuses numpy scalars.
        save_policy(out / name, net, metrics=json.loads(json.dumps(metrics)), progress=progress, **setup)

    def save_last(metrics: dict[str, Any]) -> None:
        torch.save(opt.state_dict(), out / "optimizer.pt")
        checkpoint("last.pt", metrics)
        if fast_net is not None and fast_opt is not None:
            torch.save(fast_opt.state_dict(), out / "fast-optimizer.pt")
            progress = {"step": step, "samples": window.added, "games": int(totals.v["games"])}
            save_policy(out / "fast-last.pt", fast_net, progress=progress, **setup)

    def report() -> dict[str, Any]:
        nonlocal t_log, evals_log, fast_evals_log
        now = time.monotonic()
        dt = now - t_log
        v = stats.v
        m: dict[str, Any] = {
            "step": step,
            "time": time.time(),
            "hours": (now - t0) / 3600,
            "games": int(totals.v["games"]),
            "samples": window.added,
            "window": window.count,
            "evals_per_s": (games.evals - evals_log) / dt,
            "fast_evals_per_s": (games.fast_evals - fast_evals_log) / dt,  # of those, by the fast network
            "skipped": stats.mean("skipped", "decisions"),  # share of searches that ended at the root
            "games_per_h": v["games"] * 3600 / dt,
            "samples_per_h": v["samples"] * 3600 / dt,
            "days": stats.mean("days", "games"),
            "capped": stats.mean("capped", "games"),
            "resigned": stats.mean("resigned", "games"),
            # Games that played on where the rule would have resigned, and those the
            # would-be resigner won, since the run (or its resume) started.
            "resign_checked": int(totals.v["resign_checked"]),
            "resign_wrong": int(totals.v["resign_wrong"]),
            "first_player_won": stats.mean("p1_won", "games"),
            "drawn": stats.mean("drawn", "games"),
            "decisions": stats.mean("decisions", "games"),
            "train": {
                "steps": int(v["s_steps"]),
                "policy_ce": stats.mean("s_ce", "s_steps"),
                "policy_kl": stats.mean("s_kl", "s_steps"),
                "value_ce": stats.mean("s_vce", "s_valued"),
            },
            "human": {
                "steps": int(v["h_steps"]),
                "nll": stats.mean("h_nll", "h_steps"),
                "value_ce": stats.mean("h_vce", "h_valued"),
            },
            "fresh": {
                "policy_ce": stats.mean("f_ce", "f_n"),
                "policy_kl": stats.mean("f_kl", "f_n"),
                "value_ce": stats.mean("f_vce", "f_n"),
            },
            "lr": opt.param_groups[0]["lr"],
        }
        if fast_opt is not None:
            m["fast"] = {
                "train": {
                    "policy_ce": stats.mean("fast_s_ce", "fast_s_steps"),
                    "policy_kl": stats.mean("fast_s_kl", "fast_s_steps"),
                    "value_ce": stats.mean("fast_s_vce", "fast_s_valued"),
                },
                "human": {
                    "nll": stats.mean("fast_h_nll", "fast_h_steps"),
                    "value_ce": stats.mean("fast_h_vce", "fast_h_valued"),
                },
            }
        tr, hu, fr = m["train"], m["human"], m["fresh"]
        log(
            f"step {step} | {m['games']:,} games, +{m['games_per_h']:,.0f}/h, {m['days']:.1f} days, "
            f"capped {m['capped']:.0%}, resigned {m['resigned']:.0%} (wrong {m['resign_wrong']}/"
            f"{m['resign_checked']}), P1 won {m['first_player_won']:.0%} | window {window.count:,} "
            f"(+{m['samples_per_h']:,.0f}/h) | {m['evals_per_s']:,.0f} evals/s"
            + (f" ({m['fast_evals_per_s']:,.0f} fast)" if fast_net is not None else "")
            + (f", {m['skipped']:.0%} unsearched" if cfg.skip_prior > 0 else "")
            + " | "
            f"train ce {tr['policy_ce']:.3f} kl {tr['policy_kl']:.3f} v {tr['value_ce']:.3f} | "
            f"fresh ce {fr['policy_ce']:.3f} kl {fr['policy_kl']:.3f} v {fr['value_ce']:.3f} | "
            f"human nll {hu['nll']:.3f} v {hu['value_ce']:.3f}"
            + (
                f" | fast: train ce {m['fast']['train']['policy_ce']:.3f} human nll {m['fast']['human']['nll']:.3f}"
                if "fast" in m
                else ""
            )
        )
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m) + "\n")
        stats.clear()
        t_log, evals_log, fast_evals_log = now, games.evals, games.fast_evals
        return m

    last: dict[str, Any] = {}
    while not cfg.max_steps or step < cfg.max_steps:
        for g in games.round():
            n = len(g.samples["z"])
            ce, kl, vce = fresh_losses(net, g, movement, device)
            rec = g.record
            totals.add(
                games=1,
                resign_checked=rec.would_resign >= 0,
                resign_wrong=rec.would_resign >= 0 and rec.winner == rec.would_resign,
            )
            stats.add(
                games=1,
                samples=n,
                days=rec.days,
                capped=rec.capped,
                resigned=rec.resigned,
                p1_won=rec.winner == g.first,
                drawn=rec.winner == _core.DRAW,
                decisions=rec.decisions,
                skipped=rec.skipped,
                f_ce=ce * n,
                f_kl=kl * n,
                f_vce=vce * n,
                f_n=n,
            )
            window.add(g.samples)
        if window.count >= cfg.min_window:
            while trained < window.added * cfg.train_ratio and (not cfg.max_steps or step < cfg.max_steps):
                if human is not None and rng.random() < cfg.human_fraction:
                    train_step(True)
                else:
                    train_step(False)
                    trained += cfg.batch_size
                if step % cfg.checkpoint_every == 0:
                    checkpoint(f"net-{step:06d}.pt", last)
        else:
            trained = window.added * cfg.train_ratio  # training starts from here, not to catch up
        if time.monotonic() - t_log >= cfg.log_every:
            last = report()
            if step:
                save_last(last)
    last = report()
    save_last(last)
