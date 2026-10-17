"""Self-play search throughput (PLAN.md §8): evaluations/s, the network's
padding and the time spent waiting for it, for each number of games and group
size, with no training.

    uv run --frozen python bench/selfplay.py CHECKPOINT --data cache/datasets/v1-full \\
        --compile --parallel 512 1024 2048 --group-size 256

Games start at turn starts spread over the dataset, so they are at all stages
and end at different times. ``--waiting`` also runs each setting with every
batch waited for as it is queued: fixed slots without the overlap.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from awbw.data.shards import Dataset
from awbw.nn.critic import pick_device
from awbw.nn.policy import load_policy
from awbw.nn.train_selfplay import Games
from awbw.search import NetEvaluator, Pending, SearchSettings


class Waiting(NetEvaluator):
    """Waits for each batch as it queues it."""

    waited = 0.0

    def launch(self, *args, **kwargs) -> Pending:
        pending = super().launch(*args, **kwargs)
        pending.result()
        self.waited += pending.waited
        return pending


def turn_starts(ds: Dataset, shards: int, per_shard: int, rng: np.random.Generator) -> np.ndarray:
    """States at the start of a turn, ``per_shard`` from each of ``shards`` spread over ``ds``."""
    out = []
    for path in ds.shards[:: max(1, len(ds.shards) // shards)][:shards]:
        with np.load(path) as f:
            game, turn = f["game"], f["turn"]
            first = np.flatnonzero(np.r_[True, (game[1:] != game[:-1]) | (turn[1:] != turn[:-1])])
            out.append(f["state"][rng.choice(first, min(per_shard, len(first)), replace=False)])
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("cache/datasets/v1-full"))
    ap.add_argument("--shards", type=int, default=60, help="spread over the dataset, starts from each (default 60)")
    ap.add_argument("--per-shard", type=int, default=100)
    ap.add_argument("--parallel", type=int, nargs="+", default=[512])
    ap.add_argument("--group-size", type=int, nargs="+", default=[256])
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--fast-sims", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30, help="rounds first, compiling included (default 30)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--waiting", action="store_true", help="also each setting waiting for every batch at once")
    args = ap.parse_args()

    device = pick_device(args.device)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    ds = Dataset(args.data)
    starts = turn_starts(ds, args.shards, args.per_shard, np.random.default_rng(1))
    net, _ = load_policy(args.checkpoint, device)
    settings = SearchSettings(sims=args.sims, considered=16, max_days=30, value_scale=0.1)
    cfg = settings.config(
        net.cfg.movement, 1, fast_sims=args.fast_sims, full_fraction=0.25, resign_value=-0.9, resign_turns=2
    )
    print(f"{args.checkpoint} on {device}; {len(starts):,} turn starts from {args.data}", flush=True)
    for parallel in args.parallel:
        for group_size in args.group_size:
            for waiting in (False, True) if args.waiting else (False,):
                evaluator = (Waiting if waiting else NetEvaluator)(net, device, compile=args.compile)
                games = Games(evaluator, starts, np.random.default_rng(0), cfg, parallel, group_size=group_size)
                for _ in range(args.warmup):
                    games.round()
                sync()
                evals, rows, busy = evaluator.evals, evaluator.rows, evaluator.busy
                waited = games.waited + getattr(evaluator, "waited", 0.0)
                ended = 0
                t = time.perf_counter()
                for _ in range(args.rounds):
                    ended += len(games.round())
                sync()
                wall = time.perf_counter() - t
                n, rows = evaluator.evals - evals, evaluator.rows - rows
                waited = games.waited + getattr(evaluator, "waited", 0.0) - waited
                busy_note = f", GPU busy {(evaluator.busy - busy) / wall:.0%}" if device.type == "cuda" else ""
                print(
                    f"{parallel:5d} games, groups {[g.slots for g in games.groups]}"
                    f"{', waiting' if waiting else ''}: {n / wall:8,.0f} evals/s, padding {1 - n / rows:.1%}, "
                    f"waiting {waited / wall:.0%}{busy_note} ({args.rounds} rounds, {wall:.0f}s, {ended} games ended)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
