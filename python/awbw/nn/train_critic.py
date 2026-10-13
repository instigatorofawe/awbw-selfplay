"""Train the critic on a v0 dataset: turn-start positions labelled with the game's result.

The loss is cross-entropy over (loss, draw, win) for the player to move.
Each batch gets one of the 8 flips and transposes of its square bucket.
Validation games are a fixed 5% (``loader.in_validation``), so metrics are
comparable across runs and dataset rebuilds.

An output directory holds ``metrics.jsonl`` (one line per evaluation),
``critic-epNN.pt`` after each epoch, ``best.pt`` (lowest validation loss) and
``last.pt`` (also the optimizer and progress, for ``resume``).
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import Batch, batches, in_validation
from ..data.shards import Dataset, Shard
from .critic import load_critic, save_critic
from .model import LOSS, WIN, CriticConfig, CriticNet

DAY_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 20), (21, 999))


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 300
    min_rating: int = 0  # keep games whose players are both rated at least this
    augment: bool = True
    compile: bool = False  # torch.compile the network (faster on GPUs; compiles once per bucket)
    val_fraction: float = 0.05
    eval_every: int = 0  # steps between evaluations; 0 = once per epoch
    log_every: int = 200
    seed: int = 0


def rating_filter(min_rating: int) -> Callable[[Shard], np.ndarray] | None:
    """A ``keep`` mask for games whose players are both rated at least ``min_rating``."""
    if min_rating <= 0:
        return None
    return lambda sh: (sh.g_rating[sh.game] >= min_rating).all(1)


def count_samples(ds: Dataset, split: str, keep: Callable[[Shard], np.ndarray] | None, val_fraction: float) -> int:
    n = 0
    for path in ds.shards:
        sh = ds.load(path)
        val = in_validation(sh.g_id, val_fraction)[sh.game]
        mask = val if split == "val" else ~val
        if keep is not None:
            mask &= keep(sh)
        n += int(mask.sum())
    return n


def dihedral(planes: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 symmetries of a square board [..., S, S]: bits flip x, flip y, transpose."""
    if k & 1:
        planes = planes.flip(-1)
    if k & 2:
        planes = planes.flip(-2)
    if k & 4:
        planes = planes.transpose(-1, -2)
    return planes.contiguous()


def prefetch[T](items: Iterable[T], depth: int = 4) -> Iterator[T]:
    """Run an iterator in a background thread (encoding and decompression release the GIL)."""
    q: queue.Queue = queue.Queue(depth)
    done = object()

    def run() -> None:
        try:
            for item in items:
                q.put(item)
        except BaseException as e:  # handed to the consumer
            q.put(e)
        q.put(done)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


def _to_device(b: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    planes = torch.from_numpy(b.planes).to(device)
    globals_ = torch.from_numpy(b.globals).to(device)
    target = torch.from_numpy(b.z.astype(np.int64) + 1).to(device)  # z -1/0/+1 -> LOSS/DRAW/WIN
    return planes, globals_, target


def calibration(p_win: np.ndarray, won: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """Per bin of predicted P(win): (mean prediction, observed win rate, count)."""
    edges = np.minimum((p_win * bins).astype(int), bins - 1)
    out = []
    for i in range(bins):
        m = edges == i
        if m.any():
            out.append((float(p_win[m].mean()), float(won[m].mean()), int(m.sum())))
    return out


def metrics(prob: np.ndarray, z: np.ndarray, day: np.ndarray) -> dict[str, Any]:
    """Log loss, accuracy, Brier score, calibration error and the same by game day."""
    target = z.astype(np.int64) + 1
    nll = -np.log(np.clip(prob[np.arange(len(z)), target], 1e-7, 1.0))
    value = prob[:, WIN] - prob[:, LOSS]
    decisive = z != 0
    correct = np.sign(value) == z
    won = (z == 1).astype(np.float64)
    cal = calibration(prob[:, WIN], won)
    ece = sum(n * abs(pred - obs) for pred, obs, n in cal) / max(len(z), 1)
    by_day = []
    for lo, hi in DAY_BANDS:
        m = (day >= lo) & (day <= hi)
        if m.any():
            by_day.append(
                {
                    "days": f"{lo}-{hi}" if hi < 999 else f"{lo}+",
                    "n": int(m.sum()),
                    "loss": float(nll[m].mean()),
                    "accuracy": float(correct[m & decisive].mean()) if (m & decisive).any() else None,
                }
            )
    return {
        "n": len(z),
        "loss": float(nll.mean()),
        "accuracy": float(correct[decisive].mean()) if decisive.any() else None,
        "brier": float(np.mean((prob[:, WIN] - won) ** 2)),
        "ece": float(ece),
        "by_day": by_day,
        "calibration": cal,
    }


@torch.no_grad()
def predict(
    net: CriticNet,
    ds: Dataset,
    device: torch.device,
    *,
    split: str = "val",
    keep: Callable[[Shard], np.ndarray] | None = None,
    val_fraction: float = 0.05,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(loss, draw, win) probabilities, results and days for every sample of ``split``."""
    was_training = net.training
    net.eval()
    probs, zs, days = [], [], []
    for b in prefetch(batches(ds, batch_size, split=split, val_fraction=val_fraction, keep=keep, shuffle=False)):
        planes, globals_, _ = _to_device(b, device)
        probs.append(net(planes, globals_).float().softmax(-1).cpu().numpy())
        zs.append(b.z)
        days.append(b.day)
    net.train(was_training)
    if not probs:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int16)
    return np.concatenate(probs), np.concatenate(zs), np.concatenate(days)


def _optimizer(net: CriticNet, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for n, p in net.named_parameters() if p.ndim >= 2 and "table" not in n]
    other = [p for n, p in net.named_parameters() if not (p.ndim >= 2 and "table" not in n)]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def _lr(cfg: TrainConfig, step: int, progress: float) -> float:
    """Linear warmup, then cosine decay to zero over the planned samples."""
    warm = min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
    return cfg.lr * warm * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(
    data: Path,
    out: Path,
    cfg: TrainConfig,
    model_cfg: CriticConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train (or with ``resume``, continue from ``out/last.pt``); returns the last evaluation."""
    device = torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset(data)
    keep = rating_filter(cfg.min_rating)
    n_train = count_samples(ds, "train", keep, cfg.val_fraction)
    if n_train == 0:
        raise ValueError(f"{data}: no training samples (min rating {cfg.min_rating})")
    planned = cfg.epochs * n_train

    epoch = step = seen = 0
    best = math.inf
    if resume and (out / "last.pt").exists():
        net, info = load_critic(out / "last.pt", device)
        progress = info["progress"]
        epoch, step, seen, best = progress["epoch"], progress["step"], progress["seen"], progress["best"]
        opt = _optimizer(net, cfg)
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=device, weights_only=True))
        log(f"resumed from {out / 'last.pt'} after epoch {epoch} (step {step})")
    else:
        net = CriticNet(model_cfg or CriticConfig()).to(device)
        opt = _optimizer(net, cfg)
    net.train()
    forward = torch.compile(net, dynamic=False) if cfg.compile else net
    params = sum(p.numel() for p in net.parameters())
    games = ds.meta["games"]
    log(f"{params / 1e6:.2f}M parameters on {device}; {n_train:,} training samples per epoch; {games:,} games")
    setup = {"train": asdict(cfg), "dataset": str(data), "dataset_games": ds.meta["games"], "train_samples": n_train}
    rng = np.random.default_rng(cfg.seed)
    last: dict[str, Any] = {}

    def evaluate() -> dict[str, Any]:
        prob, z, day = predict(net, ds, device, keep=keep, val_fraction=cfg.val_fraction)
        m = metrics(prob, z, day)
        m.update(epoch=round(seen / n_train, 3), step=step, seen=seen, time=time.time())
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m) + "\n")
        log(f"  val: loss {m['loss']:.4f} accuracy {m['accuracy'] or 0:.3f} brier {m['brier']:.4f} ece {m['ece']:.4f}")
        log("  by day: " + "  ".join(f"{d['days']}: {d['loss']:.3f}" for d in m["by_day"]))
        return m

    while epoch < cfg.epochs:
        t0, t_log, n_log = time.perf_counter(), time.perf_counter(), 0
        loss_sum = torch.zeros((), device=device)
        stream = batches(
            ds,
            cfg.batch_size,
            split="train",
            val_fraction=cfg.val_fraction,
            keep=keep,
            seed=cfg.seed * 1000 + epoch,
            drop_last=True,  # one batch shape per bucket, so a compiled network compiles once each
        )
        for b in prefetch(stream):
            for group in opt.param_groups:
                group["lr"] = _lr(cfg, step, seen / planned)
            planes, globals_, target = _to_device(b, device)
            if cfg.augment:
                planes = dihedral(planes, int(rng.integers(8)))
            loss = F.cross_entropy(forward(planes, globals_).float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_sum += loss.detach() * len(target)
            step += 1
            seen += len(target)
            n_log += len(target)
            if step % cfg.log_every == 0:
                dt = time.perf_counter() - t_log
                log(
                    f"epoch {seen / n_train:.2f} step {step}: loss {loss_sum.item() / n_log:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {n_log / dt:.0f} samples/s"
                )
                loss_sum.zero_()
                t_log, n_log = time.perf_counter(), 0
            if cfg.eval_every and step % cfg.eval_every == 0:
                last = evaluate()
        epoch += 1
        log(f"epoch {epoch} done in {time.perf_counter() - t0:.0f}s")
        last = evaluate()
        if last["loss"] < best:
            best = last["loss"]
            save_critic(out / "best.pt", net, metrics=last, **setup)
        save_critic(out / f"critic-ep{epoch:02d}.pt", net, metrics=last, **setup)
        progress = {"epoch": epoch, "step": step, "seen": seen, "best": best}
        torch.save(opt.state_dict(), out / "optimizer.pt")
        save_critic(out / "last.pt", net, metrics=last, progress=progress, **setup)
    return last
