"""Pre-train the policy-value network on a v1 dataset: behaviour cloning plus the result.

Each training sample is one micro-step decision of an action a player took
(``loader.decision_batches``; forced steps are left out). The policy loss is
the cross-entropy of the step taken, over the steps legal there. The value
loss is v0's cross-entropy over (loss, draw, win), on a random
``value_fraction`` of the decisions only: a game has about 1,100 decisions
and they all share its result, so valuing every one would show each result
some 40 times as often as the critic sees it in v0 (one sample per turn),
and the critic already overfits v0 in its third epoch.

A network with ``PolicyConfig.aux`` also learns what happens next in the
game (``loader.aux_targets``) on a random ``aux_fraction`` of the decisions:
a Huber loss over the scaled changes in army value and unit count, plus the
cross-entropy of the power-bar targets where they count, weighted by
``aux_weight``. These shape the network's features; nothing reads the head.

An epoch is a random ``fraction`` of the training decisions. Each batch gets
one of the 8 flips and transposes of its square bucket; the spatial policy
logits are mapped back before the loss. Ratings are hidden at random as for
the critic, so the policy can imitate players of a given strength or of any.
On CUDA the forward pass runs in bfloat16 (``bf16``); the losses and
validation stay in float32.

Validation scores the decisions of a fixed ``val_rows`` share of the
validation games' actions (by phase, against a uniform choice among the
legal steps, and whole actions), and values every validation turn start, as
``eval-critic`` does. Those decisions are read from the shards once and kept
in memory (about 1.5 GiB for v1-70k). An output directory holds
``metrics.jsonl``, ``policy-epNN.pt``, ``best.pt`` (lowest validation policy
loss) and ``last.pt`` + ``optimizer.pt`` for ``resume``. ``last.pt`` is also
saved between groups of shards every ``checkpoint_minutes``, so a resumed run
redoes at most that much.

The learning rate warms up over ``warmup_steps``, then falls to zero along a
cosine over the planned decisions, or (``schedule`` "wsd") stays flat until
the last ``decay_epochs`` and falls linearly over them. While a "wsd" rate is
flat, a run can be resumed with other ``epochs``, which may end partway
through an epoch (``policy-ep04.20.pt``). ``pause_at`` stops a run after that
many epochs. A copy of its ``last.pt`` and ``optimizer.pt`` resumed with
``epochs`` a little past the pause, decaying over that little, then gives a
finished network to evaluate, and the run itself resumes where it paused.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .. import _core
from ..data.loader import (
    AUX_BINARY,
    AUX_HORIZONS,
    AUX_REGRESSION,
    DecisionBatch,
    decision_batches,
    in_subset,
    in_validation,
)
from ..data.shards import Dataset, Shard
from .model import NUM_AUX_REGRESSION, PolicyConfig, PolicyValueNet, flat_policy, legal_mask, masked_log_softmax
from .policy import load_policy, save_policy
from .train_critic import _lr, _optimizer, dihedral, metrics, prefetch, rating_filter

PHASE_NAMES = tuple(_core.PHASES)
SCHEDULES = ("cosine", "wsd")


@dataclass
class PolicyTrainConfig:
    epochs: float = 2  # the run's length; a fraction of an epoch ends the last one early
    fraction: float = 0.25  # share of the training decisions in each epoch
    batch_size: int = 256
    lr: float = 1e-3
    schedule: str = "cosine"  # or "wsd": flat after the warmup, then a linear fall over decay_epochs
    decay_epochs: float = 1.0  # "wsd": the epochs at the end over which the rate falls to zero
    weight_decay: float = 0.01
    warmup_steps: int = 500
    value_weight: float = 1.0
    value_fraction: float = 0.05  # share of decisions whose result trains the value head
    aux_weight: float = 1.0  # weight of the auxiliary losses (networks with PolicyConfig.aux)
    aux_fraction: float = 0.25  # share of decisions that train the auxiliary head
    min_rating: int = 0  # keep games whose players are both rated at least this
    games_fraction: float = 1.0  # train on this share of the training games (``in_subset``)
    hold_out_maps: tuple[int, ...] = ()  # train on no game on these maps
    game_ids: tuple[int, int] | None = None  # train only on games with lo <= id < hi
    rating_dropout: float = 0.25
    augment: bool = True
    compile: bool = False
    bf16: bool = True  # autocast the forward pass on CUDA (validation stays float32)
    recompute: bool = False  # recompute blocks' activations in the backward pass: less memory, more time
    val_fraction: float = 0.05
    val_rows: float = 0.1  # share of the validation games' actions whose decisions are scored
    shards_at_once: int = 8
    eval_every: int = 0  # steps between evaluations; 0 = once per epoch
    checkpoint_minutes: float = 60.0  # between saves of last.pt within an epoch, at groups' ends; 0: each group
    pause_at: int = 0  # stop after this many epochs, the schedule still planning ``epochs``; 0: don't
    log_every: int = 200
    seed: int = 0


def lr_at(cfg: PolicyTrainConfig, step: int, seen: float, planned: float, per_epoch: float) -> float:
    """The learning rate at ``step``, with ``seen`` of the ``planned`` decisions trained on."""
    if cfg.schedule == "cosine":
        return _lr(cfg, step, seen / planned)  # type: ignore[arg-type]
    warm = min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
    return cfg.lr * warm * min(1.0, max(planned - seen, 0.0) / (cfg.decay_epochs * per_epoch))


def undihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """The inverse of ``dihedral(x, k)``."""
    if k & 4:
        x = x.transpose(-1, -2)
    if k & 2:
        x = x.flip(-2)
    if k & 1:
        x = x.flip(-1)
    return x


def games_keep(
    min_rating: int = 0,
    *,
    fraction: float = 1.0,
    maps: tuple[int, ...] = (),
    exclude_maps: tuple[int, ...] = (),
    game_ids: tuple[int, int] | None = None,
) -> Callable[[Shard], np.ndarray] | None:
    """A ``keep`` mask for the games whose players are both rated at least
    ``min_rating``, in the ``in_subset`` share ``fraction``, on one of ``maps``
    (if any) and none of ``exclude_maps``, with ids in [lo, hi); None keeps all."""
    rated = rating_filter(min_rating)
    if rated is None and fraction >= 1 and not maps and not exclude_maps and game_ids is None:
        return None

    def keep(sh: Shard) -> np.ndarray:
        g = np.ones(len(sh.g_id), bool)
        if fraction < 1:
            g &= in_subset(sh.g_id, fraction)
        if maps:
            g &= np.isin(sh.g_map, maps)
        if exclude_maps:
            g &= ~np.isin(sh.g_map, exclude_maps)
        if game_ids is not None:
            g &= (sh.g_id >= game_ids[0]) & (sh.g_id < game_ids[1])
        mask = g[sh.game]
        return mask & rated(sh) if rated is not None else mask

    return keep


def training_games(cfg: PolicyTrainConfig) -> Callable[[Shard], np.ndarray] | None:
    """The ``keep`` mask of the games ``cfg`` trains on (validation keeps only ``min_rating``)."""
    return games_keep(
        cfg.min_rating,
        fraction=cfg.games_fraction,
        exclude_maps=tuple(cfg.hold_out_maps),
        game_ids=tuple(cfg.game_ids) if cfg.game_ids is not None else None,  # type: ignore[arg-type]
    )


def count_decisions(ds: Dataset, split: str, keep: Callable[[Shard], np.ndarray] | None, val_fraction: float) -> int:
    """Decisions ``decision_batches`` gives for ``split`` (forced steps left out)."""
    return split_size(ds, split, keep, val_fraction)[1]


def split_size(
    ds: Dataset, split: str, keep: Callable[[Shard], np.ndarray] | None, val_fraction: float
) -> tuple[int, int]:
    """Games and decisions ``decision_batches`` gives for ``split`` (forced steps left out)."""
    games = n = 0
    for path in ds.shards:
        with np.load(path) as f:
            arrays = {k: f[k] for k in f.files if k != "state"}
        sh = Shard(state=np.empty((len(arrays["game"]), 0), np.uint8), **arrays)
        assert sh.n_steps is not None and sh.n_legal is not None
        val = in_validation(sh.g_id, val_fraction)[sh.game]
        mask = val if split == "val" else ~val if split == "train" else np.ones(len(sh), bool)
        if keep is not None:
            mask &= keep(sh)
        taken = np.arange(3)[None, :] < sh.n_steps[:, None]
        n += int((taken & (sh.n_legal > 1))[mask].sum())
        games += len(np.unique(sh.game[mask]))
    return games, n


def _to_device(b: DecisionBatch, device: torch.device) -> dict[str, torch.Tensor]:
    t = {
        "planes": torch.from_numpy(b.planes).to(device),
        "globals": torch.from_numpy(b.globals).to(device),
        "nonspatial": torch.from_numpy(b.nonspatial).to(device),
        "target": torch.from_numpy(b.target).to(device),
        "z": torch.from_numpy(b.z.astype(np.int64) + 1).to(device),  # -1/0/+1 -> LOSS/DRAW/WIN
        "rating": torch.from_numpy(b.rating.astype(np.int32)).to(device),
    }
    if b.aux_reg is not None:
        assert b.aux_bin is not None and b.aux_mask is not None
        t |= {
            "aux_reg": torch.from_numpy(b.aux_reg).to(device),
            "aux_bin": torch.from_numpy(b.aux_bin).to(device),
            "aux_mask": torch.from_numpy(b.aux_mask).to(device),
        }
    return t


def _forward(net: Any, t: dict[str, torch.Tensor], rating: torch.Tensor | None, k: int = 0, aux: bool = False):
    """Log-probabilities over the policy indices, value logits and (with ``aux``)
    the auxiliary predictions, else None, through flip ``k``."""
    out = net(dihedral(t["planes"], k) if k else t["planes"], t["globals"], rating, aux=aux)
    spatial, nonspatial, value = out[:3]
    if k:
        spatial = undihedral(spatial, k)
    legal = legal_mask(t["planes"], t["nonspatial"])
    return masked_log_softmax(flat_policy(spatial, nonspatial), legal), value, out[3] if aux else None


def aux_losses(pred: torch.Tensor, t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per decision and target: the Huber loss of each AUX_REGRESSION change [B, R] and
    the cross-entropy of each AUX_BINARY target [B, K] (count it only under aux_mask)."""
    pred = pred.float()
    reg = F.smooth_l1_loss(pred[:, :NUM_AUX_REGRESSION], t["aux_reg"], reduction="none")
    bce = F.binary_cross_entropy_with_logits(pred[:, NUM_AUX_REGRESSION:], t["aux_bin"], reduction="none")
    return reg, bce


def aux_metrics(pred: np.ndarray, reg: np.ndarray, binary: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Validation metrics of the auxiliary head: per AUX_REGRESSION target, the mean
    squared error against predicting no change; per AUX_BINARY target, the
    cross-entropy and accuracy where it counts against its base rate's entropy."""
    err = pred[:, :NUM_AUX_REGRESSION] - reg
    out: dict[str, Any] = {
        "regression": {
            name: {"mse": float((err[:, i] ** 2).mean()), "no_change": float((reg[:, i] ** 2).mean())}
            for i, name in enumerate(AUX_REGRESSION)
        },
        "binary": {},
    }
    logits = pred[:, NUM_AUX_REGRESSION:]
    for i, name in enumerate(AUX_BINARY):
        m = mask[:, i]
        if not m.any():
            continue
        y, z = binary[m, i], logits[m, i]
        bce = np.logaddexp(0, z) - y * z
        rate = float(y.mean())
        base = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate)) if 0 < rate < 1 else 0.0
        out["binary"][name] = {
            "n": int(m.sum()),
            "loss": float(bce.mean()),
            "accuracy": float(((z > 0) == (y > 0.5)).mean()),
            "rate": rate,
            "base": base,
        }
    return out


def policy_metrics(
    nll: np.ndarray, correct: np.ndarray, n_legal: np.ndarray, phase: np.ndarray, row: np.ndarray
) -> dict[str, Any]:
    """Per decision log loss and accuracy, against a uniform choice among the legal
    steps, by phase; and per action (its decisions' losses summed, all right)."""

    def summary(m: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(m.sum()),
            "loss": float(nll[m].mean()),
            "accuracy": float(correct[m].mean()),
            "uniform": float(np.log(n_legal[m]).mean()),
        }

    everything = np.ones(len(nll), bool)
    out: dict[str, Any] = summary(everything) if len(nll) else {"n": 0}
    out["by_phase"] = {PHASE_NAMES[p]: summary(phase == p) for p in range(len(PHASE_NAMES)) if (phase == p).any()}
    if len(nll):
        _, action = np.unique(row, return_inverse=True)
        out["action_loss"] = float(np.bincount(action, nll).mean())
        out["action_accuracy"] = float((np.bincount(action, ~correct) == 0).mean())
        out["actions"] = int(action.max() + 1)
    return out


@torch.no_grad()
def evaluate_policy(
    net: PolicyValueNet,
    ds: Dataset,
    device: torch.device,
    *,
    keep: Callable[[Shard], np.ndarray] | None = None,
    val_fraction: float = 0.05,
    row_fraction: float = 0.1,
    split: str = "val",
    ratings: bool = True,
    batch_size: int = 512,
    memo: list | None = None,
) -> dict[str, Any]:
    """``policy_metrics`` on a fixed share of ``split``'s actions, and with an
    auxiliary head ``aux_metrics`` under "aux". ``memo`` keeps those decisions
    for the next evaluation (``decision_batches``)."""
    was_training = net.training
    net.eval()
    aux = net.cfg.aux
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("nll", "correct", "n_legal", "phase", "row", "aux", "aux_reg", "aux_bin", "aux_mask")
    }
    stream = decision_batches(
        ds,
        batch_size,
        split=split,
        val_fraction=val_fraction,
        keep=keep,
        shuffle=False,
        row_fraction=row_fraction,
        movement=net.cfg.movement,
        aux=aux,
        memo=memo,
    )
    for b in prefetch(stream):
        t = _to_device(b, device)
        logp, _, pred = _forward(net, t, t["rating"] if ratings else None, aux=aux)
        cols["nll"].append(-logp.gather(1, t["target"][:, None])[:, 0].cpu().numpy())
        cols["correct"].append((logp.argmax(1) == t["target"]).cpu().numpy())
        cols["n_legal"].append(b.n_legal)
        cols["phase"].append(b.phase)
        cols["row"].append(b.row)
        if pred is not None:
            assert b.aux_reg is not None and b.aux_bin is not None and b.aux_mask is not None
            cols["aux"].append(pred.float().cpu().numpy())
            cols["aux_reg"].append(b.aux_reg)
            cols["aux_bin"].append(b.aux_bin)
            cols["aux_mask"].append(b.aux_mask)
    net.train(was_training)
    if not cols["nll"]:
        return {"n": 0, "by_phase": {}}
    out = policy_metrics(*(np.concatenate(cols[k]) for k in ("nll", "correct", "n_legal", "phase", "row")))
    if aux:
        out["aux"] = aux_metrics(*(np.concatenate(cols[k]) for k in ("aux", "aux_reg", "aux_bin", "aux_mask")))
    return out


def aux_summary(m: dict[str, Any]) -> list[str]:
    """``aux_metrics`` in two lines: regression error by horizon (against no change),
    then each power target's cross-entropy (against its base rate's)."""
    by_h = []
    for h in AUX_HORIZONS:
        names = [n for n in AUX_REGRESSION if n.endswith(f"@{h or 'end'}")]
        mse = np.mean([m["regression"][n]["mse"] for n in names])
        zero = np.mean([m["regression"][n]["no_change"] for n in names])
        by_h.append(f"@{h or 'end'} {mse:.3f} ({zero:.3f})")
    powers = [f"{n} {v['loss']:.3f} ({v['base']:.3f})" for n, v in m["binary"].items()]
    return ["mse by horizon (no change): " + "  ".join(by_h), "powers (base rate): " + "  ".join(powers)]


@torch.no_grad()
def evaluate_value(
    net: PolicyValueNet,
    ds: Dataset,
    device: torch.device,
    *,
    keep: Callable[[Shard], np.ndarray] | None = None,
    val_fraction: float = 0.05,
    split: str = "val",
    ratings: bool = True,
    batch_size: int = 512,
    memo: list | None = None,
) -> dict[str, Any]:
    """The critic's metrics (``train_critic.metrics``) over ``split``'s turn
    starts; ``memo`` as in ``evaluate_policy``."""
    was_training = net.training
    net.eval()

    def starts(sh: Shard) -> np.ndarray:
        assert sh.action is not None
        mask = sh.action == 0
        return mask & keep(sh) if keep is not None else mask

    probs, zs, days = [], [], []
    stream = decision_batches(
        ds,
        batch_size,
        split=split,
        val_fraction=val_fraction,
        keep=starts,
        shuffle=False,
        forced=True,
        steps=(0,),
        movement=net.cfg.movement,
        memo=memo,
    )
    for b in prefetch(stream):
        t = _to_device(b, device)
        _, _, value = net(t["planes"], t["globals"], t["rating"] if ratings else None)
        probs.append(value.float().softmax(-1).cpu().numpy())
        zs.append(b.z)
        days.append(b.day)
    net.train(was_training)
    if not probs:
        return {"n": 0}
    return metrics(np.concatenate(probs), np.concatenate(zs), np.concatenate(days))


def train(
    data: Path,
    out: Path,
    cfg: PolicyTrainConfig,
    model_cfg: PolicyConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train (or with ``resume``, continue from ``out/last.pt``); returns the last evaluation."""
    device = torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset(data)
    if not ds.has_actions:
        raise ValueError(f"{data}: not a v1 dataset (build one with `awbw dataset --v1`)")
    if cfg.schedule not in SCHEDULES:
        raise ValueError(f"schedule {cfg.schedule!r}: not one of {SCHEDULES}")
    if cfg.schedule == "wsd" and not 0 < cfg.decay_epochs <= cfg.epochs:
        raise ValueError(f"a decay over {cfg.decay_epochs} epochs: some, and at most the run's {cfg.epochs}")
    keep = rating_filter(cfg.min_rating)
    train_keep = training_games(cfg)
    train_games, n_train = split_size(ds, "train", train_keep, cfg.val_fraction)
    if n_train == 0:
        raise ValueError(f"{data}: no training decisions (min rating {cfg.min_rating}, games {cfg.games_fraction})")
    per_epoch = n_train * cfg.fraction
    planned = cfg.epochs * per_epoch

    epoch = group = step = seen = 0  # group: the current epoch's groups of shards done
    best = math.inf
    if resume and (out / "last.pt").exists():
        net, info = load_policy(out / "last.pt", device)
        progress = info["progress"]
        epoch, step, seen, best = progress["epoch"], progress["step"], progress["seen"], progress["best"]
        group = progress.get("group", 0)
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=device, weights_only=True))
        groups = f" and {group} groups of shards" if group else ""
        log(f"resumed from {out / 'last.pt'} after epoch {epoch}{groups} (step {step})")
    else:
        net = PolicyValueNet(model_cfg or PolicyConfig()).to(device)
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
    net.train()
    net.recompute = cfg.recompute
    forward = torch.compile(net, dynamic=False) if cfg.compile else net
    params = sum(p.numel() for p in net.parameters())
    log(
        f"{params / 1e6:.2f}M parameters on {device}; {n_train:,} training decisions, "
        f"{per_epoch:,.0f} per epoch; {train_games:,} training games of {ds.meta['games']:,}"
    )
    setup = {
        "train": asdict(cfg),
        "dataset": str(data),
        "dataset_games": ds.meta["games"],
        "train_games": train_games,
        "train_decisions": n_train,
    }
    rng = np.random.default_rng(cfg.seed)
    last: dict[str, Any] = {}
    held: dict[str, list] = {"policy": [], "value": []}  # validation's decisions, read once

    def evaluate() -> dict[str, Any]:
        t0 = time.perf_counter()
        pol = evaluate_policy(
            net,
            ds,
            device,
            keep=keep,
            val_fraction=cfg.val_fraction,
            row_fraction=cfg.val_rows,
            memo=held["policy"],
        )
        m: dict[str, Any] = {"policy": pol}
        if not pol["n"]:
            log("  no validation decisions")
            pol["loss"] = math.inf
        else:
            log(
                f"  val policy: loss {pol['loss']:.4f} accuracy {pol['accuracy']:.3f} (uniform {pol['uniform']:.4f}); "
                f"action loss {pol['action_loss']:.3f} accuracy {pol['action_accuracy']:.3f}"
            )
            log(
                "    by phase: "
                + "  ".join(f"{name}: {m['loss']:.3f} / {m['accuracy']:.3f}" for name, m in pol["by_phase"].items())
            )
            if "aux" in pol:
                for line in aux_summary(pol["aux"]):
                    log(f"  val aux {line}")
        for given in (True, False) if net.cfg.ratings else (True,):
            r = evaluate_value(
                net, ds, device, keep=keep, val_fraction=cfg.val_fraction, ratings=given, memo=held["value"]
            )
            what = "" if not net.cfg.ratings else " with ratings" if given else " without"
            if r["n"]:
                log(f"  val value{what}: loss {r['loss']:.4f} accuracy {r['accuracy'] or 0:.3f} ece {r['ece']:.4f}")
            m["value" if given else "value_no_ratings"] = r
        m.update(epoch=round(seen / per_epoch, 3), step=step, seen=seen, time=time.time())
        log(f"  (evaluated in {time.perf_counter() - t0:.0f}s)")
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    saved = time.monotonic()

    def checkpoint() -> None:
        nonlocal saved
        progress = {"epoch": epoch, "group": group, "step": step, "seen": seen, "best": best}
        torch.save(opt.state_dict(), out / "optimizer.tmp")
        (out / "optimizer.tmp").replace(out / "optimizer.pt")
        save_policy(out / "last.pt", net, metrics=last, progress=progress, **setup)
        saved = time.monotonic()

    use_aux = net.cfg.aux
    while epoch < cfg.epochs and seen < planned:
        if cfg.pause_at and epoch >= cfg.pause_at:
            log(f"paused after epoch {epoch} of {cfg.epochs:g}")
            break
        t0, t_log, n_log = time.perf_counter(), time.perf_counter(), 0
        sums = torch.zeros(5, device=device)  # policy loss, value loss, value samples, aux loss, steps
        stream = decision_batches(
            ds,
            cfg.batch_size,
            split="train",
            val_fraction=cfg.val_fraction,
            keep=train_keep,
            seed=cfg.seed * 1000 + epoch,
            shards_at_once=cfg.shards_at_once,
            fraction=cfg.fraction,
            drop_last=True,
            movement=net.cfg.movement,
            aux=use_aux,
            skip_groups=group,
        )
        part = epoch + 1 > cfg.epochs  # the run ends partway through this epoch
        for b in prefetch(stream):
            if b.group != group:  # the groups before b's are done
                group = b.group
                if time.monotonic() - saved >= cfg.checkpoint_minutes * 60:
                    checkpoint()
            if part and seen >= planned:
                break
            if len(b.target) != cfg.batch_size:
                continue  # a step the engine no longer takes was dropped; keep one shape per bucket
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, step, seen, planned, per_epoch)
            t = _to_device(b, device)
            hide = torch.from_numpy(rng.random(len(b.target)) < cfg.rating_dropout).to(device)
            rating = t["rating"].masked_fill(hide[:, None], -1)
            with torch.autocast(device.type, torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
                logp, value, pred = _forward(forward, t, rating, int(rng.integers(8)) if cfg.augment else 0, use_aux)
            policy_loss = -logp.gather(1, t["target"][:, None]).mean()
            valued = torch.from_numpy(rng.random(len(b.target)) < cfg.value_fraction).to(device)
            ce = F.cross_entropy(value.float(), t["z"], reduction="none")
            value_loss = (ce * valued).sum() / valued.sum().clamp_min(1)
            loss = policy_loss + cfg.value_weight * value_loss
            aux_loss = torch.zeros((), device=device)
            if pred is not None:
                chosen = torch.from_numpy(rng.random(len(b.target)) < cfg.aux_fraction).to(device)
                reg, bce = aux_losses(pred, t)
                counted = t["aux_mask"] & chosen[:, None]
                aux_loss = (reg.mean(1) * chosen).sum() / chosen.sum().clamp_min(1) + (
                    bce * counted
                ).sum() / counted.sum().clamp_min(1)
                loss = loss + cfg.aux_weight * aux_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sums += torch.stack(
                [
                    policy_loss.detach() * len(b.target),
                    (ce * valued).sum().detach(),
                    valued.sum().float(),
                    aux_loss.detach(),
                    torch.ones((), device=device),
                ]
            )
            step += 1
            seen += len(b.target)
            n_log += len(b.target)
            if step % cfg.log_every == 0:
                dt = time.perf_counter() - t_log
                p, v, nv, a, ns = sums.tolist()
                aux_part = f" aux {a / max(ns, 1):.4f}" if use_aux else ""
                log(
                    f"epoch {seen / per_epoch:.2f} step {step}: policy {p / n_log:.4f} value {v / max(nv, 1):.4f}"
                    f"{aux_part} lr {opt.param_groups[0]['lr']:.2e} {n_log / dt:.0f} decisions/s"
                )
                sums.zero_()
                t_log, n_log = time.perf_counter(), 0
            if cfg.eval_every and step % cfg.eval_every == 0:
                last = evaluate()
        ended = part and seen >= planned  # else the epoch is whole
        if not ended:
            epoch, group = epoch + 1, 0
        name = f"{cfg.epochs:05.2f}" if ended else f"{epoch:02d}"
        log(f"epoch {name} done in {time.perf_counter() - t0:.0f}s")
        last = evaluate()
        if last["policy"]["loss"] < best or not (out / "best.pt").exists():
            best = last["policy"]["loss"]
            save_policy(out / "best.pt", net, metrics=last, **setup)
        save_policy(out / f"policy-ep{name}.pt", net, metrics=last, **setup)
        checkpoint()
    return last
