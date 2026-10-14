"""Pre-train the policy-value network on a v1 dataset: behaviour cloning plus the result.

Each training sample is one micro-step decision of an action a player took
(``loader.decision_batches``; forced steps are left out). The policy loss is
the cross-entropy of the step taken, over the steps legal there. The value
loss is v0's cross-entropy over (loss, draw, win), on a random
``value_fraction`` of the decisions only: a game has about 1,100 decisions
and they all share its result, so valuing every one would show each result
some 40 times as often as the critic sees it in v0 (one sample per turn),
and the critic already overfits v0 in its third epoch.

An epoch is a random ``fraction`` of the training decisions. Each batch gets
one of the 8 flips and transposes of its square bucket; the spatial policy
logits are mapped back before the loss. Ratings are hidden at random as for
the critic, so the policy can imitate players of a given strength or of any.

Validation scores the decisions of a fixed ``val_rows`` share of the
validation games' actions (by phase, against a uniform choice among the
legal steps, and whole actions), and values every validation turn start, as
``eval-critic`` does. An output directory holds ``metrics.jsonl``,
``policy-epNN.pt``, ``best.pt`` (lowest validation policy loss) and
``last.pt`` + ``optimizer.pt`` for ``resume``.
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
from ..data.loader import DecisionBatch, decision_batches, in_validation
from ..data.shards import Dataset, Shard
from .model import PolicyConfig, PolicyValueNet, flat_policy, legal_mask, masked_log_softmax
from .policy import load_policy, save_policy
from .train_critic import _lr, _optimizer, dihedral, metrics, prefetch, rating_filter

PHASE_NAMES = tuple(_core.PHASES)


@dataclass
class PolicyTrainConfig:
    epochs: int = 2
    fraction: float = 0.25  # share of the training decisions in each epoch
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    value_weight: float = 1.0
    value_fraction: float = 0.05  # share of decisions whose result trains the value head
    min_rating: int = 0  # keep games whose players are both rated at least this
    rating_dropout: float = 0.25
    augment: bool = True
    compile: bool = False
    val_fraction: float = 0.05
    val_rows: float = 0.1  # share of the validation games' actions whose decisions are scored
    shards_at_once: int = 8
    eval_every: int = 0  # steps between evaluations; 0 = once per epoch
    log_every: int = 200
    seed: int = 0


def undihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """The inverse of ``dihedral(x, k)``."""
    if k & 4:
        x = x.transpose(-1, -2)
    if k & 2:
        x = x.flip(-2)
    if k & 1:
        x = x.flip(-1)
    return x


def count_decisions(ds: Dataset, split: str, keep: Callable[[Shard], np.ndarray] | None, val_fraction: float) -> int:
    """Decisions ``decision_batches`` gives for ``split`` (forced steps left out)."""
    n = 0
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
    return n


def _to_device(b: DecisionBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planes": torch.from_numpy(b.planes).to(device),
        "globals": torch.from_numpy(b.globals).to(device),
        "nonspatial": torch.from_numpy(b.nonspatial).to(device),
        "target": torch.from_numpy(b.target).to(device),
        "z": torch.from_numpy(b.z.astype(np.int64) + 1).to(device),  # -1/0/+1 -> LOSS/DRAW/WIN
        "rating": torch.from_numpy(b.rating.astype(np.int32)).to(device),
    }


def _forward(net: Any, t: dict[str, torch.Tensor], rating: torch.Tensor | None, k: int = 0):
    """Log-probabilities over the policy indices and value logits, through flip ``k``."""
    spatial, nonspatial, value = net(dihedral(t["planes"], k) if k else t["planes"], t["globals"], rating)
    if k:
        spatial = undihedral(spatial, k)
    legal = legal_mask(t["planes"], t["nonspatial"])
    return masked_log_softmax(flat_policy(spatial, nonspatial), legal), value


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
) -> dict[str, Any]:
    """``policy_metrics`` on a fixed share of ``split``'s actions."""
    was_training = net.training
    net.eval()
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("nll", "correct", "n_legal", "phase", "row")}
    stream = decision_batches(
        ds, batch_size, split=split, val_fraction=val_fraction, keep=keep, shuffle=False, row_fraction=row_fraction
    )
    for b in prefetch(stream):
        t = _to_device(b, device)
        logp, _ = _forward(net, t, t["rating"] if ratings else None)
        cols["nll"].append(-logp.gather(1, t["target"][:, None])[:, 0].cpu().numpy())
        cols["correct"].append((logp.argmax(1) == t["target"]).cpu().numpy())
        cols["n_legal"].append(b.n_legal)
        cols["phase"].append(b.phase)
        cols["row"].append(b.row)
    net.train(was_training)
    if not cols["nll"]:
        return {"n": 0, "by_phase": {}}
    return policy_metrics(*(np.concatenate(cols[k]) for k in ("nll", "correct", "n_legal", "phase", "row")))


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
) -> dict[str, Any]:
    """The critic's metrics (``train_critic.metrics``) over ``split``'s turn starts."""
    was_training = net.training
    net.eval()

    def starts(sh: Shard) -> np.ndarray:
        assert sh.action is not None
        mask = sh.action == 0
        return mask & keep(sh) if keep is not None else mask

    probs, zs, days = [], [], []
    stream = decision_batches(
        ds, batch_size, split=split, val_fraction=val_fraction, keep=starts, shuffle=False, forced=True, steps=(0,)
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
    keep = rating_filter(cfg.min_rating)
    n_train = count_decisions(ds, "train", keep, cfg.val_fraction)
    if n_train == 0:
        raise ValueError(f"{data}: no training decisions (min rating {cfg.min_rating})")
    per_epoch = n_train * cfg.fraction
    planned = cfg.epochs * per_epoch

    epoch = step = seen = 0
    best = math.inf
    if resume and (out / "last.pt").exists():
        net, info = load_policy(out / "last.pt", device)
        progress = info["progress"]
        epoch, step, seen, best = progress["epoch"], progress["step"], progress["seen"], progress["best"]
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
        opt.load_state_dict(torch.load(out / "optimizer.pt", map_location=device, weights_only=True))
        log(f"resumed from {out / 'last.pt'} after epoch {epoch} (step {step})")
    else:
        net = PolicyValueNet(model_cfg or PolicyConfig()).to(device)
        opt = _optimizer(net, cfg)  # type: ignore[arg-type]
    net.train()
    forward = torch.compile(net, dynamic=False) if cfg.compile else net
    params = sum(p.numel() for p in net.parameters())
    log(
        f"{params / 1e6:.2f}M parameters on {device}; {n_train:,} training decisions, "
        f"{per_epoch:,.0f} per epoch; {ds.meta['games']:,} games"
    )
    setup = {"train": asdict(cfg), "dataset": str(data), "dataset_games": ds.meta["games"], "train_decisions": n_train}
    rng = np.random.default_rng(cfg.seed)
    last: dict[str, Any] = {}

    def evaluate() -> dict[str, Any]:
        t0 = time.perf_counter()
        pol = evaluate_policy(net, ds, device, keep=keep, val_fraction=cfg.val_fraction, row_fraction=cfg.val_rows)
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
        for given in (True, False) if net.cfg.ratings else (True,):
            r = evaluate_value(net, ds, device, keep=keep, val_fraction=cfg.val_fraction, ratings=given)
            what = "" if not net.cfg.ratings else " with ratings" if given else " without"
            if r["n"]:
                log(f"  val value{what}: loss {r['loss']:.4f} accuracy {r['accuracy'] or 0:.3f} ece {r['ece']:.4f}")
            m["value" if given else "value_no_ratings"] = r
        m.update(epoch=round(seen / per_epoch, 3), step=step, seen=seen, time=time.time())
        log(f"  (evaluated in {time.perf_counter() - t0:.0f}s)")
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    while epoch < cfg.epochs:
        t0, t_log, n_log = time.perf_counter(), time.perf_counter(), 0
        sums = torch.zeros(3, device=device)  # policy loss, value loss, value samples
        stream = decision_batches(
            ds,
            cfg.batch_size,
            split="train",
            val_fraction=cfg.val_fraction,
            keep=keep,
            seed=cfg.seed * 1000 + epoch,
            shards_at_once=cfg.shards_at_once,
            fraction=cfg.fraction,
            drop_last=True,
        )
        for b in prefetch(stream):
            if len(b.target) != cfg.batch_size:
                continue  # a step the engine no longer takes was dropped; keep one shape per bucket
            for group in opt.param_groups:
                group["lr"] = _lr(cfg, step, seen / planned)  # type: ignore[arg-type]
            t = _to_device(b, device)
            hide = torch.from_numpy(rng.random(len(b.target)) < cfg.rating_dropout).to(device)
            rating = t["rating"].masked_fill(hide[:, None], -1)
            logp, value = _forward(forward, t, rating, int(rng.integers(8)) if cfg.augment else 0)
            policy_loss = -logp.gather(1, t["target"][:, None]).mean()
            valued = torch.from_numpy(rng.random(len(b.target)) < cfg.value_fraction).to(device)
            ce = F.cross_entropy(value.float(), t["z"], reduction="none")
            value_loss = (ce * valued).sum() / valued.sum().clamp_min(1)
            loss = policy_loss + cfg.value_weight * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sums += torch.stack(
                [policy_loss.detach() * len(b.target), (ce * valued).sum().detach(), valued.sum().float()]
            )
            step += 1
            seen += len(b.target)
            n_log += len(b.target)
            if step % cfg.log_every == 0:
                dt = time.perf_counter() - t_log
                p, v, nv = sums.tolist()
                log(
                    f"epoch {seen / per_epoch:.2f} step {step}: policy {p / n_log:.4f} value {v / max(nv, 1):.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {n_log / dt:.0f} decisions/s"
                )
                sums.zero_()
                t_log, n_log = time.perf_counter(), 0
            if cfg.eval_every and step % cfg.eval_every == 0:
                last = evaluate()
        epoch += 1
        log(f"epoch {epoch} done in {time.perf_counter() - t0:.0f}s")
        last = evaluate()
        if last["policy"]["loss"] < best or not (out / "best.pt").exists():
            best = last["policy"]["loss"]
            save_policy(out / "best.pt", net, metrics=last, **setup)
        save_policy(out / f"policy-ep{epoch:02d}.pt", net, metrics=last, **setup)
        progress = {"epoch": epoch, "step": step, "seen": seen, "best": best}
        torch.save(opt.state_dict(), out / "optimizer.pt")
        save_policy(out / "last.pt", net, metrics=last, progress=progress, **setup)
    return last
