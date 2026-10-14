"""``awbw`` command line: build indexes, fetch replays and maps, inspect, summarize."""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .data.build import build_v1, game_ratings
from .data.shards import ShardWriter
from .league import fetch_rotation, load_rotation
from .maps import fetch_map
from .net import Http, HttpError, cache_dir
from .replay import load_replay
from .replay.combat_check import check_file
from .replay.extract import extract_v0
from .replay.move_check import MoveReport
from .replay.move_check import check_replay as check_moves
from .replay.sources import (
    AwbwSession,
    IndexEntry,
    RatingHistory,
    ReplayUnavailable,
    fetch_replay,
    next_slice_before,
    read_index,
    replay_path,
    search_mooo,
    write_index,
)
from .replay.stats import MIN_DAYS, summarize, v1_rejection
from .replay.step_check import StepReport
from .replay.step_check import check_file as check_turns


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="awbw")
    p.add_argument("--rate", type=float, default=1.0, help="seconds between HTTP requests (default 1.0)")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("index", help="search awbw.mooo.com and save the results as JSONL")
    q.add_argument("query")
    q.add_argument("--max-pages", type=int)
    q.add_argument("--out", type=Path)

    f = sub.add_parser("fetch", help="download replays into the cache")
    f.add_argument("game_ids", nargs="*", type=int)
    f.add_argument("--index", type=Path, help="JSONL index from `awbw index`")
    f.add_argument("--limit", type=int)
    f.add_argument("--source", choices=["mooo", "awbw"], default="mooo")

    cr = sub.add_parser("crawl", help="index and fetch a mooo search back in time, slice by slice, up to a size cap")
    cr.add_argument("query", help='mooo search without a date, e.g. "2p global league std"')
    cr.add_argument("--before", required=True, help="start from games strictly before this date (YYYY-MM-DD)")
    cr.add_argument("--max-gb", type=float, required=True, help="stop once the replay cache holds this many GB")

    m = sub.add_parser("fetch-map", help="download maps into the cache")
    m.add_argument("map_ids", nargs="*", type=int)
    m.add_argument("--index", type=Path, help="also fetch every map in this JSONL index")

    lg = sub.add_parser("league", help="snapshot the Global League map rotation and settings")
    lg.add_argument("--mode", choices=["std", "fog", "hf"], default="std")
    lg.add_argument("--fetch-maps", action="store_true", help="also download terrain for the listed maps")

    i = sub.add_parser("inspect", help="print a summary of one replay")
    i.add_argument("replay", help="path to a replay zip, or a cached game id")

    s = sub.add_parser("stats", help="summarize every replay in a directory")
    s.add_argument("dir", nargs="?", type=Path)
    s.add_argument("--workers", type=int)

    c = sub.add_parser("check-combat", help="check every logged attack against the engine's damage formula")
    c.add_argument("dir", nargs="?", type=Path)
    c.add_argument("--workers", type=int)
    c.add_argument("--show", type=int, default=10, help="print this many mismatches (default 10)")

    mv = sub.add_parser("check-moves", help="load every snapshot into the engine and check logged moves")
    mv.add_argument("dir", nargs="?", type=Path)
    mv.add_argument("--workers", type=int)
    mv.add_argument("--show", type=int, default=10, help="print this many failures (default 10)")

    ds = sub.add_parser(
        "dataset", help="build a dataset from replays: v0 (turn-start states + results) or v1 (every action)"
    )
    ds.add_argument("dir", nargs="?", type=Path, help="replay directory (default: the cache)")
    ds.add_argument("--v1", action="store_true", help="a sample per action taken, labelled for the policy")
    ds.add_argument("--out", type=Path, help="dataset directory (default: cache/datasets/v0 or v1)")
    ds.add_argument("--workers", type=int)
    ds.add_argument(
        "--games-per-shard", type=int, help="v0: games per shard (default 1000); v1: replays per shard (100)"
    )
    ds.add_argument("--overwrite", action="store_true", help="replace a dataset already in --out")

    tc = sub.add_parser("check-turns", help="replay every turn with the engine and compare it with the next snapshot")
    tc.add_argument("dir", nargs="?", type=Path)
    tc.add_argument("--workers", type=int)
    tc.add_argument("--show", type=int, default=3, help="examples per mismatch kind (default 3)")
    tc.add_argument("--skips", action="store_true", help="also list why turns were skipped")

    tr = sub.add_parser("train-critic", help="train the critic on a v0 dataset")
    tr.add_argument("--data", type=Path, help="dataset directory (default: cache/datasets/v0)")
    tr.add_argument("--out", type=Path, required=True, help="run directory for checkpoints and metrics.jsonl")
    tr.add_argument("--epochs", type=int)
    tr.add_argument("--batch-size", type=int)
    tr.add_argument("--lr", type=float)
    tr.add_argument("--weight-decay", type=float)
    tr.add_argument("--min-rating", type=int, help="keep games whose players are both rated at least this")
    tr.add_argument("--rating-dropout", type=float, help="chance of hiding a sample's ratings (default 0.25)")
    tr.add_argument("--eval-every", type=int, help="steps between evaluations (default: once per epoch)")
    tr.add_argument("--seed", type=int)
    tr.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    tr.add_argument("--compile", action="store_true", default=None, help="torch.compile the network")
    tr.add_argument("--channels", type=int)
    tr.add_argument("--blocks", type=int)
    tr.add_argument("--device", default="auto", help="cuda, mps, cpu or auto (default)")
    tr.add_argument("--resume", action="store_true", help="continue the run in --out from its last epoch")

    ec = sub.add_parser("eval-critic", help="validation metrics and calibration of a saved critic")
    ec.add_argument("checkpoint", type=Path)
    ec.add_argument("--data", type=Path, help="dataset directory (default: cache/datasets/v0)")
    ec.add_argument("--split", choices=["val", "train", "all"], default="val")
    ec.add_argument("--min-rating", type=int, default=0)
    ec.add_argument("--device", default="auto")

    tp = sub.add_parser("train-policy", help="pre-train the policy-value network on a v1 dataset")
    tp.add_argument("--data", type=Path, help="dataset directory (default: cache/datasets/v1)")
    tp.add_argument("--out", type=Path, required=True, help="run directory for checkpoints and metrics.jsonl")
    tp.add_argument("--epochs", type=int)
    tp.add_argument("--fraction", type=float, help="share of the training decisions per epoch (default 0.25)")
    tp.add_argument("--batch-size", type=int)
    tp.add_argument("--lr", type=float)
    tp.add_argument("--weight-decay", type=float)
    tp.add_argument("--value-weight", type=float, help="weight of the value loss (default 1)")
    tp.add_argument("--value-fraction", type=float, help="share of decisions whose result trains the value (0.05)")
    tp.add_argument("--min-rating", type=int, help="keep games whose players are both rated at least this")
    tp.add_argument("--rating-dropout", type=float, help="chance of hiding a sample's ratings (default 0.25)")
    tp.add_argument("--val-rows", type=float, help="share of validation actions scored (default 0.1)")
    tp.add_argument("--shards-at-once", type=int, help="shards whose decisions are shuffled together (default 8)")
    tp.add_argument("--eval-every", type=int, help="steps between evaluations (default: once per epoch)")
    tp.add_argument("--seed", type=int)
    tp.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    tp.add_argument("--compile", action="store_true", default=None, help="torch.compile the network")
    tp.add_argument("--no-bf16", dest="bf16", action="store_false", default=None, help="train in float32 on CUDA too")
    tp.add_argument("--channels", type=int)
    tp.add_argument("--blocks", type=int)
    tp.add_argument("--device", default="auto", help="cuda, mps, cpu or auto (default)")
    tp.add_argument("--resume", action="store_true", help="continue the run in --out from its last epoch")

    ep = sub.add_parser("eval-policy", help="validation metrics of a saved policy-value network")
    ep.add_argument("checkpoint", type=Path)
    ep.add_argument("--data", type=Path, help="dataset directory (default: cache/datasets/v1)")
    ep.add_argument("--split", choices=["val", "train", "all"], default="val")
    ep.add_argument("--rows", type=float, default=0.1, help="share of the split's actions scored (default 0.1)")
    ep.add_argument("--min-rating", type=int, default=0)
    ep.add_argument("--device", default="auto")

    ar = sub.add_parser("arena", help="play a saved policy against a random mover or another policy")
    ar.add_argument("checkpoint", type=Path)
    ar.add_argument("--opponent", default="random", help="'random' (default) or another policy checkpoint")
    ar.add_argument("--games", type=int, default=32, help="start positions, each played twice with seats swapped")
    ar.add_argument("--max-days", type=int, default=30, help="then more properties wins (default 30)")
    ar.add_argument("--temperature", type=float, default=0.0, help="0 (default): always the likeliest step")
    ar.add_argument("--ratings", type=int, nargs=2, metavar=("MOVER", "OPPONENT"), help="ratings the nets are told")
    ar.add_argument("--data", type=Path, help="v1 dataset whose validation games give the starts (default: v1)")
    ar.add_argument("--device", default="auto")
    ar.add_argument("--seed", type=int, default=0)

    va = sub.add_parser("value", help="a saved critic's view of a replay, turn by turn")
    va.add_argument("checkpoint", type=Path)
    va.add_argument("replay", help="path to a replay zip, or a cached game id")
    va.add_argument("--ratings", type=int, nargs=2, metavar=("P1", "P2"), help="the players' ratings (default unknown)")
    va.add_argument("--device", default="auto")

    lv = sub.add_parser("live", help="a saved critic's view of a game in progress on awbw.amarriner.com")
    lv.add_argument("checkpoint", type=Path)
    lv.add_argument("game", help="game id or game page URL")
    lv.add_argument("--history", action="store_true", help="every turn so far, not just the current one")
    lv.add_argument(
        "--no-beliefs",
        action="store_true",
        help="fetch only the current turn; HP is known only as the page shows it (faster on first use)",
    )
    lv.add_argument("--ratings", type=int, nargs=2, metavar=("P1", "P2"), help="the players' ratings (default unknown)")
    lv.add_argument("--device", default="auto")

    args = p.parse_args(argv)
    http = Http(min_interval=args.rate)
    commands = {
        "index": _index,
        "fetch": _fetch,
        "crawl": _crawl,
        "fetch-map": _fetch_map,
        "league": _league,
        "inspect": _inspect,
        "stats": _stats,
        "check-combat": _check_combat,
        "check-moves": _check_moves,
        "check-turns": _check_turns,
        "dataset": _dataset,
        "train-critic": _train_critic,
        "eval-critic": _eval_critic,
        "train-policy": _train_policy,
        "eval-policy": _eval_policy,
        "arena": _arena,
        "value": _value,
        "live": _live,
    }
    return commands[args.cmd](args, http)


def _index(args, http: Http) -> int:
    entries = search_mooo(args.query, http, max_pages=args.max_pages)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.query).strip("_").lower() or "all"
    out = args.out or cache_dir() / "index" / f"{slug}.jsonl"
    write_index(out, entries)
    print(f"{len(entries)} games -> {out}")
    return 0


def _fetch(args, http: Http) -> int:
    ids = list(args.game_ids)
    if args.index:
        ids += [e.game_id for e in read_index(args.index)]
    ids = list(dict.fromkeys(ids))[: args.limit]
    session = None
    if args.source == "awbw":
        session = AwbwSession(Http(min_interval=args.rate))
        session.login()
    fetched = cached = failed = 0
    for n, game_id in enumerate(ids, 1):
        if replay_path(game_id).exists():
            cached += 1
            continue
        try:
            fetch_replay(game_id, http, source=args.source, session=session)
            fetched += 1
        except ReplayUnavailable as e:
            failed += 1
            print(e, file=sys.stderr)
        if n % 50 == 0:
            print(f"[{n}/{len(ids)}] fetched={fetched} cached={cached} failed={failed}", file=sys.stderr)
    print(f"fetched={fetched} cached={cached} failed={failed}")
    return 0 if not failed else 1


def _crawl(args, http: Http) -> int:
    """Walk ``query`` back through date slices, fetching each slice's maps and replays.

    Each slice's index is saved as it is searched, so a rerun resumes without
    searching again, and cached replays are never fetched twice.
    """
    replays = cache_dir() / "replays"
    cap = int(args.max_gb * 1e9)
    used = sum(p.stat().st_size for p in replays.glob("*.zip"))
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.query).strip("_").lower()
    before: str | None = args.before
    fetched = failed = 0
    while before is not None and used < cap:
        out = cache_dir() / "index" / f"{slug}_before_{before.replace('-', '')}.jsonl"
        if out.exists():
            entries = read_index(out)
        else:
            entries = search_mooo(f"{args.query} before {before}", http)
            write_index(out, entries)
        dates = sorted(e.date for e in entries)
        print(f"slice before {before}: {len(entries)} games {dates[:1]}..{dates[-1:]} -> {out}", file=sys.stderr)
        for map_id in dict.fromkeys(e.map_id for e in entries):
            try:
                fetch_map(map_id, http)
            except (HttpError, ValueError) as e:
                print(f"map {map_id}: {e}", file=sys.stderr)
        for e in entries:
            if used >= cap:
                break
            if replay_path(e.game_id).exists():
                continue
            try:
                used += fetch_replay(e.game_id, http).stat().st_size
                fetched += 1
            except ReplayUnavailable as err:
                failed += 1
                print(err, file=sys.stderr)
            if (fetched + failed) % 100 == 0:
                print(f"fetched={fetched} failed={failed} cache={used / 1e9:.2f} GB", file=sys.stderr)
        before = next_slice_before(entries, before)
    print(f"fetched={fetched} failed={failed} cache={used / 1e9:.2f} GB; next slice: {before or 'none'}")
    return 0


def _fetch_map(args, http: Http) -> int:
    ids = list(args.map_ids)
    if args.index:
        ids += [e.map_id for e in read_index(args.index)]
    for map_id in dict.fromkeys(ids):
        print(fetch_map(map_id, http))
    return 0


def _league(args, http: Http) -> int:
    path = fetch_rotation(http)
    rotation = load_rotation(path)
    maps = rotation.mode(args.mode)
    print(f"{len(maps)} {args.mode} maps in the rotation of {rotation.date} ({path})")
    for m in maps:
        tiers = ",".join(t.name for t in m.tiers if t.enabled)
        bans = ", ".join(m.unit_bans) or "none"
        print(f"  {m.id:7d}  {m.name:28s} cap={m.cap_limit:<3d} units={m.unit_limit:<3d} tiers={tiers:12s} bans={bans}")
    if args.fetch_maps:
        for m in maps:
            fetch_map(m.id, http)
        print(f"fetched {len(maps)} maps")
    return 0


def _inspect(args, http: Http) -> int:
    path = Path(args.replay)
    if not path.exists() and args.replay.isdigit():
        path = replay_path(int(args.replay))
    r = load_replay(path)
    g = r.game
    print(f"game {g.id}: {g.name!r} on map {g.map_id}")
    print(
        f"  type={g.type} fog={g.fog} weather={g.weather_type} powers={g.use_powers} "
        f"funds/property={g.funds_per_property} starting={g.starting_funds} capture_win={g.capture_win}"
    )
    by_id = {p.id: p for p in r.turns[0].players}
    for pid in r.player_ids:
        p = by_id[pid]
        print(f"  player {pid}: order={p.order} country={p.countries_id} co={p.co_id} tag_co={p.tags_co_id}")
    print(f"  {len(r.turns)} turns over {r.days} days; orphan action turns: {r.orphan_action_turns}")
    kinds = Counter(a.get("action", "?") for t in r.turns for a in t.actions or [])
    print("  actions:", dict(kinds.most_common()))
    over = r.game_over()
    print(f"  result: {over}" if over else "  result: none in log")
    last = r.turns[-1]
    print(f"  last snapshot: day {last.day}, {len(last.units)} units, {len(last.buildings)} buildings")
    return 0


def _stats(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    paths = sorted(root.glob("*.zip"))
    if not paths:
        print(f"no replays in {root}")
        return 1
    with ProcessPoolExecutor(args.workers) as pool:
        summaries = list(pool.map(summarize, paths, chunksize=16))

    def show(title: str, counter: Counter, top: int = 12) -> None:
        print(f"{title}:")
        for key, n in counter.most_common(top):
            print(f"  {n:7d}  {key}")

    print(f"{len(summaries)} replays in {root}")
    show("errors", Counter(s.error.split(":")[0] for s in summaries if s.error))
    ok = [s for s in summaries if not s.error]
    show("players", Counter(s.players for s in ok))
    show("fog", Counter(s.fog for s in ok))
    show("weather", Counter(s.weather_type for s in ok))
    show("powers", Counter(s.use_powers for s in ok))
    show("funds/property", Counter(s.funds_per_property for s in ok))
    show("starting funds", Counter(s.starting_funds for s in ok))
    show("capture win", Counter(s.capture_win for s in ok))
    show("game type", Counter(s.type for s in ok))
    show("ended by", Counter(s.end_action for s in ok))
    show("days", Counter(min(s.days // 5 * 5, 40) for s in ok), top=20)
    show("map size", Counter(f"{s.width}x{s.height}" if s.width else "not cached" for s in ok), top=20)
    show("action kinds", sum((s.actions for s in ok), Counter()), top=30)
    show("turns cut off at max turn length (no End)", Counter(min(s.turns_without_end, 3) for s in ok))
    rejections = Counter(v1_rejection(s) or "ACCEPTED" for s in summaries)
    show("v1 filter", rejections)
    return 0


def _check_combat(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    paths = sorted(root.glob("*.zip"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(check_file, paths, chunksize=8))
    skipped = Counter(reason for _, reason in results if reason)
    checks = [c for found, _ in results for c in found]
    bad = [c for c in checks if c.ok is False]
    print(f"{len(paths) - sum(skipped.values())} replays checked; skipped: {dict(skipped)}")
    outcome = Counter("ok" if c.ok else "mismatch" if c.ok is False else f"skipped ({c.reason})" for c in checks)
    print(f"{len(checks)} attacks: {dict(outcome)}")
    if bad:
        print("mismatches by (attacker CO, power) vs (defender CO, power):")
        for key, n in Counter(((c.att_co, c.att_power), (c.def_co, c.def_power)) for c in bad).most_common(10):
            print(f"  {n:6d}  {key}")
        for c in bad[: args.show]:
            print(
                f"  game {c.game_id} day {c.day}: {c.attacker} (co {c.att_co}/{c.att_power}, hp {c.att_hp[:3]}) -> "
                f"{c.defender} (co {c.def_co}/{c.def_power}, hp {c.def_hp[:3]}) at {c.dist}: "
                f"logged {c.logged}, engine {c.predicted[:6]} {c.detail}"
            )
    return 0 if not bad else 1


def _check_moves(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    paths = sorted(root.glob("*.zip"))
    total = MoveReport()
    with ProcessPoolExecutor(args.workers) as pool:
        for rep in pool.map(check_moves, paths, chunksize=8):
            total.add(rep)
    print(f"{len(paths)} replays; skipped: {dict(total.skipped)}")
    print(f"{total.snapshots} snapshots loaded, {len(total.snapshot_errors)} rejected")
    print(f"{total.moves} moves checked: {total.moves_ok} ok, {len(total.failures)} failed")
    for line in total.snapshot_errors[: args.show]:
        print(f"  snapshot {line}")
    for line in total.failures[: args.show]:
        print(f"  move {line}")
    return 0 if not total.failures and not total.snapshot_errors else 1


def _check_turns(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    paths = sorted(root.glob("*.zip"))
    total = StepReport()
    with ProcessPoolExecutor(args.workers) as pool:
        for rep in pool.map(check_turns, paths, chunksize=4):
            total.add(rep)
    checked = total.turns - sum(total.skipped.values())
    bad = sum(total.diverged.values())
    print(f"{total.replays} replays, {total.turns} turns; replays skipped: {dict(total.replays_skipped)}")
    print(f"{checked} turns checked: {total.ok} match ({total.actions_ok} actions), {bad} differ")
    print(f"{sum(total.skipped.values())} turns skipped")
    for what, (hit, n) in sorted(total.aims.items()):
        print(f"  missiles aimed where AWBW aimed them: {what} {hit} of {n}")
    if args.skips:
        for reason, n in total.skipped.most_common():
            print(f"  {n:7d}  {reason}")
    if bad:
        print("differences:")
    for kind, n in total.diverged.most_common():
        print(f"  {n:7d}  {kind}")
        for line in total.examples.get(kind, [])[: args.show]:
            print(f"             {line}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())


def _cached_index() -> tuple[dict[int, IndexEntry], RatingHistory]:
    """Every cached index row of a 2-player game by game id (mooo lists players in
    turn order), and the players' rating histories."""
    entries = [e for path in sorted((cache_dir() / "index").glob("*.jsonl")) for e in read_index(path)]
    return {e.game_id: e for e in entries if len(e.players) == 2}, RatingHistory(entries)


def _dataset(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    out = args.out or cache_dir() / "datasets" / ("v1" if args.v1 else "v0")
    paths = sorted(root.glob("*.zip"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.overwrite:
        for old in [*out.glob("shard-*.npz"), out / "maps.npz", out / "meta.json"]:
            old.unlink(missing_ok=True)
    index, history = _cached_index()
    if args.v1:
        return _dataset_v1(args, paths, out, index, history)
    writer = ShardWriter(out, games_per_shard=args.games_per_shard or 1000)
    rejected: Counter = Counter()
    ratings_seen: Counter = Counter()
    ends: Counter = Counter()
    unstepped = hp_unknown = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for n, g in enumerate(pool.map(extract_v0, paths, chunksize=16), 1):
            if isinstance(g, str):
                rejected[g] += 1
            else:
                unstepped += g.unstepped
                hp_unknown += g.hp_unknown
                ratings, how = game_ratings(g, index, history)
                ratings_seen[how] += 1
                ends[g.end_action] += 1
                writer.add(g, ratings)
            if n % 2000 == 0:
                print(f"[{n}/{len(paths)}] kept {n - rejected.total()}", file=sys.stderr)
    meta = writer.close(
        kind="v0: the state at the start of every turn, labelled with the result; HP beliefs from stepping every turn",
        unstepped_turns=unstepped,
        hp_beliefs_reset=hp_unknown,
        built=datetime.date.today().isoformat(),
        replays=len(paths),
        min_days=MIN_DAYS,
        rejected=dict(rejected.most_common()),
        ratings=dict(ratings_seen),
        ended_by=dict(ends.most_common()),
    )
    size = sum(f.stat().st_size for f in out.glob("*.npz"))
    shards = len(meta["shards"])
    print(f"{meta['games']} games, {meta['samples']} samples in {shards} shards ({size / 1e6:.0f} MB) -> {out}")
    print(f"rejected: {meta['rejected']}")
    print(f"ratings: {meta['ratings']}; ended by: {meta['ended_by']}")
    print(f"turns the engine couldn't step: {unstepped}; HP beliefs restarted after them: {hp_unknown}")
    return 0


def _dataset_v1(args, paths: list[Path], out: Path, index, history) -> int:
    meta, st = build_v1(
        paths,
        out,
        index,
        history,
        workers=args.workers,
        replays_per_shard=args.games_per_shard or 100,
        kind="v1: the state before every action taken, labelled with its micro-steps and the result",
        built=datetime.date.today().isoformat(),
        replays=len(paths),
        min_days=MIN_DAYS,
    )
    size = sum(f.stat().st_size for f in out.glob("*.npz"))
    print(f"{st.games} games, {st.samples:,} samples in {len(st.shards)} shards ({size / 1e6:.0f} MB) -> {out}")
    print(f"labelled: {st.labelled:,} actions, {st.steps:,} micro-steps ({st.forced:,} forced)")
    print(f"rejected: {meta['rejected']}")
    print(f"ratings: {meta['ratings']}; ended by: {meta['ended_by']}")
    print(f"turns the engine couldn't step: {st.unstepped}; HP beliefs restarted after them: {st.hp_unknown}")
    return 0


# The critic commands import torch only when they run, so the others start quickly.


def _train_critic(args, http: Http) -> int:
    from .nn.critic import pick_device
    from .nn.model import CriticConfig
    from .nn.train_critic import TrainConfig, train

    def given(cls):
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__ if getattr(args, k, None) is not None})

    def log(line: str) -> None:
        print(line, flush=True)

    data = args.data or cache_dir() / "datasets" / "v0"
    device = pick_device(args.device)
    train(data, args.out, given(TrainConfig), given(CriticConfig), device=device, resume=args.resume, log=log)
    return 0


def _eval_critic(args, http: Http) -> int:
    from .data.shards import Dataset
    from .nn.critic import load_critic, pick_device
    from .nn.train_critic import metrics, predict, rating_filter

    device = pick_device(args.device)
    ds = Dataset(args.data or cache_dir() / "datasets" / "v0")
    net, info = load_critic(args.checkpoint, device)
    val_fraction = info.get("train", {}).get("val_fraction", 0.05)
    keep = rating_filter(args.min_rating)
    prob, z, day = predict(net, ds, device, split=args.split, keep=keep, val_fraction=val_fraction)
    m = metrics(prob, z, day)
    print(f"{args.checkpoint}: {m['n']:,} {args.split} samples")
    print(f"  loss {m['loss']:.4f}  accuracy {m['accuracy'] or 0:.3f}  brier {m['brier']:.4f}  ece {m['ece']:.4f}")
    if net.cfg.ratings:
        prob, z, day = predict(net, ds, device, split=args.split, keep=keep, val_fraction=val_fraction, ratings=False)
        b = metrics(prob, z, day)
        print(f"  without ratings: loss {b['loss']:.4f}  accuracy {b['accuracy'] or 0:.3f}  brier {b['brier']:.4f}")
    print("  by day:")
    for d in m["by_day"]:
        print(f"    {d['days']:>6s}  {d['n']:8,d}  loss {d['loss']:.4f}  accuracy {d['accuracy'] or 0:.3f}")
    print("  calibration (predicted win chance -> observed):")
    for pred, obs, n in m["calibration"]:
        print(f"    {pred:5.1%} -> {obs:5.1%}  ({n:,})")
    return 0


def _train_policy(args, http: Http) -> int:
    from .nn.critic import pick_device
    from .nn.model import PolicyConfig
    from .nn.train_policy import PolicyTrainConfig, train

    def given(cls):
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__ if getattr(args, k, None) is not None})

    def log(line: str) -> None:
        print(line, flush=True)

    data = args.data or cache_dir() / "datasets" / "v1"
    device = pick_device(args.device)
    train(data, args.out, given(PolicyTrainConfig), given(PolicyConfig), device=device, resume=args.resume, log=log)
    return 0


def _eval_policy(args, http: Http) -> int:
    from .data.shards import Dataset
    from .nn.critic import pick_device
    from .nn.policy import load_policy
    from .nn.train_critic import rating_filter
    from .nn.train_policy import evaluate_policy, evaluate_value

    device = pick_device(args.device)
    ds = Dataset(args.data or cache_dir() / "datasets" / "v1")
    net, info = load_policy(args.checkpoint, device)
    val_fraction = info.get("train", {}).get("val_fraction", 0.05)
    keep = rating_filter(args.min_rating)
    pol = evaluate_policy(
        net, ds, device, keep=keep, val_fraction=val_fraction, row_fraction=args.rows, split=args.split
    )
    print(f"{args.checkpoint}: {pol['n']:,} {args.split} decisions of {pol.get('actions', 0):,} actions")
    print(f"  policy loss {pol['loss']:.4f} (uniform {pol['uniform']:.4f})  accuracy {pol['accuracy']:.3f}")
    print(f"  whole actions: loss {pol['action_loss']:.4f}  accuracy {pol['action_accuracy']:.3f}")
    print("  by phase:")
    for name, m in pol["by_phase"].items():
        print(
            f"    {name:>8s}  {m['n']:9,d}  loss {m['loss']:.4f} (uniform {m['uniform']:.4f})  "
            f"accuracy {m['accuracy']:.3f}"
        )
    for given in (True, False) if net.cfg.ratings else (True,):
        v = evaluate_value(net, ds, device, keep=keep, val_fraction=val_fraction, split=args.split, ratings=given)
        what = "with ratings" if given else "without ratings"
        print(
            f"  value at {v['n']:,} turn starts, {what}: loss {v['loss']:.4f}  accuracy {v['accuracy'] or 0:.3f}  "
            f"brier {v['brier']:.4f}  ece {v['ece']:.4f}"
        )
    return 0


def _arena(args, http: Http) -> int:
    import numpy as np

    from . import _core
    from .agents import DRAW, PolicyAgent, RandomAgent, play
    from .data.loader import in_validation
    from .data.shards import Dataset
    from .nn.policy import Policy

    ds = Dataset(args.data or cache_dir() / "datasets" / "v1")
    starts: list = []
    for i in range(len(ds.shards)):
        if len(starts) >= args.games:
            break
        sh = ds.load(i)
        for g in np.flatnonzero(in_validation(sh.g_id))[: args.games - len(starts)]:
            starts.append(_core.GameState.from_bytes(sh.state[sh.g_start[g]].tobytes()))
    ratings = tuple(args.ratings) if args.ratings else None
    me = PolicyAgent(Policy.load(args.checkpoint, args.device), args.temperature, ratings, seed=args.seed)
    if args.opponent == "random":
        them = RandomAgent(args.seed + 1)
    else:
        them = PolicyAgent(Policy.load(args.opponent, args.device), args.temperature, ratings, seed=args.seed + 1)
    first = play(starts, (me, them), max_days=args.max_days)
    second = play(starts, (them, me), max_days=args.max_days)
    tally = {"won": 0, "lost": 0, "drawn": 0}
    for seat, results in ((0, first), (1, second)):
        won = sum(r.winner == seat for r in results)
        drawn = sum(r.winner == DRAW for r in results)
        lost = len(results) - won - drawn
        tally["won"] += won
        tally["lost"] += lost
        tally["drawn"] += drawn
        ended = sum(r.decided == "game over" for r in results)
        days = np.mean([r.days for r in results])
        print(
            f"as player {seat + 1}: won {won}, lost {lost}, drawn {drawn}; {ended} ended before the day limit, "
            f"{days:.1f} days on average"
        )
    n = sum(tally.values())
    print(f"{args.checkpoint} vs {args.opponent}: {tally['won']}-{tally['lost']}-{tally['drawn']} of {n} games")
    return 0


def _value(args, http: Http) -> int:
    import numpy as np

    from . import engine
    from .nn.critic import Critic
    from .rules import load as load_rules

    path = Path(args.replay)
    if not path.exists() and args.replay.isdigit():
        path = replay_path(int(args.replay))
    r = load_replay(path)
    critic = Critic.load(args.checkpoint, args.device)
    md = engine.map_data(r.game.map_id)
    states, movers, days = [], [], []
    for turn in r.turns:
        s = engine.state_from_turn(r, turn, md)
        states.append(np.frombuffer(s.to_bytes(), np.uint8))
        movers.append(s.active)
        days.append(s.day)
    prob = critic.wdl(np.stack(states), _mover_ratings(args.ratings, movers))
    cos = load_rules().cos
    first, second = sorted(r.turns[0].players, key=lambda p: p.order)
    names = [f"P1 {cos[first.co_id].name}", f"P2 {cos[second.co_id].name}"]
    result = "no result in the log"
    if over := r.game_over():
        order = engine.turn_order(r)
        won = sorted(f"P{order[p] + 1}" for p in over.winners if p in order)
        result = f"{' and '.join(won) or 'nobody'} won ({over.end_action}, day {over.day})"
    print(f"game {r.game.id} on map {r.game.map_id}: {names[0]} vs {names[1]}; {result}")
    _print_values(names, prob, movers, days)
    return 0


def _mover_ratings(ratings: list[int] | None, movers: list[int]) -> list[tuple[int, int]] | None:
    """Ratings by player index to each state's (player to move's, opponent's)."""
    if ratings is None:
        return None
    return [(ratings[m], ratings[1 - m]) for m in movers]


def _print_values(names: list[str], prob, movers: list[int], days: list[int]) -> None:
    """P1's win chance turn by turn, from (loss, draw, win) for each turn's player to move."""
    from .nn.model import LOSS, WIN

    width = max(len(n) for n in names)
    print(f"  day  {'to move':{width}s}  P1 wins")
    for p, mover, day in zip(prob, movers, days, strict=True):
        p1 = float(p[WIN] if mover == 0 else p[LOSS])
        print(f"  {day:3d}  {names[mover]:{width}s}  {p1:6.1%}  {'#' * round(p1 * 40)}")


def _live(args, http: Http) -> int:
    import numpy as np

    from . import live
    from .nn.critic import Critic
    from .nn.model import DRAW, LOSS, WIN
    from .rules import load as load_rules

    try:
        game, turns = live.fetch(live.game_id_from(args.game), http, history=args.history or not args.no_beliefs)
        if args.no_beliefs:
            states, stepped = [live.state(game, t) for t in turns], None
        else:
            states, stepped = live.states(game, turns)
            if not args.history:
                turns, states = turns[-1:], states[-1:]
    except (live.LiveUnavailable, live.Unsupported, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    critic = Critic.load(args.checkpoint, args.device)
    movers = [s.active for s in states]
    prob = critic.wdl(
        np.stack([np.frombuffer(s.to_bytes(), np.uint8) for s in states]), _mover_ratings(args.ratings, movers)
    )
    cos = load_rules().cos
    first = turns[0].turn.players
    co = {p.id: cos[p.co_id].name for p in first}
    names = [f"P{i + 1} {co[pid]} ({game.names[pid]})" for i, pid in enumerate(game.order)]
    g = game.game
    print(f"game {g.id} {g.name!r} on map {g.map_id}: {names[0]} vs {names[1]}" + ("; over" if game.ended else ""))

    def limit(n: int) -> str:
        return str(n) if n else "off"

    cap = limit(g.capture_win if g.capture_win < 1000 else 0)
    print(
        f"  funds {g.funds_per_property}, weather {g.weather_type.lower()}, powers {'on' if g.use_powers else 'off'}, "
        f"capture limit {cap}, day limit {limit(game.day_limit)}, unit limit {limit(game.unit_limit)}"
    )
    if args.history:
        _print_values(names, prob, movers, [s.day for s in states])
    now, p, mover = turns[-1], prob[-1], movers[-1]
    p1, p2 = (p[WIN], p[LOSS]) if mover == 0 else (p[LOSS], p[WIN])
    print(
        f"  day {now.turn.day}, {names[mover]} to move, at the start of the turn: "
        f"P1 wins {p1:.1%}, draw {p[DRAW]:.1%}, P2 wins {p2:.1%}"
    )
    who = {pid: f"P{i + 1}" for i, pid in enumerate(game.order)}
    if game.acted and now.turn.player_id == game.to_move:
        print(f"  note: {game.acted} of {who[game.to_move]}'s units have acted since; the critic values turn starts")
    if stepped is not None:
        why = ", ".join(f"{n} {reason}" for reason, n in stepped.failed.most_common())
        tried = stepped.stepped + sum(stepped.failed.values())
        print(
            f"  HP beliefs from watching the game: {stepped.stepped} of {tried} turns stepped"
            + (f" (not: {why}; their units' HP restarts from the page)" if why else "")
        )
        if now.unknown_hp and now.turn.player_id in {u.players_id for u in now.turn.units if u.id in now.hidden_hp}:
            print(f"  note: {now.unknown_hp} units hide their HP (Sonja); to her, they show the likeliest displays")
    else:
        if now.estimated_hp:
            print(f"  note: {now.estimated_hp} damaged units show whole HP only; their exact HP is estimated")
        if now.unknown_hp:
            print(f"  note: {now.unknown_hp} units hide their HP (Sonja): any display to her opponent, full to her")
    for u in now.put_back:
        owner, mover_id = who[u.players_id], now.turn.player_id
        where = "on its own board at a guessed tile" if u.players_id == mover_id else f"out of {who[mover_id]}'s sight"
        print(f"  note: the site hides {owner}'s {u.name} from spectators; put back {where}")
    for (pid, name), n in now.unplaced.items():
        print(f"  note: {n} {name} of {who[pid]}'s missing from the page and left out")
    return 0
