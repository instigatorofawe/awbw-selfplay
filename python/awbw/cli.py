"""``awbw`` command line: build indexes, fetch replays and maps, inspect, summarize."""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .data.shards import UNKNOWN_RATING, ShardWriter
from .league import fetch_rotation, load_rotation
from .maps import fetch_map
from .net import Http, HttpError, cache_dir
from .replay import load_replay
from .replay.combat_check import check_file
from .replay.extract import DRAW, extract_v0
from .replay.move_check import MoveReport
from .replay.move_check import check_replay as check_moves
from .replay.sources import (
    AwbwSession,
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

    ds = sub.add_parser("dataset", help="build the v0 critic dataset (turn-start states + results) from replays")
    ds.add_argument("dir", nargs="?", type=Path, help="replay directory (default: the cache)")
    ds.add_argument("--out", type=Path, help="dataset directory (default: cache/datasets/v0)")
    ds.add_argument("--workers", type=int)
    ds.add_argument("--games-per-shard", type=int, default=1000)
    ds.add_argument("--overwrite", action="store_true", help="replace a dataset already in --out")

    tc = sub.add_parser("check-turns", help="replay every turn with the engine and compare it with the next snapshot")
    tc.add_argument("dir", nargs="?", type=Path)
    tc.add_argument("--workers", type=int)
    tc.add_argument("--show", type=int, default=3, help="examples per mismatch kind (default 3)")
    tc.add_argument("--skips", action="store_true", help="also list why turns were skipped")

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


def _ratings_by_game() -> dict[int, tuple[tuple[int, int], tuple[bool, bool]]]:
    """Game id -> (ratings, lost flags) from every cached index; mooo lists players in turn order."""
    out = {}
    for path in sorted((cache_dir() / "index").glob("*.jsonl")):
        for e in read_index(path):
            if len(e.players) == 2:
                r0, r1 = (UNKNOWN_RATING if p.rating is None else p.rating for p in e.players)
                out[e.game_id] = ((r0, r1), (e.players[0].lost, e.players[1].lost))
    return out


def _dataset(args, http: Http) -> int:
    root = args.dir or cache_dir() / "replays"
    out = args.out or cache_dir() / "datasets" / "v0"
    paths = sorted(root.glob("*.zip"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.overwrite:
        for old in [*out.glob("shard-*.npz"), out / "maps.npz", out / "meta.json"]:
            old.unlink(missing_ok=True)
    writer = ShardWriter(out, games_per_shard=args.games_per_shard)
    index = _ratings_by_game()
    rejected: Counter = Counter()
    ratings_seen: Counter = Counter()
    ends: Counter = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for n, g in enumerate(pool.map(extract_v0, paths, chunksize=16), 1):
            if isinstance(g, str):
                rejected[g] += 1
            else:
                ratings = (UNKNOWN_RATING, UNKNOWN_RATING)
                if (entry := index.get(g.game_id)) is None:
                    ratings_seen["not in an index"] += 1
                elif g.winner != DRAW and entry[1] != (g.winner != 0, g.winner != 1):
                    ratings_seen["index result disagrees; dropped"] += 1
                else:
                    ratings, _ = entry
                    ratings_seen["known"] += 1
                ends[g.end_action] += 1
                writer.add(g, ratings)
            if n % 2000 == 0:
                print(f"[{n}/{len(paths)}] kept {n - rejected.total()}", file=sys.stderr)
    meta = writer.close(
        kind="v0: the state at the start of every turn, labelled with the result",
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
    return 0
