"""``awbw`` command line: build indexes, fetch replays and maps, inspect, summarize."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .league import fetch_rotation, load_rotation
from .maps import fetch_map
from .net import Http, cache_dir
from .replay import load_replay
from .replay.combat_check import check_file
from .replay.move_check import MoveReport
from .replay.move_check import check_replay as check_moves
from .replay.sources import (
    AwbwSession,
    ReplayUnavailable,
    fetch_replay,
    read_index,
    replay_path,
    search_mooo,
    write_index,
)
from .replay.stats import summarize, v1_rejection


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

    args = p.parse_args(argv)
    http = Http(min_interval=args.rate)
    commands = {
        "index": _index,
        "fetch": _fetch,
        "fetch-map": _fetch_map,
        "league": _league,
        "inspect": _inspect,
        "stats": _stats,
        "check-combat": _check_combat,
        "check-moves": _check_moves,
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


if __name__ == "__main__":
    sys.exit(main())
