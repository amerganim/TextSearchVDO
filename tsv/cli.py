"""Command line entry point: `python -m tsv ...`"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from tsv import db
from tsv.config import DEFAULT, Config
from tsv.ingest import IngestResult, ingest_path


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _config(args: argparse.Namespace) -> Config:
    return dataclasses.replace(DEFAULT, data_dir=Path(args.data_dir))


def _print_result(result: IngestResult) -> None:
    if result.status == "skipped":
        print(f"  skip    {result.path.name}")
        return
    if result.status == "failed":
        print(f"  FAILED  {result.path.name}: {result.note}")
        return
    speed = result.duration / result.elapsed if result.elapsed else 0.0
    print(
        f"  ok      {result.path.name}: {_hms(result.duration)} -> "
        f"{result.n_segments} segments, {_hms(result.active_seconds)} active "
        f"({result.compression:.1%} skipped) [{speed:.0f}x realtime]"
    )
    if result.note:
        print(f"          note: {result.note}")


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    conn = db.open_db(cfg.db_path)
    root = Path(args.path)
    if not root.exists():
        print(f"no such path: {root}")
        return 1

    print(f"ingesting {root} -> {cfg.db_path}")
    summary = ingest_path(conn, root, cfg, force=args.force, on_result=_print_result)

    n_failed = sum(1 for r in summary.results if r.status == "failed")
    if not summary.ingested:
        print("\nnothing new ingested.")
        return 1 if n_failed else 0

    duration, active = summary.total_duration, summary.total_active
    print(
        f"\n{len(summary.ingested)} file(s): {_hms(duration)} of footage -> "
        f"{_hms(active)} worth watching across {summary.total_segments} segments."
    )
    print(f"{1 - active / duration:.1%} of the recording needs no review." if duration else "")
    if n_failed:
        print(f"{n_failed} file(s) failed.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _config(args)
    conn = db.open_db(cfg.db_path)
    row = conn.execute(
        """SELECT COUNT(*) n, COALESCE(SUM(duration),0) d, COALESCE(SUM(active_seconds),0) a
           FROM videos"""
    ).fetchone()
    n_segments = conn.execute("SELECT COUNT(*) n FROM segments").fetchone()["n"]
    print(f"videos    {row['n']}")
    print(f"footage   {_hms(row['d'])}")
    print(f"active    {_hms(row['a'])} in {n_segments} segments")
    if row["d"]:
        print(f"reduction {1 - row['a'] / row['d']:.1%}")
    print()
    for cam in conn.execute(
        """SELECT c.name, COUNT(v.id) n, COALESCE(SUM(v.duration),0) d
           FROM cameras c LEFT JOIN videos v ON v.camera_id=c.id
           GROUP BY c.id ORDER BY c.name"""
    ):
        print(f"  {cam['name']:<16} {cam['n']:>4} file(s)  {_hms(cam['d'])}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from tsv.api import create_app

    app = create_app(_config(args))
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsv", description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT.data_dir))
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a video file or folder")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--force", action="store_true", help="re-ingest unchanged files")
    p_ingest.set_defaults(func=cmd_ingest)

    p_stats = sub.add_parser("stats", help="what is in the index")
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="run the timeline UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))
