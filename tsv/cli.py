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
    return dataclasses.replace(
        DEFAULT,
        data_dir=Path(args.data_dir),
        model_dir_override=Path(args.model_dir) if args.model_dir else None,
    )


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


def _print_analyze(result) -> None:
    if result.status == "skipped":
        print(f"  skip    {result.path.name}")
        return
    if result.status == "failed":
        print(f"  FAILED  {result.path.name}: {result.note}")
        return
    found = ", ".join(f"{n} {label}" for label, n in result.labels.most_common(5)) or "nothing"
    print(
        f"  ok      {result.path.name}: {result.frames} frames over "
        f"{result.n_segments} segments -> {result.n_tracklets} tracklets "
        f"({found}) [{result.fps:.1f} fps]"
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    from tsv.analyze import analyze_all

    cfg = _config(args)
    detect = cfg.detect
    if args.fps:
        detect = dataclasses.replace(detect, detect_fps=args.fps)
    if args.backend:
        detect = dataclasses.replace(detect, force_backend=args.backend)
    if args.model:
        detect = dataclasses.replace(detect, model_file=args.model)
    cfg = dataclasses.replace(cfg, detect=detect)

    if not cfg.detect_model_path.is_file():
        print(f"no detection model at {cfg.detect_model_path}")
        print("fetch one with:  python tools/fetch_models.py")
        return 1

    conn = db.open_db(cfg.db_path)
    print(f"analyzing {cfg.db_path}")
    summary = analyze_all(conn, cfg, force=args.force, on_result=_print_analyze)
    print(f"\nbackend: {summary.backend}")

    if not summary.analyzed:
        print("nothing to analyze. Run `ingest` first, or pass --force.")
        return 0

    found = ", ".join(f"{n} {label}" for label, n in summary.labels.most_common(8))
    print(f"{summary.total_tracklets} tracklets from {summary.total_frames} frames")
    print(f"found: {found}" if found else "found: nothing")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Time one forward pass on every backend that will load the model."""
    import numpy as np

    from tsv.models.backend import DEFAULT_PREFERENCE, benchmark, load_model

    cfg = _config(args)
    if not cfg.detect_model_path.is_file():
        print(f"no detection model at {cfg.detect_model_path}")
        return 1

    size = cfg.detect.input_size
    dummy = np.zeros((1, 3, size, size), dtype=np.float32)
    for runtime, device in DEFAULT_PREFERENCE:
        try:
            backend = load_model(cfg.detect_model_path, force=f"{runtime}:{device}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {runtime}:{device:<5} unavailable ({type(exc).__name__})")
            continue
        seconds = benchmark(backend, {backend.input_names[0]: dummy}, runs=args.runs)
        print(f"  {runtime}:{device:<5} {seconds * 1000:7.1f} ms   {1 / seconds:5.1f} fps")
    return 0


def cmd_zones(args: argparse.Namespace) -> int:
    """List, add or remove zones, and rebuild the events they imply."""
    import json as _json

    from tsv.events import create_zone, delete_zone, list_zones, recompute_events

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)

    if args.zone_command == "add":
        camera = conn.execute(
            "SELECT id FROM cameras WHERE name = ?", (args.camera,)
        ).fetchone()
        if camera is None:
            print(f"no camera named {args.camera!r}")
            return 1
        try:
            points = [tuple(float(v) for v in pair.split(",")) for pair in args.points]
            zone = create_zone(conn, int(camera["id"]), args.name, args.kind, points)
        except ValueError as exc:
            print(f"bad zone: {exc}")
            return 1
        summary = recompute_events(conn, zone.camera_id)
        print(f"added {zone.kind} {zone.name!r} on {args.camera}")
        print(f"{summary.n_events} events across {summary.n_tracklets} tracklets")
        return 0

    if args.zone_command == "remove":
        print("removed" if delete_zone(conn, args.id) else f"no zone with id {args.id}")
        return 0

    if args.zone_command == "recompute":
        summary = recompute_events(conn)
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(summary.by_kind.items()))
        print(f"{summary.n_zones} zones over {summary.n_tracklets} tracklets "
              f"-> {summary.n_events} events ({kinds or 'none'}) in {summary.elapsed:.2f}s")
        return 0

    zones = list_zones(conn)
    if not zones:
        print("no zones defined. Add one with:")
        print("  python -m tsv zones add --camera ch01 --name 'front door' \\")
        print("      --kind line --points 0.2,0.8 0.8,0.8")
        return 0
    for zone in zones:
        camera = conn.execute(
            "SELECT name FROM cameras WHERE id = ?", (zone.camera_id,)
        ).fetchone()["name"]
        n = conn.execute(
            "SELECT COUNT(*) c FROM events WHERE zone_id = ?", (zone.id,)
        ).fetchone()["c"]
        print(f"  [{zone.id}] {camera:<10} {zone.kind:<7} {zone.name:<20} "
              f"{n:>5} events  {_json.dumps(zone.points)}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)
    clauses, params = [], []
    if args.zone:
        clauses.append("AND z.name = ?")
        params.append(args.zone)
    if args.kind:
        clauses.append("AND e.kind = ?")
        params.append(args.kind)
    if args.label:
        clauses.append("AND e.label = ?")
        params.append(args.label)

    rows = conn.execute(
        f"""SELECT e.ts, e.kind, e.label, e.duration, z.name AS zone
            FROM events e JOIN zones z ON z.id = e.zone_id
            WHERE 1=1 {" ".join(clauses)}
            ORDER BY e.ts LIMIT ?""",
        [*params, args.limit],
    ).fetchall()

    if not rows:
        print("no matching events.")
        return 0
    for r in rows:
        when = _dt.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        extra = f" for {r['duration']:.0f}s" if r["duration"] else ""
        print(f"  {when}  {r['label']:<8} {r['kind']:<10} {r['zone']}{extra}")
    return 0


def cmd_people(args: argparse.Namespace) -> int:
    """Name people, and let matching name the rest."""
    from tsv.identity import (
        assign_identities, delete_identity, enroll_tracklet, list_identities,
    )

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)

    if args.people_command == "name":
        exists = conn.execute(
            "SELECT id FROM tracklets WHERE id = ?", (args.tracklet,)
        ).fetchone()
        if exists is None:
            print(f"no tracklet with id {args.tracklet}")
            return 1
        identity, added = enroll_tracklet(conn, args.tracklet, args.name)
        print(f"tracklet {args.tracklet} is {identity.name}")
        if not added:
            print("  note: no embeddings stored yet, so this teaches the gallery nothing")
        return 0

    if args.people_command == "forget":
        print("forgotten" if delete_identity(conn, args.id) else f"no identity {args.id}")
        return 0

    if args.people_command == "assign":
        summary = assign_identities(
            conn, args.kind, args.threshold, args.margin, reassign=args.reassign
        )
        named = ", ".join(f"{n} x {name}" for name, n in sorted(summary.by_name.items()))
        print(f"considered {summary.n_considered} tracklets")
        print(f"  named        {summary.n_assigned}" + (f"  ({named})" if named else ""))
        print(f"  too close    {summary.n_ambiguous}")
        print(f"  no match     {summary.n_below_threshold}")
        return 0

    people = list_identities(conn)
    if not people:
        print("nobody enrolled yet. Name a tracklet with:")
        print("  python -m tsv people name --tracklet 42 --name 'Rafi'")
        return 0
    for person in people:
        print(f"  [{person['id']}] {person['name']:<20} "
              f"{person['n_examples']:>3} examples  {person['n_sightings']:>5} sightings")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt

    from tsv.models.clip import build_clip
    from tsv.search import SearchFilters, rebuild_text_index, search

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)

    if args.reindex:
        print(f"indexed {rebuild_text_index(conn)} segments")

    vector = None
    if args.query and not args.no_semantic:
        clip = build_clip(
            cfg.model_dir, cfg.clip.image_file, cfg.clip.text_file,
            crop_mode=cfg.clip.crop_mode, force_backend=cfg.clip.force_backend,
        )
        if clip is None:
            print("(no CLIP models; falling back to word matching alone)")
        else:
            vector = clip.embed_text(args.query)

    filters = SearchFilters(
        day=args.day, identity=args.who, zone=args.zone,
        label=args.label, event_kind=args.event,
    )
    hits = search(conn, text=args.query or "", query_vector=vector,
                  filters=filters, limit=args.limit,
                  min_similarity=args.min_similarity)

    if not hits:
        print("nothing matched.")
        return 0

    for hit in hits:
        when = _dt.fromtimestamp(hit.ts_start).strftime("%Y-%m-%d %H:%M:%S")
        labels = ""
        if hit.labels:
            import json as _json
            labels = " ".join(f"{n}x{k}" for k, n in _json.loads(hit.labels).items())
        sim = f"{hit.semantic_score:+.3f}" if hit.semantic_score is not None else "  -   "
        print(f"  {when}  sim={sim}  {labels:<22} [{'+'.join(hit.sources)}]")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from datetime import datetime as _dt

    from tsv.models.clip import build_clip
    from tsv.query import ask

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)

    embed = None
    if not args.no_semantic:
        clip = build_clip(
            cfg.model_dir, cfg.clip.image_file, cfg.clip.text_file,
            crop_mode=cfg.clip.crop_mode, force_backend=cfg.clip.force_backend,
        )
        if clip is not None:
            embed = clip.embed_text

    result = ask(conn, args.question, embed_text=embed, limit=args.limit)
    plan = result.plan

    understood = ", ".join(f"{kind}={value}" for kind, value in plan.matched)
    print(f"  understood: {understood or 'nothing specific'}"
          f"{'  + ' + repr(plan.semantic_text) if plan.semantic_text else ''}")
    print()

    if result.answer is not None:
        print(f"  {result.answer.headline}")
        for row in result.answer.rows:
            when = _dt.fromtimestamp(row.ts).strftime("%a %d %b %H:%M:%S")
            who = row.who or row.label
            where = f" at {row.zone}" if row.zone else ""
            what = f" ({row.kind.replace('_', ' ')})" if row.kind else ""
            extra = f" for {row.duration:.0f}s" if row.duration else ""
            print(f"    {when}  {who}{what}{where}{extra}")
        if result.caveat:
            print(f"\n  note: {result.caveat}")

    if result.hits and (result.answer is None or not result.answer.found):
        print(f"  closest matches ({len(result.hits)}):")
        for hit in result.hits[: args.limit]:
            when = _dt.fromtimestamp(hit.ts_start).strftime("%a %d %b %H:%M:%S")
            sim = f"{hit.semantic_score:+.3f}" if hit.semantic_score is not None else "  -   "
            print(f"    {when}  sim={sim}  [{'+'.join(hit.sources)}]")

    if result.mode == "empty":
        print("  nothing matched.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _config(args)
    conn = db.open_db(cfg.db_path)
    row = conn.execute(
        """SELECT COUNT(*) n, COALESCE(SUM(duration),0) d, COALESCE(SUM(active_seconds),0) a
           FROM videos"""
    ).fetchone()
    n_segments = conn.execute("SELECT COUNT(*) n FROM segments").fetchone()["n"]
    n_analyzed = conn.execute(
        "SELECT COUNT(*) n FROM segments WHERE analyzed_at IS NOT NULL"
    ).fetchone()["n"]
    print(f"videos    {row['n']}")
    print(f"footage   {_hms(row['d'])}")
    print(f"active    {_hms(row['a'])} in {n_segments} segments")
    if row["d"]:
        print(f"reduction {1 - row['a'] / row['d']:.1%}")
    print(f"analyzed  {n_analyzed}/{n_segments} segments")

    objects = conn.execute(
        "SELECT label, COUNT(*) n FROM tracklets GROUP BY label ORDER BY n DESC LIMIT 10"
    ).fetchall()
    if objects:
        print("objects   " + ", ".join(f"{r['n']} {r['label']}" for r in objects))

    people = conn.execute(
        """SELECT i.name, COUNT(t.id) n FROM identities i
           LEFT JOIN tracklets t ON t.identity_id = i.id
           GROUP BY i.id ORDER BY n DESC"""
    ).fetchall()
    if people:
        print("people    " + ", ".join(f"{r['name']} ({r['n']})" for r in people))

    n_events = conn.execute("SELECT COUNT(*) n FROM events").fetchone()["n"]
    n_zones = conn.execute("SELECT COUNT(*) n FROM zones").fetchone()["n"]
    if n_zones:
        print(f"zones     {n_zones} defined, {n_events} events")
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
    parser.add_argument("--model-dir", default=None,
                        help="where the ONNX models live (default: <data-dir>/models)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a video file or folder")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--force", action="store_true", help="re-ingest unchanged files")
    p_ingest.set_defaults(func=cmd_ingest)

    p_analyze = sub.add_parser("analyze", help="detect and track objects in indexed segments")
    p_analyze.add_argument("--force", action="store_true", help="re-analyze segments already done")
    p_analyze.add_argument("--fps", type=float, help="frames per second to sample")
    p_analyze.add_argument("--backend", help='pin a backend, e.g. "onnxruntime:CPU"')
    p_analyze.add_argument("--model", help="model filename inside the data model dir")
    p_analyze.set_defaults(func=cmd_analyze)

    p_bench = sub.add_parser("bench", help="time the detector on each available backend")
    p_bench.add_argument("--runs", type=int, default=20)
    p_bench.set_defaults(func=cmd_bench)

    p_zones = sub.add_parser("zones", help="manage zones and rebuild their events")
    p_zones.set_defaults(func=cmd_zones, zone_command="list")
    zone_sub = p_zones.add_subparsers(dest="zone_command")

    z_add = zone_sub.add_parser("add", help="add a zone")
    z_add.add_argument("--camera", required=True)
    z_add.add_argument("--name", required=True)
    z_add.add_argument("--kind", choices=["region", "line"], required=True)
    z_add.add_argument("--points", nargs="+", required=True,
                       metavar="X,Y", help="normalised 0..1, e.g. 0.2,0.8 0.8,0.8")
    z_add.set_defaults(func=cmd_zones)

    z_rm = zone_sub.add_parser("remove", help="remove a zone by id")
    z_rm.add_argument("id", type=int)
    z_rm.set_defaults(func=cmd_zones)

    z_re = zone_sub.add_parser("recompute", help="rebuild every event from stored tracklets")
    z_re.set_defaults(func=cmd_zones)

    p_events = sub.add_parser("events", help="list zone events")
    p_events.add_argument("--zone")
    p_events.add_argument("--kind")
    p_events.add_argument("--label")
    p_events.add_argument("--limit", type=int, default=50)
    p_events.set_defaults(func=cmd_events)

    p_people = sub.add_parser("people", help="name people and assign identities")
    p_people.set_defaults(func=cmd_people, people_command="list")
    people_sub = p_people.add_subparsers(dest="people_command")

    pp_name = people_sub.add_parser("name", help="name a tracklet, teaching the gallery")
    pp_name.add_argument("--tracklet", type=int, required=True)
    pp_name.add_argument("--name", required=True)
    pp_name.set_defaults(func=cmd_people)

    pp_forget = people_sub.add_parser("forget", help="remove an identity by id")
    pp_forget.add_argument("id", type=int)
    pp_forget.set_defaults(func=cmd_people)

    pp_assign = people_sub.add_parser("assign", help="name tracklets that match the gallery")
    pp_assign.add_argument("--kind", choices=["face", "body"], default="face")
    pp_assign.add_argument("--threshold", type=float)
    pp_assign.add_argument("--margin", type=float)
    pp_assign.add_argument("--reassign", action="store_true",
                           help="revisit automatic labels (manual ones are never touched)")
    pp_assign.set_defaults(func=cmd_people)

    p_search = sub.add_parser("search", help="find moments by text and filters")
    p_search.add_argument("query", nargs="?", default="")
    p_search.add_argument("--day")
    p_search.add_argument("--who", help="only segments containing this person")
    p_search.add_argument("--zone", help="only segments touching this zone")
    p_search.add_argument("--label", help="only segments containing this object class")
    p_search.add_argument("--event", help="only segments with this event kind")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--reindex", action="store_true",
                          help="rebuild the word index first")
    p_search.add_argument("--min-similarity", type=float, default=None,
                          help="discard semantic matches below this cosine score")
    p_search.add_argument("--no-semantic", action="store_true",
                          help="word matching only, no CLIP")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="ask a question in plain language")
    p_ask.add_argument("question")
    p_ask.add_argument("--limit", type=int, default=20)
    p_ask.add_argument("--no-semantic", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_stats = sub.add_parser("stats", help="what is in the index")
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="run the timeline UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))
