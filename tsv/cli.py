"""Command line entry point: `python -m tsv ...`"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from tsv import db
from tsv.config import DEFAULT, Config
from tsv.ingest import IngestResult, ingest_path
from tsv.setup import DETECTORS


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
    if result.status == "duplicate":
        print(f"  dup     {result.path.name}: {result.note}")
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
        if added:
            print(f"  learned: {', '.join(added)}")
        else:
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


def cmd_caption(args: argparse.Namespace) -> int:
    """Describe what people in the index are doing."""
    from tsv.captioning import caption_tracklets
    from tsv.search import rebuild_text_index

    cfg = _config(args)
    if not cfg.has_caption_model:
        print(f'no captioning model in {cfg.caption_model_dir}')
        print('fetch it with:  .venv-export/Scripts/python tools/fetch_caption_model.py')
        return 1

    conn = db.open_db(cfg.db_path)
    print('describing tracklets (about 6 seconds each on a CPU)')

    def on_progress(done: int, total: int) -> None:
        # A plain line every few, rather than a carriage-return spinner:
        # this output is usually being read in a scrollback or a log.
        if done == total or done % 5 == 0:
            print(f"  {done}/{total} described", flush=True)

    summary = caption_tracklets(conn, cfg, force=args.force, limit=args.limit,
                                on_progress=on_progress)
    print()
    print(f'  described  {summary.captioned}')
    print(f'  too small  {summary.skipped_small}')
    print(f'  failed     {summary.failed}')
    if summary.captioned:
        print(f'  {summary.per_caption:.1f}s each, {summary.elapsed:.0f}s total')
    for text in summary.samples[:5]:
        print(f'    "{text}"')
    if summary.captioned:
        print(f'indexed {rebuild_text_index(conn)} segments for search')
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Read speech out of the indexed videos, or out of one file."""
    import time as _time

    from tsv.audio import (
        has_audio, load_model, pick_device, transcribe_file, transcribe_videos,
    )
    from tsv.search import rebuild_text_index

    cfg = _config(args)
    if not cfg.has_audio_model:
        print("no transcription model. Fetch it with:")
        print("  python tools/fetch_audio_model.py --out data/models")
        return 1

    # --file is the "does this work at all" path: one file, nothing indexed,
    # nothing changed. The index is the wrong place to find that out.
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"no such file: {path}")
            return 1

        device, compute = pick_device(cfg.audio.device)
        print(path.name)
        if not has_audio(path):
            print("  no audio stream in this file, so there is nothing to hear.")
            return 0

        model = load_model(cfg)
        started = _time.time()
        found = transcribe_file(model, path, language=cfg.audio.language)
        elapsed = _time.time() - started
        print(f"  {cfg.audio.model_dir} on {device}/{compute}, {elapsed:.1f}s")

        if not found:
            print("  nothing said, or nothing said clearly enough to keep.")
            print("  silence is discarded deliberately: run over a silent track,")
            print("  Whisper invents fluent text rather than returning nothing.")
            return 0

        for utterance in found:
            print(f"  {utterance.t_start:7.1f}s  {utterance.text}")
        spoken = sum(u.t_end - u.t_start for u in found)
        print()
        print(f"  {len(found)} line(s), {spoken:.0f}s of speech")
        return 0

    conn = db.open_db(cfg.db_path)

    def on_progress(done: int, total: int, fraction: float) -> None:
        line = f"  {done + 1}/{total}  {fraction:.0%}"
        print(chr(13) + line, end="", flush=True)

    summary = transcribe_videos(conn, cfg, force=args.force, limit=args.limit,
                                on_progress=on_progress)
    print()
    print(f"  listened to   {summary.videos}")
    print(f"  utterances    {summary.utterances}")
    print(f"  speech        {summary.seconds_of_speech:.0f}s")
    print(f"  no audio      {summary.skipped_no_audio}")
    print(f"  silent        {summary.skipped_silent}")
    for line in summary.failed:
        print(f"  FAILED  {line}")
    for text in summary.samples:
        print(f'    "{text}"')
    if summary.utterances:
        print(f"indexed {rebuild_text_index(conn)} segments for search")
    return 0


def cmd_shortcut(args: argparse.Namespace) -> int:
    """Make a launcher with the app's icon on it."""
    from tsv.shortcut import create, desktop, start_menu

    project = Path(__file__).resolve().parent.parent
    made = []

    targets = [(project / "TextSearchVDO.lnk", "in this folder")]
    if args.desktop:
        where = desktop()
        if where is None:
            print("  could not find your Desktop folder")
        else:
            targets.append((where / "TextSearchVDO.lnk", "on the Desktop"))
    if args.start_menu:
        where = start_menu()
        if where is None:
            print("  could not find your Start Menu folder")
        else:
            targets.append((where / "TextSearchVDO.lnk", "in the Start Menu"))

    for link, where in targets:
        ok, message = create(project, link)
        if ok:
            made.append(where)
            print(f"  created {where}")
        else:
            print(f"  FAILED {where}: {message}")

    if made:
        print()
        print("  Double-click TextSearchVDO - it now has the app's icon, so it")
        print("  is the one to pick rather than the .bat or the .vbs.")
    return 0 if made else 1


def cmd_hardware(args: argparse.Namespace) -> int:
    """What this machine can run, and what it is allowed to ship."""
    from tsv.catalogue import (
        STAGE_TITLES, STAGES, choices, fits, in_use, recommend, unshippable,
    )
    from tsv.hardware import probe

    cfg = _config(args)
    detector = cfg.detect_model_path if cfg.detect_model_path.is_file() else None
    if detector is None:
        print("no detector installed, so GPUs are listed rather than tested.")
        print("run 'python -m tsv setup' first for a real answer.")
        print()

    machine = probe(verify_with=detector)
    print(machine.summary())
    for accelerator in machine.accelerators:
        if accelerator.usable is False:
            print(f"  unusable: {accelerator} - {accelerator.note}")
    print()

    picked = recommend(machine)
    running = in_use(cfg)
    print("  + fits this machine, * running now")
    print()
    for stage in STAGES:
        print(STAGE_TITLES[stage])
        for choice in choices(stage):
            ok, why = fits(choice, machine)
            current = running.get(stage) == choice.key
            mark = "*" if current else ("+" if ok else " ")
            note = ""
            if current:
                note = " <- in use"
            elif picked.get(stage) is choice:
                note = " <- could upgrade to this"
            licence = "" if choice.shippable else f"  [{choice.licence}]"
            print(f"  {mark} {choice.title:<34} {choice.approx_mb:>5} MB{licence}{note}")
            if not ok and why:
                print(f"      {why}")
        print()

    blocked = unshippable()
    if blocked:
        print("Not shippable in a product as it stands:")
        for choice in blocked:
            print(f"  {choice.title} - {choice.licence}")
            if choice.licence_note:
                for line in _wrap(choice.licence_note, 72):
                    print(f"      {line}")
    else:
        print("Every model in use can be distributed commercially.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def cmd_setup(args: argparse.Namespace) -> int:
    """Fetch or build every model the app can use."""
    from tsv.setup import components_for, run_setup, status

    cfg = _config(args)

    if args.check:
        print(f"models in {cfg.model_dir}")
        for component, ready in status(cfg, detector=args.detector):
            mark = "yes" if ready else "no "
            tail = "" if ready else f"  ({component.why})"
            print(f"  [{mark}] {component.title}{tail}")
        absent = [c for c, ready in status(cfg, detector=args.detector) if not ready]
        print()
        print("everything is present." if not absent
              else f"{len(absent)} missing. Run:  python -m tsv setup")
        return 0 if not absent else 1

    only = set(args.only) if args.only else None
    print(f"setting up into {cfg.model_dir}")
    if args.detector:
        print(f"detector: {args.detector}")
    report = run_setup(cfg, only=only, keep_export_env=not args.clean,
                       detector=args.detector)

    print()
    for title in report.installed:
        print(f"  added   {title}")
    for title, why in report.failed:
        print(f"  FAILED  {title}: {why}")

    # Re-check everything, not just what this run touched: with --only, a
    # successful run can still leave the app half installed, and saying
    # "ready" then would be false.
    remaining = [c for c, is_ready in status(cfg, detector=args.detector) if not is_ready]
    ready_now = len(COMPONENTS) - len(remaining)
    print(f"\n{ready_now} of {len(COMPONENTS)} ready in {report.elapsed:.0f}s")

    if report.failed:
        print("some parts did not install; the app runs with less of it until they are.")
        return 1
    if remaining:
        names = ", ".join(c.title.split(" (")[0].lower() for c in remaining)
        print(f"still missing: {names}")
        print("run  python -m tsv setup  without --only to finish")
        return 0

    print("ready. Start it with:  python -m tsv app")
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


def cmd_app(args: argparse.Namespace) -> int:
    """Open the desktop window."""
    cfg = _config(args)
    try:
        from tsv.desktop import run
    except ImportError:
        print("the desktop window needs pywebview:")
        print("  .venv/Scripts/python -m pip install pywebview")
        return 1
    return run(cfg)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from tsv.api import create_app

    app = create_app(_config(args))
    print(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _print_qr(url: str) -> None:
    """A QR in the terminal, if the console can render one.

    Windows consoles still default to cp1252, and segno draws with block
    characters - printing them there raises UnicodeEncodeError and takes the
    whole command down with it. The URL is the thing that matters; the QR is
    a convenience, so it is allowed to fail quietly.
    """
    try:
        import io

        import segno

        buffer = io.StringIO()
        segno.make(url, error="m").terminal(buffer, compact=True)
        drawing = buffer.getvalue()
        drawing.encode(sys.stdout.encoding or "utf-8")
    except Exception:      # noqa: BLE001 - no QR is not a failure
        return
    print(drawing)


def cmd_share(args: argparse.Namespace) -> int:
    """Serve on the local network so a phone can reach this."""
    import uvicorn

    from tsv.api import create_app
    from tsv.share import describe_addresses, local_addresses

    cfg = _config(args)
    addresses = local_addresses()
    offer, warnings = describe_addresses(addresses)

    for warning in warnings:
        print(f"  ! {warning}")
    if not offer and not args.force:
        print()
        print("Nothing safe to share on. Pass --force to bind anyway.")
        return 1

    app = create_app(cfg, share=True)
    # The middleware owns the code, so the console reads it from there rather
    # than holding a second copy that could drift out of step.
    code = app.state.pairing_code() if hasattr(app.state, "pairing_code") else None

    port = args.port
    primary = offer[0].ip if offer else "127.0.0.1"
    url = f"http://{primary}:{port}/pair"

    print()
    print("  Sharing on your local network. Leave this running.")
    print()
    for address in offer:
        print(f"    http://{address.ip}:{port}/pair    ({address.hint})")
    if not offer:
        print(f"    http://{primary}:{port}/pair")
    print()
    if code:
        print(f"    pairing code:  {code[:3]} {code[3:]}")
        print()
    print("  On the phone: join the same WiFi, or plug it in and turn on USB")
    print("  tethering, then open the address above and enter the code.")
    print()

    # The failure that looks exactly like success: everything above is true,
    # and Windows drops every packet from the phone without telling anybody.
    from tsv.share import firewall_allows, firewall_command, firewall_fixer

    if firewall_allows(port) is False:
        print("  Windows Firewall is not letting anything reach this port, so")
        print("  the phone will scan the code and then sit there.")
        print()
        fixer = firewall_fixer()
        if fixer is not None:
            print(f"  Fix it by double-clicking:  {fixer.name}")
            print("  (it asks Windows for permission, so say yes to the prompt)")
        else:
            print("  Run this once in an Administrator PowerShell:")
            print()
            print("    " + firewall_command(port))
        print()
    _print_qr(url)
    print("  Ctrl+C to stop sharing.")
    print()
    # The address and the code are the whole point of this command, and
    # uvicorn is about to take the thread. Flush so they are on screen
    # even when the output is piped somewhere that buffers.
    sys.stdout.flush()

    if hasattr(app.state, "note_lan"):
        app.state.note_lan(port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    """List the phones that have been paired, or remove one."""
    from datetime import datetime as _dt

    from tsv.share import list_devices, revoke_device

    cfg = _config(args)
    conn = db.open_db(cfg.db_path)

    if args.revoke is not None:
        if revoke_device(conn, args.revoke):
            print(f"device {args.revoke} revoked; it will be refused immediately.")
            return 0
        print(f"no device with id {args.revoke}")
        return 1

    devices = list_devices(conn)
    if not devices:
        print("No phones paired. Run `tsv share` and pair one.")
        return 0

    for device in devices:
        seen = _dt.fromtimestamp(device["last_seen"] or device["paired_at"])
        print(f'  [{device["id"]}] {device["name"][:24]:<24} '
              f'last seen {seen:%Y-%m-%d %H:%M}  from {device["address"]}')
    print()
    print("Remove one with:  python -m tsv devices --revoke <id>")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Transcripts and captions are model output in whatever language was
    # spoken, and a Windows console defaults to cp1252. Printing a Bengali
    # line there raises UnicodeEncodeError and takes the whole command down
    # - the transcription worked, and `tsv listen` died reporting it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

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
    p_analyze.add_argument("--captions", action="store_true",
                           help="also describe what people are doing (slow)")
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

    p_caption = sub.add_parser("caption", help="describe what people are doing (slow)")
    p_caption.add_argument("--force", action="store_true", help="re-caption everything")
    p_caption.add_argument("--limit", type=int, help="stop after this many")
    p_caption.set_defaults(func=cmd_caption)

    p_setup = sub.add_parser("setup", help="fetch or build the models (run this first)")
    p_setup.add_argument("--check", action="store_true", help="report what is present, change nothing")
    p_setup.add_argument("--only", nargs="+", metavar="PART",
                         choices=["detector", "faces", "search", "captions", "audio"],
                         help="limit to certain parts")
    p_setup.add_argument("--clean", action="store_true",
                         help="delete the export environment afterwards")
    p_setup.add_argument("--detector", choices=sorted(DETECTORS),
                         help="which detector to install; the yolox ones are "
                              "Apache-2.0 and need no toolchain, yolo11n is AGPL-3.0")
    p_setup.set_defaults(func=cmd_setup)

    p_listen = sub.add_parser("listen", help="transcribe speech in the indexed videos")
    p_listen.add_argument("--force", action="store_true", help="re-transcribe everything")
    p_listen.add_argument("--limit", type=int, help="stop after this many videos")
    p_listen.add_argument("--file", help="transcribe one file and print it, "
                                         "changing nothing - use this to check "
                                         "that transcription works")
    p_listen.set_defaults(func=cmd_listen)

    p_hardware = sub.add_parser(
        "hardware", help="what this machine can run, and what it can ship")
    p_hardware.set_defaults(func=cmd_hardware)

    p_stats = sub.add_parser("stats", help="what is in the index")
    p_stats.set_defaults(func=cmd_stats)

    p_app = sub.add_parser("app", help="open the desktop window")
    p_app.set_defaults(func=cmd_app)

    p_share = sub.add_parser(
        "share", help="serve on the local network so a phone can use it"
    )
    p_share.add_argument("--port", type=int, default=8000)
    p_share.add_argument("--force", action="store_true",
                         help="bind even with no private network address")
    p_share.set_defaults(func=cmd_share)

    p_shortcut = sub.add_parser(
        "shortcut", help="make a launcher with the app icon on it"
    )
    p_shortcut.add_argument("--desktop", action="store_true",
                            help="also put one on the Desktop")
    p_shortcut.add_argument("--start-menu", action="store_true",
                            help="also put one in the Start Menu")
    p_shortcut.set_defaults(func=cmd_shortcut)

    p_devices = sub.add_parser("devices", help="phones that have been paired")
    p_devices.add_argument("--revoke", type=int, metavar="ID",
                           help="remove a phone's access immediately")
    p_devices.set_defaults(func=cmd_devices)

    p_serve = sub.add_parser("serve", help="run the web UI without a window")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))
