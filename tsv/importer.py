"""One call that takes a video file and makes it searchable.

The CLI exposes ingest, analyze and reindex separately because they are
separate concerns and each is worth running alone. Someone who has just
dragged a file onto a window does not care: they want the video searchable,
and they want to see it happening.

Videos are indexed **where they already are**. Copying a night of 1080p
footage into an application folder would double the disk it occupies for no
benefit, and people keep recordings where they keep them on purpose. Only
files arriving over HTTP - a browser upload, where there is no path to point
at - get written to a staging directory.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tsv.analyze import analyze_all
from tsv.config import Config
from tsv.events import recompute_events
from tsv.ingest import ingest_path
from tsv.jobs import Reporter
from tsv.probe import iter_videos
from tsv.search import rebuild_text_index

# Roughly what each stage costs relative to the others, from the measured
# rates: the packet scan runs at hundreds of times realtime, detection at
# about fifteen frames a second. Detection dominates and the bar should say so.
STAGE_SHARES = {"ingest": 0.15, "analyze": 0.75, "index": 0.10}

# With captioning on, it dominates everything else: about six seconds an image
# against roughly fifteen frames a second for detection. The bar has to say so
# or it will appear to stall at the end of a run.
STAGE_SHARES_WITH_CAPTIONS = {
    "ingest": 0.05, "analyze": 0.25, "caption": 0.62, "listen": 0.04, "index": 0.04,
}

# Transcription is cheap next to captioning - measured at over a hundred times
# realtime on a CPU, because voice activity detection skips the silence - so
# it barely moves the bar.
STAGE_SHARES_WITH_AUDIO = {
    "ingest": 0.14, "analyze": 0.70, "listen": 0.06, "index": 0.10,
}


@dataclass
class ImportResult:
    files: int = 0
    segments: int = 0
    tracklets: int = 0
    duration: float = 0.0
    active: float = 0.0
    faces: int = 0
    captions: int = 0
    utterances: int = 0
    skipped: int = 0
    # Files that turned out to be a copy of something already indexed. Counted
    # apart from `skipped`, which means "unchanged since last time": one is
    # nothing to do, the other is worth telling the user about, because they
    # asked for a video and got no new library entry.
    duplicates: list[str] | None = None
    failed: list[str] | None = None

    @property
    def reduction(self) -> float:
        return (1.0 - self.active / self.duration) if self.duration else 0.0

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "segments": self.segments,
            "tracklets": self.tracklets,
            "duration": round(self.duration, 1),
            "active": round(self.active, 1),
            "reduction": round(self.reduction, 4),
            "faces": self.faces,
            "captions": self.captions,
            "utterances": self.utterances,
            "skipped": self.skipped,
            "duplicates": self.duplicates or [],
            "failed": self.failed or [],
        }


def stage_video(source: Path, staging_dir: Path, name: str | None = None) -> Path:
    """Move a received file into the staging area under a final name.

    `name` exists because uploads are written to a ".part-" file first; without
    it the temporary prefix became the video's name in the library, and every
    repeat upload added "-1", "-2" to it.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    wanted = Path(name or source.name)
    target = staging_dir / wanted.name
    counter = 1
    while target.exists():
        target = staging_dir / f"{wanted.stem}-{counter}{wanted.suffix}"
        counter += 1
    shutil.move(str(source), target)
    return target


def import_videos(
    conn: sqlite3.Connection,
    path: Path,
    cfg: Config,
    report: Reporter | None = None,
    force: bool = False,
    with_captions: bool | None = None,
    with_audio: bool | None = None,
) -> ImportResult:
    """Ingest, analyse and index a file or folder, reporting progress."""
    result = ImportResult(failed=[], duplicates=[])
    captions_on = cfg.caption.enabled if with_captions is None else with_captions
    captions_on = captions_on and cfg.has_caption_model
    listen_on = (cfg.audio.enabled if with_audio is None else with_audio)
    listen_on = listen_on and cfg.has_audio_model
    if captions_on:
        shares = STAGE_SHARES_WITH_CAPTIONS
    elif listen_on:
        shares = STAGE_SHARES_WITH_AUDIO
    else:
        shares = STAGE_SHARES
    videos = iter_videos(path)
    if not videos:
        raise ValueError(f"no video files found in {path}")

    # ---- motion segmentation ----
    if report:
        report.stage("Finding activity", shares["ingest"],
                     f"scanning {len(videos)} file(s)")
    done = 0

    def on_ingest(item) -> None:
        nonlocal done
        done += 1
        if item.status == "failed":
            result.failed.append(f"{item.path.name}: {item.note}")
        elif item.status == "duplicate":
            result.duplicates.append(f"{item.path.name}: {item.note}")
        elif item.status == "skipped":
            result.skipped += 1
        else:
            result.files += 1
            result.segments += item.n_segments
            result.duration += item.duration
            result.active += item.active_seconds
        if report:
            report.step(done / len(videos), f"{item.path.name}: {item.status}")

    ingest_path(conn, path, cfg, force=force, on_result=on_ingest)

    # ---- detection, faces, embeddings ----
    pending = conn.execute(
        "SELECT COUNT(DISTINCT video_id) AS n FROM segments WHERE analyzed_at IS NULL"
    ).fetchone()["n"]

    if report:
        report.stage(
            "Recognising objects", shares["analyze"],
            f"{pending} file(s) to analyse" if pending else "nothing new to analyse",
        )

    # The bar is driven by seconds of footage examined across every pending
    # file, so it moves continuously rather than once per file or per segment.
    total_seconds = conn.execute(
        "SELECT COALESCE(SUM(t_end - t_start), 0) AS s FROM segments "
        "WHERE analyzed_at IS NULL"
    ).fetchone()["s"] or 1.0

    if pending:
        seen = 0
        seconds_before = 0.0

        def on_progress(done: float, total: float) -> None:
            if not report:
                return
            overall = min(1.0, (seconds_before + done) / total_seconds)
            report.step(
                overall,
                f"{int(seconds_before + done)}s of {int(total_seconds)}s examined",
            )

        def on_analyze(item) -> None:
            nonlocal seen, seconds_before
            seen += 1
            seconds_before += item.analysed_seconds
            if item.status == "analyzed":
                result.tracklets += item.n_tracklets
                result.faces += item.n_faces
            elif item.status == "failed":
                result.failed.append(f"{item.path.name}: {item.note}")
            if report:
                report.say(f"{item.path.name}: {item.n_tracklets} object(s) found")

        try:
            analyze_all(conn, cfg, force=force, on_result=on_analyze,
                        on_progress=on_progress)
        except FileNotFoundError:
            # No detector present. Motion segmentation still worked and the
            # timeline is usable, so this is a note rather than a failure - but
            # it has to say what to do, not print a path and leave the reader
            # to infer that a model is missing.
            result.failed.append(
                "No object detection model, so only movement was found. "
                "Run: python -m tsv setup"
            )

    # ---- captions, when asked for ----
    if captions_on:
        from tsv.captioning import caption_tracklets

        if report:
            report.stage("Describing what people are doing", shares["caption"],
                         "this is the slow part")

        def on_caption(done: int, total: int) -> None:
            if report and total:
                report.step(done / total, f"{done} of {total} described")

        try:
            captions = caption_tracklets(conn, cfg, on_progress=on_caption)
            result.captions = captions.captioned
        except FileNotFoundError as exc:
            result.failed.append(f"captioning skipped: {exc}")

    # ---- what was said ----
    #
    # After captioning rather than before: it is far cheaper, so putting it
    # last means the expensive stage is not held up by it, and the transcript
    # lands in the same index rebuild below.
    if listen_on:
        from tsv.audio import transcribe_videos

        if report:
            report.stage("Listening for speech", shares["listen"], "")

        def on_listen(done: int, total: int, fraction: float) -> None:
            if report and total:
                report.step((done + fraction) / total, f"{done + 1} of {total}")

        try:
            heard = transcribe_videos(conn, cfg, force=force, on_progress=on_listen)
            result.utterances = heard.utterances
            result.failed.extend(heard.failed)
        except Exception as exc:  # noqa: BLE001 - a missing model is not a failed import
            result.failed.append(f"transcription skipped: {type(exc).__name__}: {exc}")

    if report:
        report.stage("Building the index", shares["index"], "")
    recompute_events(conn)
    if report:
        report.step(0.5, "")
    rebuild_text_index(conn)
    if report:
        report.step(1.0, "ready")

    return result
