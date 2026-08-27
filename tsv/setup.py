"""One command that makes a fresh checkout ready to use.

Getting the models used to be nine manual steps: two virtual environments,
four installs and four scripts, in an order nobody could be expected to
remember. This does all of it, skips whatever is already there, and says what
it is doing.

**Why a second environment exists at all.** Two of the four model sets have to
be exported rather than downloaded - Ultralytics publishes YOLO11 only as
PyTorch checkpoints, and the CLIP ONNX on the hub is a single fused graph
rather than the separate image and text encoders this runtime uses. Exporting
needs torch, which is a couple of gigabytes. Keeping it in a throwaway
environment is what stops the *running* application from carrying it: at
inference this is ONNX Runtime and numpy, and that is the difference between
something shippable and something that is not.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tsv.config import Config

EXPORT_VENV = Path(".venv-export")


@dataclass
class Component:
    key: str
    title: str
    why: str
    approx_mb: int
    ready: Callable[[Config], bool]
    packages: tuple[str, ...]
    command: tuple[str, ...]
    # Without it the app still runs, just with less of it.
    optional: bool = False


def _export_python() -> Path:
    name = "python.exe" if sys.platform == "win32" else "python"
    return EXPORT_VENV / ("Scripts" if sys.platform == "win32" else "bin") / name


COMPONENTS: tuple[Component, ...] = (
    Component(
        key="detector",
        title="Object detection (YOLO11n)",
        why="finds people, vehicles and animals; without it a video is only motion",
        approx_mb=11,
        ready=lambda cfg: cfg.detect_model_path.is_file(),
        packages=("ultralytics", "onnx", "onnxslim"),
        command=("tools/export_model.py", "--out"),
    ),
    Component(
        key="faces",
        title="Face recognition (SCRFD + ArcFace)",
        why="lets a person be named once and recognised elsewhere",
        approx_mb=16,
        ready=lambda cfg: cfg.has_face_models,
        packages=("insightface",),
        command=("tools/fetch_face_models.py", "--out"),
        optional=True,
    ),
    Component(
        key="search",
        title="Semantic search (CLIP)",
        why="searching by description rather than by object name",
        approx_mb=605,
        ready=lambda cfg: cfg.has_clip_models,
        packages=("transformers",),
        command=("tools/export_clip.py", "--out"),
        optional=True,
    ),
    Component(
        key="captions",
        title="Descriptions (Florence-2)",
        why="describes what a person is doing, so actions become searchable",
        approx_mb=275,
        ready=lambda cfg: cfg.has_caption_model,
        packages=("huggingface_hub",),
        command=("tools/fetch_caption_model.py", "--out"),
        optional=True,
    ),
)


@dataclass
class SetupReport:
    present: list[str] = field(default_factory=list)
    installed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed


def status(cfg: Config) -> list[tuple[Component, bool]]:
    return [(component, component.ready(cfg)) for component in COMPONENTS]


def _run(command: list[str], label: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=3600, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"{label} timed out"
    except OSError as exc:
        return False, f"{label} could not start: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, f"{label} failed: {tail[-1] if tail else 'no output'}"
    return True, ""


def ensure_export_env(log: Callable[[str], None]) -> tuple[bool, str]:
    """Create the throwaway environment used for exporting, if absent."""
    python = _export_python()
    if python.is_file():
        return True, ""

    log(f"creating {EXPORT_VENV} (used only to build the models, never to run)")
    ok, error = _run([sys.executable, "-m", "venv", str(EXPORT_VENV)], "creating the export environment")
    if not ok:
        return False, error
    return _run(
        [str(python), "-m", "pip", "install", "--upgrade", "--quiet", "pip"],
        "upgrading pip",
    )


def run_setup(
    cfg: Config,
    only: set[str] | None = None,
    log: Callable[[str], None] = print,
    keep_export_env: bool = True,
) -> SetupReport:
    """Fetch or build every model that is not already present."""
    started = time.time()
    report = SetupReport()

    wanted = [c for c in COMPONENTS if only is None or c.key in only]
    missing = [c for c in wanted if not c.ready(cfg)]
    report.present = [c.title for c in wanted if c.ready(cfg)]

    for title in report.present:
        log(f"  have    {title}")
    if not missing:
        report.elapsed = time.time() - started
        return report

    total_mb = sum(c.approx_mb for c in missing)
    log(f"\n{len(missing)} to fetch, roughly {total_mb} MB plus a one-off toolchain")

    ok, error = ensure_export_env(log)
    if not ok:
        report.failed.append(("export environment", error))
        report.elapsed = time.time() - started
        return report

    python = str(_export_python())
    packages = sorted({p for c in missing for p in c.packages})
    log(f"installing build tools: {', '.join(packages)}")
    ok, error = _run([python, "-m", "pip", "install", "--quiet", *packages], "installing build tools")
    if not ok:
        report.failed.append(("build tools", error))
        report.elapsed = time.time() - started
        return report

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    for component in missing:
        log(f"  build   {component.title} (~{component.approx_mb} MB)")
        script, *flags = component.command
        ok, error = _run([python, script, *flags, str(cfg.model_dir)], component.title)

        if ok and component.ready(cfg):
            report.installed.append(component.title)
        else:
            # A tool can exit zero and still not produce what was wanted.
            report.failed.append(
                (component.title, error or "finished but produced no model files")
            )
            log(f"  FAILED  {component.title}: {report.failed[-1][1]}")

    if not keep_export_env and report.ok:
        log(f"removing {EXPORT_VENV}")
        shutil.rmtree(EXPORT_VENV, ignore_errors=True)

    report.elapsed = time.time() - started
    return report


def missing_summary(cfg: Config) -> str:
    """One line naming what is absent, for the app to show a user."""
    absent = [c for c in COMPONENTS if not c.ready(cfg)]
    if not absent:
        return ""
    names = ", ".join(c.title.split(" (")[0].lower() for c in absent)
    return f"Not set up yet: {names}. Run: python -m tsv setup"
