"""No console window when the app runs another program.

The app is launched by pythonw.exe, which deliberately has no console. But a
process without a console that starts a console program is *given* one, and
Windows shows it - so a black window flashed up at moments with no obvious
connection to each other: opening the app, opening the Share panel, making a
shortcut. What they had in common was not a feature, it was a system call.

These tests are mostly about coverage rather than behaviour, because the bug
was never that the flag was wrong somewhere. It was that one call site did not
have it - and one is enough to bring the flicker back and make the fix look
like it did not work.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tsv import proc

TSV = Path(__file__).resolve().parent.parent / "tsv"


def test_the_flag_is_set_on_windows_and_absent_elsewhere():
    if sys.platform == "win32":
        assert proc.NO_WINDOW == subprocess.CREATE_NO_WINDOW
    else:
        assert proc.NO_WINDOW == 0


def test_a_command_still_runs_and_still_reports():
    """Suppressing the window must not suppress the output, which is the
    whole reason these calls exist."""
    done = proc.run(
        [sys.executable, "-c", "print('hello')"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "hello"


def test_a_failure_is_still_a_failure():
    done = proc.run(
        [sys.executable, "-c", "raise SystemExit(3)"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 3


def test_creationflags_a_caller_passes_are_kept(monkeypatch):
    """Or-ed rather than overwritten: a caller that needs its own flag should
    not have to choose between that and a hidden window."""
    seen = {}

    def fake(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    proc.run(["x"], creationflags=0x00000200)
    assert seen["creationflags"] & 0x00000200
    assert seen["creationflags"] & proc.NO_WINDOW == proc.NO_WINDOW


def _spawning_calls(path: Path) -> list[str]:
    """Every way of starting a program, as written in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            name = f"{target.value.id}.{target.attr}"
            if name in ("subprocess.run", "subprocess.Popen", "os.system", "os.startfile"):
                found.append(name)
    return found


@pytest.mark.parametrize(
    "module", sorted(p.name for p in TSV.rglob("*.py") if p.name != "proc.py")
)
def test_nothing_starts_a_program_the_raw_way(module):
    """The one that matters.

    Every direct spawn is a console window waiting to happen, and it will be
    found by a user rather than by us - it costs nothing at runtime and leaves
    no trace to grep for afterwards. `tsv.proc` is the only place allowed to
    call the real thing.
    """
    for path in TSV.rglob(module):
        assert _spawning_calls(path) == [], (
            f"{path.name} starts a program directly; use tsv.proc.run instead"
        )
