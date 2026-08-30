"""One window per index.

Two copies of the app were running here for half an hour and neither of us
noticed. That is the problem: a second window is not obviously wrong, it is
just a window, and the two look identical because they are showing the same
index. What gives it away is behaviour that makes no sense - which is exactly
how the pairing bug presented, where two processes each held their own code
and typing either was refused.

**Per index, not per machine.** The lock lives beside the database, so two
windows on two different indexes are fine - that is a person working on two
sets of footage, not a mistake. What must not happen twice is two processes
sharing one index.

**Liveness is a request, not a PID.** A lock file naming a dead process is the
normal case, not an edge case: the app is closed by the window's X, and it is
killed by Task Manager and by a machine turning off. Trusting the PID alone
also breaks the other way, because Windows reuses PIDs, so a stale lock can
name something real and unrelated - and the app would refuse to start,
pointing at a process that has nothing to do with it. Asking the recorded port
whether the app answers there settles both: a dead instance cannot reply, and
neither can whatever inherited its number.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

LOCK_NAME = "app.lock"

# Short: this runs before the window appears, and a slow check here is
# indistinguishable to a user from a slow application.
PROBE_TIMEOUT = 1.5


@dataclass
class Running:
    """An instance that is already up, and answering."""

    pid: int
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def lock_path(cfg) -> Path:
    return Path(cfg.data_dir) / LOCK_NAME


def _answers(port: int) -> bool:
    """Whether our app - not merely something - is listening there."""
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/summary", timeout=PROBE_TIMEOUT) as r:
            if r.status != 200:
                return False
            json.loads(r.read().decode("utf-8"))
            return True
    except Exception:  # noqa: BLE001 - anything at all means "not our instance"
        return False


def running(cfg) -> Running | None:
    """The instance already using this index, if there is one."""
    path = lock_path(cfg)
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
        pid, port = int(held["pid"]), int(held["port"])
    except Exception:  # noqa: BLE001 - absent, empty, half-written, or garbage
        return None

    if pid == os.getpid() or not _answers(port):
        return None
    return Running(pid=pid, port=port)


def claim(cfg, port: int) -> Path:
    """Record this process as the one holding the index."""
    path = lock_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8"
    )
    return path


def release(cfg) -> None:
    """Give up the lock. Best-effort: a lock left behind is recoverable,
    because the next start probes the port rather than believing the file."""
    try:
        lock_path(cfg).unlink(missing_ok=True)
    except OSError:
        pass


def focus(pid: int) -> bool:
    """Bring the existing window forward.

    The point of the whole exercise. Refusing to open a second window and
    doing nothing else looks like the app failed to start, which is worse
    than the duplicate - at least the duplicate appeared.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def visit(hwnd, _):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length:                      # skip the invisible helpers
                    found.append(hwnd)
            return True

        user32.EnumWindows(visit, 0)
        if not found:
            return False

        window = found[0]
        user32.ShowWindow(window, 9)            # SW_RESTORE, in case minimised
        if user32.SetForegroundWindow(window):
            return True

        # Windows refuses focus to a process that is not already foreground,
        # which is most of the time here. Flashing the taskbar button is the
        # honest fallback: it points at the window that already exists rather
        # than pretending nothing happened.
        user32.FlashWindow(window, True)
        return True
    except Exception:  # noqa: BLE001 - cosmetic, never worth failing to start
        return False
