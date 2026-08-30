"""Running another program without a console window appearing.

The app is started by pythonw.exe, which has no console of its own - that is
the entire reason the .vbs launcher exists. But a process with no console that
starts a console program *gets given one*, and Windows shows it. So every
PowerShell probe in here - what kind of network this is, whether the firewall
has a rule, creating the desktop shortcut - flashed a black window in the
user's face, in the middle of whatever they were doing.

Nothing was wrong. That is what made it hard to place: the windows appear at
moments that have no obvious connection to each other, because what they have
in common is not a feature but a system call.

`CREATE_NO_WINDOW` is the fix, and it has to be everywhere, because one missed
call site brings the flicker back and looks like the fix did not work.
"""

from __future__ import annotations

import subprocess
import sys

# Only defined on Windows; on anything else there is no window to suppress.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run`, with no console window on Windows.

    Deliberately a thin pass-through rather than something with opinions about
    capture or timeouts: callers already differ on those, and a wrapper that
    quietly changed them would be a second thing to debug.
    """
    kwargs.setdefault("creationflags", 0)
    kwargs["creationflags"] |= NO_WINDOW
    return subprocess.run(command, **kwargs)


def popen(command: list[str], **kwargs) -> subprocess.Popen:
    """`subprocess.Popen`, with no console window on Windows."""
    kwargs.setdefault("creationflags", 0)
    kwargs["creationflags"] |= NO_WINDOW
    return subprocess.Popen(command, **kwargs)
