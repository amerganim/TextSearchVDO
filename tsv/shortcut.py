"""Give the launcher an icon, by making a shortcut that can carry one.

A .vbs file cannot have its own icon - Windows draws every one of them with
the same generic script glyph, so the thing somebody is meant to double-click
looks identical to the thing they are not. A .lnk can, so this makes one.

It points at the .vbs rather than straight at pythonw, which keeps the checks
that live there: a missing virtual environment becomes a message box, rather
than a double-click that appears to do nothing at all.

Written with PowerShell's WScript.Shell rather than pywin32, because pywin32
is not a dependency of this project and adding one to draw an icon would be a
poor trade.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = """
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut('{link}')
$s.TargetPath = '{target}'
$s.Arguments = '{arguments}'
$s.WorkingDirectory = '{workdir}'
$s.IconLocation = '{icon}'
$s.Description = 'Search your videos by describing what you are looking for'
$s.Save()
"""


def create(
    project: Path,
    link: Path,
    icon: Path | None = None,
) -> tuple[bool, str]:
    """Make one shortcut. Returns (worked, message)."""
    if sys.platform != "win32":
        return False, "shortcuts are a Windows thing"

    launcher = project / "TextSearchVDO.vbs"
    if not launcher.is_file():
        return False, f"no launcher at {launcher}"

    icon = icon or (project / "TextSearchVDO.ico")
    # An IconLocation pointing at a file that is not there leaves the shortcut
    # with the generic glyph and no explanation, so fall back to the launcher
    # rather than silently producing the thing this exists to avoid.
    icon_location = f"{icon},0" if icon.is_file() else str(launcher)

    script = SCRIPT.format(
        link=str(link).replace("'", "''"),
        target=r"C:\Windows\System32\wscript.exe",
        arguments=f'"{launcher}"'.replace("'", "''"),
        workdir=str(project).replace("'", "''"),
        icon=icon_location.replace("'", "''"),
    )

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run PowerShell: {exc}"

    if result.returncode != 0 or not link.is_file():
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, detail[-1] if detail else "the shortcut was not created"
    return True, str(link)


def desktop() -> Path | None:
    home = Path.home()
    for candidate in (home / "Desktop", home / "OneDrive" / "Desktop"):
        if candidate.is_dir():
            return candidate
    return None


def start_menu() -> Path | None:
    import os

    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    programs = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    return programs if programs.is_dir() else None
