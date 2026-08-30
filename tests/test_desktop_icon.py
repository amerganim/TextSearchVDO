"""The window's own icon.

The app wore the Python logo - in its title bar, its taskbar button and its
Alt-Tab card - which makes it look like a script someone left running rather
than an application. The cause is a quiet fallback: pywebview's Windows
backend reads the icon out of `sys.executable` when none is set, and
`sys.executable` is pythonw.exe.

The fallback is guarded by `os.path.isfile`, which is what makes this worth
testing rather than eyeballing once. A missing or unreadable .ico does not
raise; it lands silently back on the Python logo, and the only way to notice
is to look at the title bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tsv.desktop import ICON, _own_the_taskbar_button


def test_the_icon_ships_with_the_app():
    assert ICON.is_file(), f"no icon at {ICON}"


def test_it_is_a_real_icon_file():
    """.ico has a header of its own; a renamed .png would pass is_file and
    then fall back to Python at runtime."""
    header = ICON.read_bytes()[:4]
    assert header == b"\x00\x00\x01\x00", "not an .ico"


def test_it_carries_the_sizes_windows_asks_for():
    """Windows picks a size per surface and scales badly when it has to
    invent one: 16 for the title bar, 32 for Alt-Tab, 256 for large taskbar
    icons and the file list."""
    Image = pytest.importorskip("PIL.Image")
    sizes = {size[0] for size in Image.open(ICON).info.get("sizes", [])}
    assert {16, 32, 48, 256} <= sizes, f"missing sizes, has {sorted(sizes)}"


@pytest.mark.skipif(sys.platform != "win32", reason="WinForms is Windows-only")
def test_the_windows_backend_can_load_it():
    """The backend loads it through System.Drawing, which is stricter than
    Pillow - so a file Pillow opens is not proof of anything on its own."""
    clr = pytest.importorskip("clr")
    clr.AddReference("System.Drawing")
    from System.Drawing import Icon as WinIcon

    assert WinIcon(str(ICON)).Width > 0


def test_the_icon_is_actually_passed_to_the_window():
    """It is easy to ship an icon and never hand it over, which looks exactly
    like having no icon at all."""
    import inspect

    from tsv import desktop

    source = inspect.getsource(desktop.run)
    assert "webview.start" in source
    assert "icon=" in source


def test_claiming_the_taskbar_button_never_stops_the_app():
    """Cosmetic, so it must not be able to prevent a window opening - on an
    older Windows, or anything that is not Windows at all."""
    _own_the_taskbar_button()
    _own_the_taskbar_button("TextSearchVDO.Test")
