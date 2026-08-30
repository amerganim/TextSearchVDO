"""The desktop window.

Runs the same FastAPI server on a private port and points a native window at
it. That keeps one implementation of the app rather than two: the window, a
plain browser and the planned Android client all talk to the same endpoints.

The window earns its place by giving the app things a browser tab cannot:
a native file dialog that returns real paths, so a video is indexed where it
already lives instead of being uploaded to the machine it is already on.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from tsv import single

VIDEO_TYPES = ("Video files (*.mp4;*.mkv;*.avi;*.mov;*.m4v;*.ts;*.dav;*.flv)", "All files (*.*)")

# Shown the instant the window exists, then replaced by the app.
#
# Starting up costs about a second and a half that cannot be removed - Python
# imports, then uvicorn binding a socket - and a window that appears only at
# the end of it reads as a slow application. A window that appears at once and
# says what it is doing reads as a fast one, and it is the same second and a
# half. Deliberately inline and dependency-free: nothing here can be served,
# because the thing that serves it is what we are waiting for.
SPLASH = """
<!doctype html><meta charset="utf-8">
<style>
  html, body { height: 100%; margin: 0; }
  body {
    background: #0f1218; color: #e8ecf4;
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    display: grid; place-items: center;
  }
  .box { text-align: center; }
  .mark {
    width: 15px; height: 15px; border-radius: 4px; background: #4da3ff;
    box-shadow: inset 0 0 0 3px #0f1218; margin: 0 auto 14px;
  }
  .name { font-weight: 650; letter-spacing: .01em; }
  .what { color: #8a93a6; font-size: 13px; margin-top: 6px; }
</style>
<div class="box">
  <div class="mark"></div>
  <div class="name">TextSearchVDO</div>
  <div class="what">Starting&hellip;</div>
</div>
"""


# The window's own icon. Without it the Windows backend falls back to reading
# one out of sys.executable - which is pythonw.exe, so the app wore the Python
# logo in its title bar, its taskbar button and its Alt-Tab card. The .ico is
# beside this package rather than inside data/, because it is part of the
# application, not of anyone's index.
ICON = Path(__file__).resolve().parent.parent / "TextSearchVDO.ico"


def _own_the_taskbar_button(app_id: str = "TextSearchVDO.App") -> None:
    """Stop Windows filing this window under Python.

    Separate from the icon and needed as well as it. The taskbar groups by
    Application User Model ID, and a process that never sets one inherits the
    host executable's - so the button stayed Python's even once the title bar
    was right. Best-effort: on any other platform, and on a Windows where this
    call is unavailable, the app simply looks the way it did before.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:  # noqa: BLE001 - cosmetic; never worth failing to start
        pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until the server answers.

    Fine-grained on purpose. The server is usually up in well under a tenth of
    a second, and a 200 ms sleep between attempts spent most of the wait doing
    nothing - which the user sees as the app being slow to appear.
    """
    deadline = time.time() + timeout
    delay = 0.005
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - still starting
            time.sleep(delay)
            delay = min(delay * 1.5, 0.1)
    return False


class Bridge:
    """The small surface the page can call into.

    Only file picking: everything else goes over HTTP like any other client,
    so the browser and the window cannot drift apart in behaviour.
    """

    def __init__(self) -> None:
        self._window = None

    def attach(self, window) -> None:
        self._window = window

    def pick_videos(self) -> list[str]:
        import webview

        if self._window is None:
            return []
        chosen = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=True, file_types=VIDEO_TYPES
        )
        return [str(Path(p)) for p in (chosen or [])]

    def pick_folder(self) -> list[str]:
        import webview

        if self._window is None:
            return []
        chosen = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        return [str(Path(p)) for p in (chosen or [])]


def run(cfg, title: str = "TextSearchVDO", width: int = 1180, height: int = 800) -> int:
    """Start the server and open the window. Blocks until the window closes."""
    import uvicorn
    import webview

    from tsv.api import create_app

    _own_the_taskbar_button()

    # One window per index. A second copy is not visibly wrong - it is just a
    # window showing the same thing - so it goes unnoticed until something
    # behaves impossibly, which is how the duplicated pairing code presented.
    already = single.running(cfg)
    if already is not None:
        single.focus(already.pid)
        return 0

    port = _free_port()
    single.claim(cfg, port)
    # Sharing capability on, listening on loopback only. The middleware
    # exempts this machine, so the window is unaffected; turning sharing on
    # from the Share panel adds a second listener rather than changing
    # what this one does.
    app = create_app(cfg, share=True, warm=True)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        # Nothing here uses the lifespan protocol or websockets, and skipping
        # them is measurably less to do before the socket answers.
        lifespan="off", ws="none", access_log=False,
    ))
    threading.Thread(target=server.run, daemon=True).start()

    # The window opens on the splash immediately and moves to the app when the
    # server answers, rather than the user watching nothing for a second and a
    # half and then getting everything at once.
    base = f"http://127.0.0.1:{port}"
    bridge = Bridge()
    window = webview.create_window(
        title, html=SPLASH, js_api=bridge, width=width, height=height,
        min_size=(900, 620), background_color="#0f1218",
    )
    bridge.attach(window)

    def open_when_ready() -> None:
        if _wait_for_server(f"{base}/api/summary"):
            window.load_url(base)
        else:
            window.load_html(
                SPLASH.replace(
                    "Starting&hellip;",
                    "The local server did not start. Close and try again.",
                )
            )

    # icon= is documented as GTK/QT only, but the Windows backend reads it too
    # and only falls back to sys.executable when it is unset - which is what
    # produced the Python logo. Passed as a str because that fallback is
    # guarded by os.path.isfile, and a missing file lands back on pythonw.exe.
    try:
        webview.start(open_when_ready, icon=str(ICON) if ICON.is_file() else None)
    finally:
        # In a finally because a crash that kept the lock would shut the user
        # out of their own app until they found a file they have never heard
        # of. The port probe on the next start covers what this misses.
        single.release(cfg)
    server.should_exit = True
    return 0
