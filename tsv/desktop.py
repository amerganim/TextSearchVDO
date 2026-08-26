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
import threading
import time
from pathlib import Path
from urllib.request import urlopen

VIDEO_TYPES = ("Video files (*.mp4;*.mkv;*.avi;*.mov;*.m4v;*.ts;*.dav;*.flv)", "All files (*.*)")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - still starting
            time.sleep(0.2)
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

    port = _free_port()
    app = create_app(cfg)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    if not _wait_for_server(f"{base}/api/summary"):
        print(f"the local server did not start on {base}")
        return 1

    bridge = Bridge()
    window = webview.create_window(
        title, base, js_api=bridge, width=width, height=height,
        min_size=(900, 620), background_color="#0f1218",
    )
    bridge.attach(window)

    webview.start()
    server.should_exit = True
    return 0
