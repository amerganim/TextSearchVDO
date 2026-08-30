"""One window per index.

Two copies ran here for half an hour unnoticed, because a second window is
not visibly wrong - it is just a window, showing the same index. It surfaces
later as behaviour that cannot happen, which is exactly how the duplicated
pairing code presented.

Most of these are about the ways a guard like this fails *closed* and locks
someone out of their own application. That is a worse bug than the one being
fixed: a duplicate window is a nuisance, an app that will not start is not.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tsv import single
from tsv.config import DEFAULT


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(DEFAULT, data_dir=tmp_path)


@pytest.fixture
def an_app_listening():
    """Something that answers /api/summary the way the real app does."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/api/summary":
                self.send_error(404)
                return
            body = json.dumps({"videos": 0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


# ---------- the case it exists for ----------

def test_a_second_start_finds_the_first(cfg, an_app_listening):
    single.claim(cfg, an_app_listening)
    # Another process, so the "that is me" shortcut does not apply.
    single.lock_path(cfg).write_text(
        json.dumps({"pid": os.getpid() + 1, "port": an_app_listening})
    )

    found = single.running(cfg)
    assert found is not None
    assert found.port == an_app_listening


def test_the_first_start_finds_nothing(cfg):
    assert single.running(cfg) is None


# ---------- the ways it must not lock anyone out ----------

def test_a_lock_naming_a_dead_instance_is_ignored(cfg):
    """The normal case, not an edge case: the app is closed with the X, and
    killed by Task Manager and by the machine turning off."""
    single.lock_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    single.lock_path(cfg).write_text(json.dumps({"pid": 999999, "port": 9}))
    assert single.running(cfg) is None


def test_a_reused_pid_does_not_block_a_start(cfg):
    """Windows reuses PIDs. A stale lock can name a real, unrelated process,
    and believing the PID would refuse to start and point at something with
    nothing to do with this app."""
    single.lock_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    # A PID that certainly exists - this one - on a port nothing answers.
    single.lock_path(cfg).write_text(
        json.dumps({"pid": os.getpid() + 1, "port": 9})
    )
    assert single.running(cfg) is None


def test_something_else_on_the_port_is_not_us(cfg):
    """A port answering is not enough; it has to answer like this app."""

    class Rude(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Rude)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        single.lock_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        single.lock_path(cfg).write_text(
            json.dumps({"pid": os.getpid() + 1, "port": server.server_address[1]})
        )
        assert single.running(cfg) is None
    finally:
        server.shutdown()


@pytest.mark.parametrize("rubbish", ["", "not json", "{}", '{"pid": "x"}', "[]"])
def test_an_unreadable_lock_is_ignored(cfg, rubbish):
    """A half-written file after a power cut must not be permanent."""
    single.lock_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    single.lock_path(cfg).write_text(rubbish)
    assert single.running(cfg) is None


def test_our_own_lock_does_not_stop_us(cfg, an_app_listening):
    """Re-reading a lock this process wrote must not be read as a rival."""
    single.claim(cfg, an_app_listening)
    assert single.running(cfg) is None


# ---------- scope ----------

def test_two_different_indexes_are_two_legitimate_windows(tmp_path, an_app_listening):
    """Per index, not per machine. Someone working on two sets of footage is
    not making a mistake."""
    one = dataclasses.replace(DEFAULT, data_dir=tmp_path / "one")
    two = dataclasses.replace(DEFAULT, data_dir=tmp_path / "two")

    single.claim(one, an_app_listening)
    single.lock_path(one).write_text(
        json.dumps({"pid": os.getpid() + 1, "port": an_app_listening})
    )

    assert single.running(one) is not None
    assert single.running(two) is None


def test_the_lock_sits_beside_the_index(cfg):
    assert single.lock_path(cfg).parent == cfg.data_dir


# ---------- housekeeping ----------

def test_releasing_lets_the_next_start_through(cfg, an_app_listening):
    single.claim(cfg, an_app_listening)
    single.release(cfg)
    assert not single.lock_path(cfg).exists()
    assert single.running(cfg) is None


def test_releasing_twice_is_not_an_error(cfg):
    single.release(cfg)
    single.release(cfg)


def test_the_window_is_released_even_if_the_window_crashes():
    """A crash that kept the lock would shut the user out of their own app
    until they found a file they have never heard of."""
    import inspect

    from tsv import desktop

    source = inspect.getsource(desktop.run)
    assert "finally:" in source
    assert "single.release" in source


def test_focus_never_raises():
    """Cosmetic. Failing to raise a window must not fail to start the app."""
    assert single.focus(999999) in (True, False)
