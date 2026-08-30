"""Loading the text encoder once, and loading it early.

Measured with seven paired phones searching a freshly started server, the
first search took 10.7 seconds and every one after it took under 300ms. Two
separate faults met there. The encoder was built lazily with no lock, so all
seven arrivals saw it missing and all seven started building it on one set of
cores. And even built once, the first person to type still pays for it.

After a lock and a warm-up thread, the same test's worst request is 611ms.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from tsv import db
from tsv.api import create_app
from tsv.config import DEFAULT


@pytest.fixture
def cfg(tmp_path):
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()
    return cfg


def test_the_encoder_is_built_once_however_many_arrive_together(cfg, monkeypatch):
    """The race, stated directly.

    Without the lock every concurrent first search builds its own encoder.
    Nothing breaks, which is why it survived - it is just the slowest possible
    way to start, at exactly the moment a room full of people is watching.
    """
    builds = []
    building = threading.Event()

    def slow_build(*args, **kwargs):
        builds.append(1)
        building.set()
        time.sleep(0.3)          # long enough for the others to pile in
        return object()

    monkeypatch.setattr("tsv.models.clip.build_clip", slow_build)
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: True))

    app = create_app(cfg)
    load = app.state.load_text_encoder

    threads = [threading.Thread(target=load) for _ in range(7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(builds) == 1, f"built {sum(builds)} times, should be once"


def test_every_caller_gets_the_same_encoder(cfg, monkeypatch):
    monkeypatch.setattr("tsv.models.clip.build_clip", lambda *a, **k: object())
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: True))

    app = create_app(cfg)
    load = app.state.load_text_encoder
    assert load() is load()


def test_warm_is_off_by_default(cfg, monkeypatch):
    """The suite builds this app hundreds of times and none of them want to
    load a 243 MB model."""
    built = []
    monkeypatch.setattr("tsv.models.clip.build_clip",
                        lambda *a, **k: built.append(1))
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: True))

    create_app(cfg)
    time.sleep(0.2)
    assert built == []


def test_warm_loads_without_being_asked(cfg, monkeypatch):
    built = threading.Event()

    def build(*args, **kwargs):
        built.set()
        return object()

    monkeypatch.setattr("tsv.models.clip.build_clip", build)
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: True))

    create_app(cfg, warm=True)
    assert built.wait(timeout=5), "warm=True did not start loading"


def test_a_warm_up_that_fails_does_not_stop_the_app(cfg, monkeypatch):
    """An optimisation that can prevent the app starting is not one. Whatever
    went wrong here happens again on the first real search, where there is
    somebody to report it to."""
    def explode(*args, **kwargs):
        raise RuntimeError("no such model")

    monkeypatch.setattr("tsv.models.clip.build_clip", explode)
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: True))

    app = create_app(cfg, warm=True)
    time.sleep(0.3)
    assert app is not None


def test_nothing_is_loaded_when_there_are_no_models(cfg, monkeypatch):
    built = []
    monkeypatch.setattr("tsv.models.clip.build_clip", lambda *a, **k: built.append(1))
    monkeypatch.setattr(type(cfg), "has_clip_models", property(lambda self: False))

    create_app(cfg, warm=True)
    time.sleep(0.2)
    assert built == []
