"""The last three lines of `tsv setup`.

Found by installing onto a machine that had never seen this project. Every
model downloaded - five of five, reported as added - and then the summary line
raised NameError on a name that does not exist, so the command exited 1.

That exit code is what makes it more than cosmetic. setup.bat reads it, prints
"Some parts did not install", and skips creating the desktop shortcut. A new
user's first experience is a failure message and no icon, for an install that
completely succeeded.

Nothing above this ran it: a NameError on a line only reached after a real
download is invisible to a test suite that never downloads anything.
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from tsv import cli
from tsv.config import DEFAULT
from tsv.setup import SetupReport, components_for


@pytest.fixture
def args(tmp_path):
    return argparse.Namespace(
        data_dir=tmp_path, model_dir=None, only=None, clean=False,
        detector=None, check=False,
    )


def _stub_setup(monkeypatch, report: SetupReport, ready: bool = True):
    """Stand in for the download, so the summary after it can be reached."""
    monkeypatch.setattr("tsv.setup.run_setup", lambda *a, **k: report)
    monkeypatch.setattr(
        "tsv.setup.status",
        lambda cfg, detector=None: [(c, ready) for c in components_for(detector)],
    )


def test_the_summary_runs_after_a_successful_install(args, monkeypatch, capsys):
    """The exact path that failed: everything installed, nothing missing."""
    _stub_setup(monkeypatch, SetupReport(installed=["Object detection"], elapsed=12.0))

    assert cli.cmd_setup(args) == 0
    out = capsys.readouterr().out
    assert "ready in 12s" in out
    assert "ready. Start it with" in out


def test_it_counts_against_the_catalogue_that_was_actually_used(args, monkeypatch, capsys):
    """With a detector chosen the catalogue is not the default one, so a count
    taken from the wrong list would be quietly wrong rather than loud."""
    args.detector = "yolox-tiny"
    _stub_setup(monkeypatch, SetupReport(installed=["Object detection"], elapsed=1.0))

    cli.cmd_setup(args)
    total = len(components_for("yolox-tiny"))
    assert f"{total} of {total} ready" in capsys.readouterr().out


def test_a_partial_install_is_reported_without_crashing(args, monkeypatch, capsys):
    _stub_setup(monkeypatch, SetupReport(installed=[], elapsed=3.0), ready=False)

    assert cli.cmd_setup(args) == 0
    out = capsys.readouterr().out
    assert "still missing" in out


def test_a_failed_component_still_exits_nonzero(args, monkeypatch, capsys):
    """The exit code has to keep meaning something, or setup.bat cannot use
    it - which is the whole reason the NameError mattered."""
    _stub_setup(monkeypatch, SetupReport(failed=[("CLIP", "download failed")], elapsed=2.0))

    assert cli.cmd_setup(args) == 1
    assert "did not install" in capsys.readouterr().out
