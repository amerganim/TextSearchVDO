"""First-run setup: knowing what is missing, and saying so."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tsv.config import DEFAULT
from tsv.setup import COMPONENTS, missing_summary, status


@pytest.fixture
def empty(tmp_path):
    """A configuration pointing at a directory with no models in it."""
    return dataclasses.replace(DEFAULT, data_dir=tmp_path)


def test_every_component_reports_absent_on_a_fresh_install(empty):
    assert [ready for _, ready in status(empty)] == [False] * len(COMPONENTS)


def test_every_component_explains_what_it_is_for():
    """The check output is the first thing a new user reads."""
    for component in COMPONENTS:
        assert component.why and component.why[0].islower()
        assert component.title
        assert component.approx_mb > 0
        assert component.command[0].startswith("tools/")
        # Build tools are only needed by components that build something.
        # Anything published as ONNX or ctranslate2 already - YOLOX, Whisper -
        # is a plain download, and demanding packages here would drag the
        # two-gigabyte torch toolchain in for no reason.
        if component.needs_export_env:
            assert component.packages, f"{component.key} builds but names no tools"
        else:
            assert not component.packages, f"{component.key} downloads but wants tools"


def test_the_missing_summary_is_actionable(empty):
    summary = missing_summary(empty)
    assert "python -m tsv setup" in summary
    assert "object detection" in summary


def test_nothing_missing_means_no_summary():
    """A fully installed copy must not nag."""
    if not all(ready for _, ready in status(DEFAULT)):
        pytest.skip("this checkout is not fully set up")
    assert missing_summary(DEFAULT) == ""


def test_each_component_names_a_tool_that_exists():
    root = Path(__file__).resolve().parent.parent
    for component in COMPONENTS:
        assert (root / component.command[0]).is_file(), component.command[0]


def test_only_the_detector_is_required():
    """Everything else degrades: no faces, no search, no descriptions, no
    speech - but the timeline still works, so none may be marked required."""
    optional = {c.key for c in COMPONENTS if c.optional}
    assert optional == {"faces", "search", "captions", "audio"}
    assert [c.key for c in COMPONENTS if not c.optional] == ["detector"]
