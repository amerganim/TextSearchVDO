"""Static checks on the pages themselves.

A browser is not available in the suite, so these assert the conditions that
made a real bug possible rather than the rendering. The bug: a full-screen
`position: fixed` overlay carrying `display: flex` ignored its own `hidden`
attribute and sat above the entire app, swallowing every click.

`[hidden]` is implemented by a user-agent rule, which loses to *any* author
declaration of `display`. Every element the app hides this way therefore needs
the guard, and it is worth failing the build over.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
STYLESHEETS = ("simple.css", "style.css")
PAGES = ("app.html", "index.html")

GUARD = re.compile(r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", re.IGNORECASE)


@pytest.mark.parametrize("name", STYLESHEETS)
def test_stylesheet_makes_the_hidden_attribute_win(name: str):
    css = (WEB / name).read_text(encoding="utf-8")
    assert GUARD.search(css), (
        f"{name} has no '[hidden] {{ display: none !important }}'. Without it any "
        f"element with an author display rule ignores the hidden attribute."
    )


@pytest.mark.parametrize("page", PAGES)
def test_every_hidden_element_is_covered_by_the_guard(page: str):
    """Named so the failure explains itself if the guard is ever removed."""
    html = (WEB / page).read_text(encoding="utf-8")
    hidden_ids = re.findall(r'id="([\w-]+)"[^>]*\shidden', html)
    if not hidden_ids:
        pytest.skip(f"{page} hides nothing")

    covered = any(
        GUARD.search((WEB / sheet).read_text(encoding="utf-8")) for sheet in STYLESHEETS
    )
    assert covered, f"{page} hides {hidden_ids} but no stylesheet enforces it"


def test_the_overlay_that_caused_the_bug_is_still_hidden_by_default():
    """The player must start closed. It covers the whole window when open."""
    html = (WEB / "app.html").read_text(encoding="utf-8")
    match = re.search(r'<div class="player-backdrop"[^>]*>', html)
    assert match, "the player backdrop is gone"
    assert "hidden" in match.group(0), "the player backdrop no longer starts hidden"


def test_the_overlay_still_declares_a_display_mode():
    """If this ever stops being flex the guard is still correct, but the note
    in the stylesheet would be misleading - fail loudly rather than rot."""
    css = (WEB / "simple.css").read_text(encoding="utf-8")
    block = re.search(r"\.player-backdrop\s*\{([^}]*)\}", css)
    assert block and "display" in block.group(1)


def test_pages_reference_only_assets_that_exist():
    for page in PAGES:
        html = (WEB / page).read_text(encoding="utf-8")
        for asset in re.findall(r'/static/([\w.-]+)', html):
            assert (WEB / asset).is_file(), f"{page} references missing {asset}"
