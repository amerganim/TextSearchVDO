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


# ---------- the app is wired to its own markup ----------


def _js(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_every_element_the_scripts_reach_for_exists_in_the_page():
    """A typo'd id fails silently at runtime.

    `document.getElementById` returns null and the next property access throws
    inside an event handler, where nothing surfaces it - the button simply
    does nothing when clicked.
    """
    html = (WEB / "app.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([\w-]+)"', html))
    for script in ("simple.js", "people.js"):
        wanted = set(re.findall(r'(?:\$|pq)\("([\w-]+)"\)', _js(script)))
        missing = wanted - ids
        assert not missing, f"{script} looks for {sorted(missing)}, absent from app.html"


def test_the_naming_panel_is_reachable_from_the_app():
    """Enrolment existed as an API and in the advanced page only.

    The product's headline question - when did a named person go out - was
    therefore unanswerable through the app somebody actually opens.
    """
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'id="people-toggle"' in html
    assert 'id="people-panel"' in html
    assert 'src="/static/people.js"' in html
    assert "initPeople()" in _js("simple.js"), "the People button is never wired up"


def test_naming_teaches_the_gallery_rather_than_labelling_one_sighting():
    """Enrolling alone names the sighting in front of you.

    Matching is what finds the same person elsewhere, and it is the whole
    reason to type a name at all, so the app must run it straight after.
    """
    js = _js("people.js")
    assert "/api/identities/enroll" in js
    assert "/api/identities/assign" in js
    enroll_at = js.index("/api/identities/enroll")
    assert js.index("/api/identities/assign") > enroll_at


def test_an_exact_answer_only_wins_when_it_used_the_whole_question():
    """Otherwise the app throws away its own better result.

    "a person carrying something" grounds "a person" and leaves the rest to
    similarity. Rendering the exact answer regardless listed every person in
    the recording, while the ranked search sitting in the same response had
    already found the one holding a bag.
    """
    js = _js("simple.js")
    assert "semantic_text" in js, "the app ignores what the parse could not ground"
    assert re.search(r"answered\s*=\s*body\.answer && body\.answer\.found && !leftover", js)


def test_the_advanced_page_has_a_way_back():
    """It is reached by a link and had none returning.

    In a browser the back button covers for that. The desktop window has no
    back button at all, so somebody who opened Advanced was stuck there until
    they restarted the app.
    """
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a[^>]+href="/"', html), "no link back to the simple app"


def test_searching_finishes_any_descriptions_still_outstanding():
    """An undescribed sighting is a search that fails without saying why.

    "carrying a bag" matches nothing, and that empty screen looks exactly like
    the moment genuinely not being in the video.
    """
    js = _js("simple.js")
    assert "catchUpOnCaptions" in js
    assert re.search(r"catchUpOnCaptions\(\);", js), "it is defined but never called"
    assert "state.captionStarted" in js, "nothing stops it starting on every keystroke"


# ---------- docs ----------

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_the_architecture_page_exists_and_has_diagrams():
    page = DOCS / "HOW-IT-WORKS.md"
    assert page.is_file()
    blocks = re.findall(r"```mermaid\r?\n([\s\S]*?)```", page.read_text(encoding="utf-8"))
    assert len(blocks) >= 4, "the walkthrough lost its diagrams"


def test_diagram_labels_avoid_html_emphasis():
    """GitHub renders mermaid with HTML labels off.

    <b> and <i> inside a node come out as literal text there - the diagram
    still draws, so this is invisible until someone opens the page on GitHub.
    <br/> is handled by mermaid itself and is fine.
    """
    page = (DOCS / "HOW-IT-WORKS.md").read_text(encoding="utf-8")
    for block in re.findall(r"```mermaid\r?\n([\s\S]*?)```", page):
        offenders = re.findall(r"</?(?:b|i|em|strong|span|div)>", block)
        assert not offenders, f"HTML emphasis in a mermaid label: {offenders[:4]}"


def test_diagrams_declare_a_known_type():
    page = (DOCS / "HOW-IT-WORKS.md").read_text(encoding="utf-8")
    for block in re.findall(r"```mermaid\r?\n([\s\S]*?)```", page):
        first = block.strip().split()[0]
        assert first in {"flowchart", "graph", "erDiagram", "sequenceDiagram", "stateDiagram-v2"}, first


def test_the_page_states_the_model_set_it_actually_ships():
    """The models named in the walkthrough must be the ones the code loads."""
    page = (DOCS / "HOW-IT-WORKS.md").read_text(encoding="utf-8")
    from tsv.config import DEFAULT

    for filename in (
        DEFAULT.detect.model_file,
        DEFAULT.face.detector_file,
        DEFAULT.face.embedder_file,
        DEFAULT.clip.image_file,
        DEFAULT.clip.text_file,
    ):
        assert filename in page, f"{filename} is loaded by the app but not documented"
