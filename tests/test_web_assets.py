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
    for script in ("simple.js", "people.js", "places.js"):
        wanted = set(re.findall(r'(?:\$|pq|lq)\("([\w-]+)"\)', _js(script)))
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


def test_drawing_a_place_is_reachable_from_the_app():
    """Zones were only drawable on the advanced page.

    The app's own onboarding promises questions about going in and out, and a
    direction can only be measured against a drawn line - so the one setup
    step that makes the headline feature work lived behind a link the app
    never explained.
    """
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'id="places-toggle"' in html
    assert 'id="place-canvas"' in html, "nothing to draw on"
    assert 'src="/static/places.js"' in html
    assert "initPlaces()" in _js("simple.js"), "the Places button is never wired up"


def test_a_direction_question_with_no_place_drawn_says_so():
    """Otherwise it is an empty screen that looks like an answer.

    "when did he go out" against a library with no line drawn is not "it never
    happened" - it is a question nothing in the index can measure. Those are
    different, and only one of them is fixed by two clicks.
    """
    js = _js("simple.js")
    assert "event_kind" in js, "the app cannot tell a direction question from any other"
    assert re.search(r"wantsDirection && !state\.zones", js)
    assert 'id="draw-a-place"' in (WEB / "app.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("script", ["places.js", "zones.js"])
def test_the_inbound_arrow_points_where_the_server_says_in_is(script: str):
    """Which side counts as "in" is invisible until something draws it.

    This is computed, not matched: the normal is lifted out of the JavaScript
    and run through the server's own side_of_line for lines at four
    orientations. Canvas y grows downward, so the y-up form of a perpendicular
    is the wrong one - and with it the arrow points at the side the server
    calls cross_out. Every doorway drawn by following it would report entries
    as exits, in an app whose headline question is "when did he go out".
    """
    from tsv.zones import side_of_line

    js = _js(script)
    nx_expr = re.search(r"const nx = (-?)dy / len;", js)
    ny_expr = re.search(r"const ny = (-?)dx / len;", js)
    assert nx_expr and ny_expr, f"{script} no longer computes an inbound normal"
    nx_sign = -1.0 if nx_expr.group(1) else 1.0
    ny_sign = -1.0 if ny_expr.group(1) else 1.0

    for a, b in (((0.0, 0.0), (1.0, 0.0)),      # left to right
                 ((1.0, 0.0), (0.0, 0.0)),      # right to left
                 ((0.0, 0.0), (0.0, 1.0)),      # top to bottom
                 ((0.3, 0.2), (0.8, 0.9))):     # diagonal
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        nx, ny = nx_sign * dy / length, ny_sign * dx / length
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        tip = (mid[0] + nx * 0.2, mid[1] + ny * 0.2)
        assert side_of_line(tip, a, b) > 0, (
            f"{script}: the arrow for {a}->{b} points at the cross_out side"
        )


def test_the_library_is_visible_and_its_scope_is_stated():
    """A persistent library is the right design; a silent one is not.

    Searching across days is the point of the tool, so the fix for "it always
    has my old videos" is to show which are in scope and allow removal - not
    to throw the archive away on launch.
    """
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'id="videos-toggle"' in html
    assert 'id="video-list"' in html
    assert 'id="scope"' in html and 'id="scope-all"' in html
    assert "initVideos()" in _js("simple.js")
    assert "/api/videos" in _js("videos.js")


def test_a_search_carries_the_scope_it_is_showing():
    """Otherwise the strip says one thing and the query does another."""
    js = _js("simple.js")
    assert "searchScope" in js
    assert re.search(r"video_id=\$\{scope\}", js)


def test_a_finished_import_scopes_to_what_was_just_added():
    """Somebody who waited for a video to be read wants that video."""
    assert "scopeToNewest" in _js("simple.js")
    assert "scopeToNewest" in _js("videos.js")


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


def test_uploads_are_chunked_and_resumable():
    """A single POST asked a phone to hold one connection for gigabytes.

    Everything a phone does normally - locking, switching apps, walking out of
    range - ended it, and the reward was starting again from zero.
    """
    js = _js("upload.js")
    assert "/api/upload/begin" in js
    assert "X-Upload-Offset" in js, "the server cannot place a chunk"
    assert "file.slice(" in js, "the whole file is still being sent at once"
    assert "UPLOAD_RETRIES" in js, "a dropped connection is not retried"
    assert "initVideos" not in js, "the uploader should not know about panels"

    app = _js("simple.js")
    assert "uploadFiles(" in app, "the app is not using the resumable path"
    assert 'src="/static/upload.js"' in (WEB / "app.html").read_text(encoding="utf-8")


def test_the_client_takes_the_offset_from_the_server():
    """It must never assume where it got to.

    A reply lost after the write landed means the phone believes less arrived
    than really did; trusting its own count would rewrite bytes, and trusting
    it in the other direction would leave a hole.
    """
    js = _js("upload.js")
    assert re.search(r"state\s*=\s*sent", js), "the client keeps its own offset"
    assert "sent.offset <= state.offset" in js, "no guard against a stalled upload"


def test_a_transfer_can_be_stopped_from_the_progress_strip():
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'id="strip-stop"' in html
    js = _js("simple.js")
    assert "state.uploading.abort()" in js
    # Stopping keeps what arrived, and the wording has to say so or somebody
    # assumes the work is gone and starts again.
    assert "carry on from where it stopped" in js


def test_tapping_a_result_fetches_a_clip_not_the_whole_recording():
    """Measured: 0.68 MB against 124 MB, and 0.02 seconds to produce.

    On a desktop the difference is invisible. On a phone over WiFi it is the
    difference between watching something and waiting for it.
    """
    js = _js("simple.js")
    assert "/api/clip/" in js, "the player still asks for the whole file"
    assert "CLIP_SECONDS" in js
    # And an escape hatch, because sometimes what happened next is the point.
    assert "openWholeRecording" in js
    assert 'id="player-whole"' in (WEB / "app.html").read_text(encoding="utf-8")


def test_closing_the_player_stops_it_buffering():
    """A clip left paused behind a closed dialog is somebody's data."""
    js = _js("simple.js")
    assert 'video.removeAttribute("src")' in js


def test_the_phone_layout_puts_search_within_reach():
    """Reaching the top of a six inch screen one-handed is the single most
    awkward thing about the desktop layout on a phone."""
    css = (WEB / "simple.css").read_text(encoding="utf-8")
    phone = css[css.index("---------- on a phone ----------"):]
    assert "position: fixed" in phone and "bottom: 0" in phone
    # Under 16px, iOS zooms the page whenever the field is focused.
    assert "font-size: 16px" in phone
    assert "env(safe-area-inset-bottom)" in phone, "the home indicator will cover it"
    assert "grid-template-columns: 1fr" in phone, "results are still a desktop grid"


def test_sharing_can_be_turned_on_without_a_terminal():
    """Asking somebody to open a command prompt on a computer that is not
    theirs is where this feature stops being used."""
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'id="share-toggle"' in html and 'id="share-panel"' in html
    assert 'id="share-qr-img"' in html, "nobody types an IP into a phone correctly"
    js = _js("share.js")
    assert "/api/share/start" in js and "/api/share/stop" in js
    assert "initShare()" in _js("simple.js")
    # Windows blocks the first bind, and its dialog opens behind the app.
    assert "If Windows asks" in js


def test_the_client_asks_for_h264_only_when_it_needs_it():
    """Only the browser knows what it can decode, so it asks rather than the
    server guessing from a user agent.

    Phones both record and play HEVC, so the common case must not pay for a
    conversion it does not need.
    """
    js = _js("simple.js")
    assert "canPlayHevc" in js
    assert 'codecs="hvc1' in js, "HEVC support is never actually tested for"
    assert "h264=1" in js
    assert "state.canPlayHevc" in js


def test_a_clip_that_will_not_play_is_retried_once_in_h264():
    """canPlayType says "probably" and means "possibly".

    A browser that claims HEVC and then fails on a real file is common enough
    - missing hardware decoding, a codec behind a flag - that the honest
    answer is to find out by trying.
    """
    js = _js("simple.js")
    assert "retryAsH264" in js
    assert 'dataset.retried' in js, "nothing stops it retrying forever"
    assert 'addEventListener("error"' in js, "playback failure is not noticed"


# ---------- knowing which one to double-click ----------

def test_the_icon_is_committed_and_covers_the_small_sizes():
    """16 pixels is where an icon is actually seen - a taskbar button, a file
    listing - and it is the size most icons stop working at."""
    root = WEB.parent
    icon = root / "TextSearchVDO.ico"
    assert icon.is_file(), "run tools/make_icon.py"

    from PIL import Image

    with Image.open(icon) as image:
        sizes = {s[0] for s in image.info.get("sizes", set())}
    assert 16 in sizes and 32 in sizes and 256 in sizes, sizes


def test_the_shortcut_points_at_the_launcher_that_can_report_problems():
    """Straight at pythonw would skip the checks in the .vbs, and a missing
    virtual environment would become a double-click that does nothing."""
    source = (WEB.parent / "tsv" / "shortcut.py").read_text(encoding="utf-8")
    assert "wscript.exe" in source
    assert "TextSearchVDO.vbs" in source
    assert "IconLocation" in source


def test_the_phone_home_screen_gets_a_raster_icon_too():
    """iOS ignores SVG icons and will otherwise use a screenshot of the page."""
    html = (WEB / "app.html").read_text(encoding="utf-8")
    assert 'rel="apple-touch-icon"' in html
    assert (WEB / "icon-512.png").is_file()
