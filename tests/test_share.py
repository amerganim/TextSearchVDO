"""Letting a phone in, and keeping everyone else out.

This is the only part of the project where a bug is a privacy breach rather
than a wrong answer, so the tests are about what an *unpaired* caller can
reach. The rule being defended: with sharing on, a device that has not proved
itself sees the pairing page and nothing else - not the index, not a
thumbnail, not the application's own JavaScript.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from tsv import db
from tsv.api import create_app
from tsv.config import DEFAULT
from tsv.share import (
    COOKIE_NAME, MAX_ATTEMPTS, Pairing, describe_addresses, is_public_path,
    issue_cookie, list_devices, load_key, register_device, revoke_device,
    verify_cookie, _classify,
)


@pytest.fixture
def shared(tmp_path):
    """An app with sharing on, and a client that is not this machine.

    TestClient presents as "testclient" rather than 127.0.0.1, which is what
    makes it useful here: it exercises the stranger path rather than the
    loopback exemption.
    """
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()
    app = create_app(cfg, share=True)
    return app, TestClient(app, follow_redirects=False)


@pytest.fixture
def conn(tmp_path):
    return db.open_db((tmp_path / "index.db"))


# ---------- what a stranger may see ----------

@pytest.mark.parametrize("path", [
    "/", "/advanced", "/api/summary", "/api/videos", "/api/search?q=a",
    "/static/simple.js", "/static/simple.css", "/api/thumb/1", "/api/media/1",
])
def test_an_unpaired_device_is_refused_everywhere(shared, path):
    """Including the static files.

    Worth stating because the obvious implementation - allowing anything
    under /static/ so the pairing page can style itself - would have handed
    the whole application's JavaScript to a device that had proved nothing.
    """
    _, client = shared
    assert client.get(path).status_code in (303, 401)


def test_a_page_is_redirected_and_an_api_call_is_refused(shared):
    """Different callers need different failures.

    A phone whose cookie expired mid-search must not get the pairing form
    rendered inside its results list.
    """
    _, client = shared
    page = client.get("/")
    assert page.status_code == 303 and page.headers["location"] == "/pair"

    api = client.get("/api/summary")
    assert api.status_code == 401
    assert api.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", ["/pair", "/static/pair.css", "/manifest.json"])
def test_the_pairing_page_and_what_it_needs_are_reachable(shared, path):
    _, client = shared
    assert client.get(path).status_code == 200


def test_the_public_list_is_closed_not_a_prefix():
    assert is_public_path("/static/pair.css")
    assert not is_public_path("/static/simple.js")
    assert not is_public_path("/static/../index.db")


# ---------- pairing ----------

def test_the_right_code_pairs_and_the_wrong_one_does_not(shared):
    app, client = shared
    assert client.post("/api/pair", json={"code": "000000"}).status_code == 403
    assert client.get("/api/summary").status_code == 401

    response = client.post(
        "/api/pair", json={"code": app.state.pairing_code(), "name": "My phone"}
    )
    assert response.status_code == 200
    assert client.get("/api/summary").status_code == 200
    assert client.get("/").status_code == 200


def test_a_code_works_once(shared):
    """Overhearing a code should not be worth anything afterwards."""
    app, client = shared
    code = app.state.pairing_code()
    assert client.post("/api/pair", json={"code": code}).status_code == 200

    second = TestClient(app, follow_redirects=False)
    assert second.post("/api/pair", json={"code": code}).status_code == 403


def test_guessing_throws_the_code_away(conn):
    """Six digits is a million, which only holds up if guessing is limited."""
    pairing = Pairing(conn)
    original = pairing.code
    wrong = "000000" if original != "000000" else "111111"

    for _ in range(MAX_ATTEMPTS - 1):
        assert pairing.check(wrong) is False
    assert pairing.code == original, "rotated too early to be usable"

    pairing.check(wrong)
    assert pairing.code != original, "an attacker keeps a fixed target"


def test_a_code_is_accepted_however_it_was_typed(conn):
    pairing = Pairing(conn)
    code = pairing.code
    assert pairing.check(f"{code[:3]} {code[3:]}") is True


# ---------- cookies ----------

def test_a_forged_cookie_is_refused(shared):
    _, client = shared
    client.cookies.set(COOKIE_NAME, "1.9999999999." + "0" * 32)
    assert client.get("/api/summary").status_code == 401


def test_revoking_takes_effect_on_the_next_request(shared):
    """Which is why the cookie names a row rather than carrying its own claim.

    A self-contained token would stay good until it expired, and "revoke"
    would mean "in thirty days".
    """
    app, client = shared
    client.post("/api/pair", json={"code": app.state.pairing_code(), "name": "phone"})
    assert client.get("/api/summary").status_code == 200

    device_id = client.get("/api/devices").json()[0]["id"]
    assert client.delete(f"/api/devices/{device_id}").status_code == 200
    assert client.get("/api/summary").status_code == 401


def test_a_cookie_signed_with_another_key_is_refused(conn, tmp_path):
    device_id = register_device(conn, "phone", "agent", "10.0.0.5")
    real = load_key(tmp_path)
    assert verify_cookie(conn, real, issue_cookie(real, device_id)) == device_id
    assert verify_cookie(conn, b"a different key" * 3,
                         issue_cookie(real, device_id)) is None


def test_a_cookie_for_a_deleted_device_is_refused(conn, tmp_path):
    key = load_key(tmp_path)
    device_id = register_device(conn, "phone", "agent", "10.0.0.5")
    cookie = issue_cookie(key, device_id)
    assert verify_cookie(conn, key, cookie) == device_id

    revoke_device(conn, device_id)
    assert verify_cookie(conn, key, cookie) is None
    assert list_devices(conn) == []


@pytest.mark.parametrize("cookie", ["", "nonsense", "1.2", "a.b.c", "1.notanumber.ff"])
def test_a_malformed_cookie_is_refused_rather_than_raising(conn, tmp_path, cookie):
    assert verify_cookie(conn, load_key(tmp_path), cookie) is None


def test_the_key_survives_a_restart_and_is_not_guessable(tmp_path):
    first = load_key(tmp_path)
    assert len(first) >= 32
    assert load_key(tmp_path) == first, "every phone would be logged out"


# ---------- sharing is off unless asked for ----------

def test_the_desktop_app_has_no_login(tmp_path):
    """Sharing off means no middleware at all.

    Adding a login to an app running on the machine that holds the files
    would protect nothing and cost somebody a password every launch.
    """
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()
    client = TestClient(create_app(cfg), follow_redirects=False)
    assert client.get("/api/summary").status_code == 200
    assert client.get("/").status_code == 200


# ---------- which network ----------

@pytest.mark.parametrize("ip,kind,hint", [
    ("127.0.0.1", "loopback", None),
    ("192.168.1.14", "private", None),
    ("192.168.42.129", "private", "USB tethering (Android)"),
    ("172.20.10.3", "private", "USB tethering (iPhone)"),
    ("192.168.137.1", "private", "this PC as a hotspot"),
    ("169.254.9.9", "link-local", None),
    ("8.8.8.8", "public", None),
])
def test_addresses_are_classified_so_the_user_can_pick_one(ip, kind, hint):
    """"Which of these is the cable" is the question at this point."""
    address = _classify(ip)
    assert address.kind == kind
    if hint:
        assert address.hint == hint


def test_a_public_address_is_warned_about_and_never_offered():
    from tsv.share import Address

    offer, warnings = describe_addresses([
        Address("203.0.113.7", "public", "reachable from the internet"),
        Address("192.168.1.14", "private", "local network"),
    ])
    assert [a.ip for a in offer] == ["192.168.1.14"]
    assert warnings and "public" in warnings[0].lower()


def test_having_nowhere_safe_to_bind_says_what_to_do():
    from tsv.share import Address

    offer, warnings = describe_addresses([Address("127.0.0.1", "loopback", "here")])
    assert offer == []
    assert any("USB tethering" in w for w in warnings)


# ---------- whose network is this ----------
#
# A private address and a trusted network are different questions, and only
# the second is about who else can see the traffic. A laptop on cafe WiFi
# still gets a 192.168.x address.

def test_a_network_windows_calls_public_is_warned_about():
    from tsv.share import Address

    offer, warnings = describe_addresses(
        [Address("192.168.1.14", "private", "local network")], category="public"
    )
    assert [a.ip for a in offer] == ["192.168.1.14"], "still usable, just flagged"
    assert any("Public" in w for w in warnings)
    assert any("USB tethering" in w for w in warnings), "no way out is offered"


def test_a_trusted_network_is_not_nagged_about():
    from tsv.share import Address

    _, warnings = describe_addresses(
        [Address("192.168.1.14", "private", "local network")], category="private"
    )
    assert not any("Public" in w for w in warnings)


def test_a_cable_is_not_judged_by_the_wifi_it_is_not_on():
    """Windows marks every new interface Public, including a USB tether.

    Warning there would send somebody away from the safest option in the
    application - a link with nobody else on it at all.
    """
    from tsv.share import Address

    _, warnings = describe_addresses(
        [Address("192.168.42.129", "private", "USB tethering (Android)")],
        category="public",
    )
    assert not any("Public" in w for w in warnings)


def test_asking_windows_never_raises():
    """Called before anything is bound, on machines this has never seen."""
    from tsv.share import network_category

    assert network_category() in {"", "public", "private"}


# ---------- the failure that looks like success ----------

def test_the_firewall_fix_is_scoped_to_the_local_network():
    """A phone on the same network is the entire use case.

    Opening the port to anything wider would be a rule somebody has to
    remember to remove, and nobody does.
    """
    from tsv.share import firewall_command

    command = firewall_command(8000)
    assert "-LocalPort 8000" in command
    assert "-Direction Inbound" in command
    assert "-RemoteAddress LocalSubnet" in command, "opened wider than needed"


def test_asking_about_the_firewall_never_raises_or_hangs():
    """Called while somebody is waiting for an address to appear.

    An earlier version walked every inbound rule asking each for its ports,
    which on a normal machine's 167 rules took over 25 seconds and timed out
    into "cannot tell" - a wrong answer delivered slowly.
    """
    import time

    from tsv.share import firewall_allows

    started = time.time()
    answer = firewall_allows(8000)
    assert answer in {True, False, None}
    assert time.time() - started < 20, "slow enough to be reported as unknown"


def test_there_is_a_double_click_fix_for_the_firewall():
    """Telling somebody to open an Administrator PowerShell and paste a
    command is where most people stop, and reasonably so."""
    from tsv.share import firewall_fixer

    script = firewall_fixer()
    assert script is not None and script.is_file()

    body = script.read_text(encoding="utf-8")
    assert "RunAs" in body, "it cannot raise the permission prompt"
    assert "net session" in body, "it would loop, re-elevating forever"
    assert "-RemoteAddress LocalSubnet" in body, "opened wider than needed"
    assert "Remove-NetFirewallRule" in body, "no way back out is documented"


def test_the_firewall_rule_covers_every_profile():
    """A phone hotspot, a USB tether and a home router are three different
    profiles to Windows, and it marks new ones Public.

    Naming only Private and Public misses DomainAuthenticated, which is what
    a work laptop gets - and that machine is exactly where somebody would be
    told the feature is broken.
    """
    from tsv.share import firewall_command

    assert "-Profile Any" in firewall_command(8000)


# ---------- one code, however many processes ----------

def test_two_instances_agree_about_the_code(tmp_path):
    """Held in memory the code was per-process.

    Two instances over the same index printed two different codes: the console
    showed one, the phone was talking to the other, and typing either was
    refused with no explanation. Anything that can run twice cannot keep a
    shared secret in a local variable.
    """
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()

    first = create_app(cfg, share=True)
    second = create_app(cfg, share=True)
    assert first.state.pairing_code() == second.state.pairing_code()


def test_a_code_from_one_instance_is_accepted_by_another(tmp_path):
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()

    shown_by = create_app(cfg, share=True)
    talking_to = create_app(cfg, share=True)

    client = TestClient(talking_to, follow_redirects=False)
    response = client.post(
        "/api/pair", json={"code": shown_by.state.pairing_code(), "name": "phone"}
    )
    assert response.status_code == 200


def test_rotating_in_one_instance_is_seen_by_the_other(tmp_path):
    """Otherwise the other keeps offering a code that no longer works."""
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()

    first = create_app(cfg, share=True)
    second = create_app(cfg, share=True)
    before = first.state.pairing_code()

    TestClient(first, follow_redirects=False).post(
        "/api/pair", json={"code": before, "name": "phone"}
    )
    assert second.state.pairing_code() != before
    assert second.state.pairing_code() == first.state.pairing_code()


def test_the_code_survives_a_restart(tmp_path):
    """A phone part-way through typing should not be defeated by the app
    being reopened."""
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    db.open_db(cfg.db_path).close()

    code = create_app(cfg, share=True).state.pairing_code()
    assert create_app(cfg, share=True).state.pairing_code() == code


def test_a_port_already_in_use_is_detected():
    """A second `tsv share` cannot get the socket, but uvicorn only logs that
    and the process lingers - printing a code nothing is checking."""
    import socket as _socket

    from tsv.share import port_in_use

    with _socket.socket() as held:
        held.bind(("0.0.0.0", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert port_in_use(taken) is True

    # Freed again once nothing holds it.
    assert port_in_use(taken) is False


def test_the_firewall_hint_is_offered_as_a_hint():
    """Windows permits traffic in ways a port-rule query cannot see, and a
    phone here reached the pairing page with no matching rule at all."""
    import inspect

    from tsv import share

    doc = inspect.getdoc(share.firewall_allows) or ""
    assert "hint" in doc.lower()
    assert "not a verdict" in doc.lower()
