"""Letting a phone in, without letting the neighbours in.

The app has always been a server; the desktop window is a WebView pointed at
it. So reaching it from a phone needs no new architecture - only for it to
stop binding to loopback. That one change is also what makes this dangerous,
because there has never been any authentication: bind to the LAN as the code
stood and every device on the WiFi could read the footage, the names of
enrolled people, and every transcript. For an app whose whole claim is that
nothing leaves your machine, that is the part that matters.

**How it works.** Sharing is off unless asked for. When it is on, a phone that
has never been seen gets one page - a pairing form - and nothing else. It
enters a six-digit code shown on the PC, and gets back a signed cookie naming
a row in `devices`. Every other route requires that cookie, and deleting the
row revokes it immediately.

**Loopback is exempt.** A request from 127.0.0.1 is somebody already sitting
at the machine, who can open the database with a text editor. Making the
desktop window authenticate against itself would add a login to an app that
does not need one, and protect nothing.

**What this does not do.** There is no TLS. On a home network that is a
reasonable trade - the alternative is asking people to trust a self-signed
certificate on a phone, which is unpleasant enough that most would give up -
but it should be said plainly rather than implied: somebody already on your
WiFi, running the right tools, can see the traffic. Your network is the
security boundary. That is why `describe_addresses` refuses to be cheerful
about addresses that are not private.
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
import socket
import sqlite3
import time
from dataclasses import dataclass
from hashlib import blake2b, sha256
from pathlib import Path

# How long a paired phone stays paired without doing anything.
COOKIE_NAME = "tsv_device"
COOKIE_DAYS = 30

# Six digits is a million codes, which is nothing against an offline attack
# and plenty against an online one - provided guessing is actually limited.
# It is: this many failures and the code is thrown away and a new one drawn,
# so an attacker never gets to work against a fixed target.
CODE_DIGITS = 6
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Address:
    """One address this machine can be reached on, and what it looks like."""

    ip: str
    kind: str          # "loopback" | "private" | "link-local" | "public"
    hint: str          # what it probably is, in words

    @property
    def safe(self) -> bool:
        return self.kind in ("loopback", "private")


def _classify(ip: str) -> Address:
    address = ipaddress.ip_address(ip)
    if address.is_loopback:
        return Address(ip, "loopback", "this machine only")
    if address.is_link_local:
        # 169.254.x means DHCP never answered. Usually a cable with nothing
        # at the other end, occasionally a direct link that happens to work.
        return Address(ip, "link-local", "no network assigned - probably not reachable")
    if address.is_private:
        # Android hands the PC 192.168.42.x when tethering, iOS 172.20.10.x.
        # Naming them is worth it: "which of these is the cable" is exactly
        # the question somebody has at this point.
        if ip.startswith("192.168.42."):
            return Address(ip, "private", "USB tethering (Android)")
        if ip.startswith("172.20.10."):
            return Address(ip, "private", "USB tethering (iPhone)")
        if ip.startswith("192.168.137."):
            return Address(ip, "private", "this PC as a hotspot")
        return Address(ip, "private", "local network")
    return Address(ip, "public", "reachable from the internet - do not share on this")


def local_addresses() -> list[Address]:
    """Every address this machine appears to answer on.

    Two sources, because neither is complete on its own: the hostname lookup
    misses interfaces that came up after boot, and the outbound-route trick
    only ever names one. Together they cover the case that matters here - a
    laptop on WiFi that has just had a phone plugged into it.
    """
    found: set[str] = set()
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass

    for probe in ("8.8.8.8", "192.168.1.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.2)
                sock.connect((probe, 80))
                found.add(sock.getsockname()[0])
        except OSError:
            continue

    addresses = []
    for ip in sorted(found):
        try:
            addresses.append(_classify(ip))
        except ValueError:
            continue
    # Most useful first: something a phone can reach, before loopback.
    order = {"private": 0, "public": 1, "link-local": 2, "loopback": 3}
    return sorted(addresses, key=lambda a: (order.get(a.kind, 9), a.ip))


def describe_addresses(addresses: list[Address]) -> tuple[list[Address], list[str]]:
    """(what to offer, what to warn about)."""
    offer = [a for a in addresses if a.kind == "private"]
    warnings = []
    if any(a.kind == "public" for a in addresses):
        warnings.append(
            "This machine has a public address. Sharing binds to every "
            "interface, so do not do this on a network you do not control."
        )
    if not offer:
        warnings.append(
            "No private network address found. Join a WiFi network, or plug "
            "the phone in and turn on USB tethering."
        )
    return offer, warnings


# ---------- the signing key ----------

def load_key(data_dir: Path) -> bytes:
    """The key that signs device cookies, created once and kept.

    Deleting it logs every phone out, which is the blunt instrument for "I do
    not know who has access any more".
    """
    path = data_dir / "share_key"
    if path.is_file():
        key = path.read_bytes()
        if len(key) >= 32:
            return key

    key = secrets.token_bytes(32)
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass       # Windows, where this is not how permissions work
    return key


# ---------- pairing codes ----------

class Pairing:
    """The code shown on the PC, and the guessing limit around it."""

    def __init__(self) -> None:
        self.code = ""
        self.attempts = 0
        self.rotate()

    def rotate(self) -> str:
        # secrets, not random: this is a credential for the rest of the index.
        self.code = f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"
        self.attempts = 0
        return self.code

    def check(self, offered: str) -> bool:
        """Compare in constant time, and rotate once guessing starts."""
        offered = (offered or "").strip().replace(" ", "").replace("-", "")
        ok = hmac.compare_digest(offered, self.code)
        if ok:
            # A code is good for one device. The next phone needs a new one,
            # so a code overheard once does not stay useful.
            self.rotate()
            return True

        self.attempts += 1
        if self.attempts >= MAX_ATTEMPTS:
            self.rotate()
        return False


# ---------- devices ----------

def register_device(
    conn: sqlite3.Connection, name: str, user_agent: str, address: str
) -> int:
    now = time.time()
    cur = conn.execute(
        """INSERT INTO devices(name, user_agent, address, paired_at, last_seen)
           VALUES (?,?,?,?,?)""",
        (name.strip()[:64] or "a phone", user_agent[:200], address, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_devices(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT id, name, user_agent, address, paired_at, last_seen "
            "FROM devices ORDER BY last_seen DESC"
        )
    ]


def revoke_device(conn: sqlite3.Connection, device_id: int) -> bool:
    cur = conn.execute("DELETE FROM devices WHERE id = ?", (device_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------- cookies ----------

def issue_cookie(key: bytes, device_id: int) -> str:
    """A cookie naming one device row, signed.

    The row id is inside it on purpose: revoking is a DELETE, and the next
    request from that phone finds nothing to match. A self-contained token
    would stay valid until it expired, which is not what "revoke" means.
    """
    issued = int(time.time())
    payload = f"{device_id}.{issued}"
    return f"{payload}.{_sign(key, payload)}"


def _sign(key: bytes, payload: str) -> str:
    return blake2b(payload.encode(), key=key, digest_size=16).hexdigest()


def verify_cookie(
    conn: sqlite3.Connection, key: bytes, cookie: str | None
) -> int | None:
    """The device id this cookie names, or None.

    Checked in this order on purpose: signature first, so a forged cookie
    never reaches the database; then age; then existence, which is what makes
    revocation immediate.
    """
    if not cookie:
        return None
    parts = cookie.split(".")
    if len(parts) != 3:
        return None

    device_id, issued, signature = parts
    if not hmac.compare_digest(_sign(key, f"{device_id}.{issued}"), signature):
        return None
    try:
        if time.time() - int(issued) > COOKIE_DAYS * 86400:
            return None
        device_id_int = int(device_id)
    except ValueError:
        return None

    row = conn.execute(
        "SELECT id FROM devices WHERE id = ?", (device_id_int,)
    ).fetchone()
    if row is None:
        return None

    _touch(conn, device_id_int)
    return device_id_int


# When each device's last_seen was last written. Kept in memory because the
# alternative - writing on every request - is what this avoids.
_TOUCHED: dict[int, float] = {}
TOUCH_EVERY = 60.0


def _touch(conn: sqlite3.Connection, device_id: int) -> None:
    """Record that a device is still in use, sparingly.

    This used to run on every authenticated request, which meant a write per
    thumbnail: a phone scrolling a results grid fired dozens of UPDATEs a
    second at a WAL database being read by other threads, and it deadlocked -
    "database is locked" in the middle of a search. Once a minute is as much
    resolution as "last seen" has ever needed.

    Best effort by design. This is bookkeeping; a request must never fail
    because the bookkeeping could not be written.
    """
    now = time.time()
    if now - _TOUCHED.get(device_id, 0.0) < TOUCH_EVERY:
        return
    _TOUCHED[device_id] = now
    try:
        conn.execute(
            "UPDATE devices SET last_seen = ? WHERE id = ?", (now, device_id)
        )
        conn.commit()
    except sqlite3.Error:
        pass


# ---------- what needs no cookie ----------

# The pairing page and what it needs to render, and nothing else. Kept as a
# closed list rather than a prefix match, because "/static/" as a prefix would
# have quietly served the whole app's JavaScript to an unpaired phone.
PUBLIC_PATHS = frozenset({
    "/pair",
    "/api/pair",
    "/static/pair.css",
    "/favicon.ico",
    "/manifest.json",
})


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS


def fingerprint(user_agent: str, address: str) -> str:
    """A short, non-identifying label for an unpaired caller, for the log."""
    return sha256(f"{user_agent}|{address}".encode()).hexdigest()[:8]
