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
import subprocess
import sys
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


def network_category() -> str:
    """What Windows thinks of the network this machine is on.

    "private", "public", or "" when it cannot be told. Worth asking, because
    a private *address* and a trusted *network* are different questions and
    only the second one is about who else is on it. A laptop on café WiFi
    still gets a 192.168.x address; what makes that different from a home
    network is a judgement Windows has already made and this had ignored.

    Best effort: one PowerShell call, a second at most, and a machine that
    will not answer is treated as unknown rather than as safe.
    """
    if sys.platform != "win32":
        return ""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-NetConnectionProfile | Select-Object -First 1"
             " -ExpandProperty NetworkCategory)"],
            capture_output=True, text=True, timeout=8, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    answer = (result.stdout or "").strip().lower()
    if "public" in answer:
        return "public"
    if "private" in answer or "domain" in answer:
        return "private"
    return ""


def firewall_allows(port: int) -> bool | None:
    """Whether Windows would let a phone reach this port.

    True when a rule plainly allows it, False when no such rule was found, and
    None when it cannot be told.

    False is a *hint*, not a verdict, and the wording around it has to say so.
    Windows has more ways to permit traffic than this query sees - rules
    scoped to a program rather than a port, group policy, an allowance made
    once from a popup - and a phone here reached the pairing page with no
    matching rule at all. Reported as certainty it sends somebody to fix a
    firewall that was never the problem.

    Testing it properly is not possible from here: a connection from this
    machine to its own address never touches the firewall.
    """
    if sys.platform != "win32":
        return None
    # Port filters first, then the rules behind them. The obvious way round -
    # walk every inbound rule and ask each for its ports - calls a slow cmdlet
    # once per rule, which on this machine's 167 rules took over 25 seconds
    # and timed out into "cannot tell". This way is one query and about two.
    #
    # Deliberately free of nested quotes: it is passed through -Command as a
    # single argument, and escaping is how the first version came back empty.
    script = (
        "$m = Get-NetFirewallPortFilter -ErrorAction SilentlyContinue | "
        "Where-Object {{ $_.LocalPort -contains '{port}' }} | "
        "Get-NetFirewallRule -ErrorAction SilentlyContinue | "
        "Where-Object {{ $_.Direction -eq 'Inbound' -and $_.Enabled -eq 'True' "
        "-and $_.Action -eq 'Allow' }}; "
        "if ($m) {{ Write-Output yes }} else {{ Write-Output no }}"
    ).format(port=port)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=25, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = (result.stdout or "").strip().lower()
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    return None


def port_in_use(port: int) -> bool:
    """Whether something is already listening here.

    Worth asking before binding. A second `tsv share` on a taken port cannot
    get the socket, but uvicorn logs that and the process lingers - so two
    instances sat there, one serving the phone and one printing to the
    console, and the code on screen was not the code being checked.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def firewall_command(port: int) -> str:
    """What to run, as administrator, to let a phone in.

    Scoped to the local subnet rather than opened to the world: this only
    ever needs to serve a device on the same network, and a rule that says so
    is one nobody has to remember to remove.
    """
    return (
        'New-NetFirewallRule -DisplayName "TextSearchVDO" -Direction Inbound '
        f"-Protocol TCP -LocalPort {port} -Action Allow "
        "-Profile Any -RemoteAddress LocalSubnet"
    )


def firewall_fixer() -> Path | None:
    """The double-clickable version of `firewall_command`, if it is there.

    Telling somebody to open an Administrator PowerShell and paste a command
    is where most people stop, and they are right to - it is a lot of
    ceremony to look at a video. The script asks Windows for the rights
    itself, so the whole job is a double-click and one prompt.
    """
    script = Path(__file__).resolve().parent.parent / "allow-phone.bat"
    return script if script.is_file() else None


def describe_addresses(
    addresses: list[Address], category: str | None = None
) -> tuple[list[Address], list[str]]:
    """(what to offer, what to warn about)."""
    offer = [a for a in addresses if a.kind == "private"]
    warnings = []

    if any(a.kind == "public" for a in addresses):
        warnings.append(
            "This machine has a public address. Sharing binds to every "
            "interface, so do not do this on a network you do not control."
        )

    # A point-to-point link has nobody else on it, so Windows calling it
    # public says nothing useful - it says that about every new interface.
    tethered = any(
        a.hint.startswith("USB tethering") or "hotspot" in a.hint for a in offer
    )
    if category is None:
        category = network_category()
    if category == "public" and offer and not tethered:
        warnings.append(
            "Windows has this network marked Public, which is what it calls "
            "one you have not said you trust. Pairing still protects the "
            "index, but nothing here is encrypted - so on a network you do "
            "not control, plug the phone in and use USB tethering instead."
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

def _setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


class Pairing:
    """The code shown on the PC, and the guessing limit around it.

    Kept in the database rather than in memory, and that is the whole point.
    Held per-process, two instances of the application had two different
    codes: the console printed one, the phone was talking to the other, and
    typing either was refused with no explanation. Anything that can be
    running twice cannot hold a shared secret in a local variable.

    Reading on every access rather than caching, for the same reason - a
    cached copy is a second source of truth and goes stale the moment the
    other process rotates the code.
    """

    CODE_KEY = "pairing_code"
    ATTEMPTS_KEY = "pairing_attempts"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        if not _setting(conn, self.CODE_KEY):
            self.rotate()

    @property
    def code(self) -> str:
        return _setting(self._conn, self.CODE_KEY) or self.rotate()

    @property
    def attempts(self) -> int:
        try:
            return int(_setting(self._conn, self.ATTEMPTS_KEY) or 0)
        except ValueError:
            return 0

    def rotate(self) -> str:
        # secrets, not random: this is a credential for the rest of the index.
        code = f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"
        _set_setting(self._conn, self.CODE_KEY, code)
        _set_setting(self._conn, self.ATTEMPTS_KEY, "0")
        return code

    def check(self, offered: str) -> bool:
        """Compare in constant time, and rotate once guessing starts."""
        offered = (offered or "").strip().replace(" ", "").replace("-", "")
        if hmac.compare_digest(offered, self.code):
            # A code is good for one device. The next phone needs a new one,
            # so a code overheard once does not stay useful.
            self.rotate()
            return True

        attempts = self.attempts + 1
        if attempts >= MAX_ATTEMPTS:
            self.rotate()
        else:
            _set_setting(self._conn, self.ATTEMPTS_KEY, str(attempts))
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
