/* Turning on phone access, from the app rather than a terminal.
 *
 * `tsv share` already did all of this, and asking somebody to open a command
 * prompt on a computer that is not theirs is where the feature stops being
 * used. Everything needed is here instead: one button, a QR code to point a
 * camera at, the code to type, and the list of phones already let in.
 *
 * These endpoints answer only to this machine. The addresses and the current
 * pairing code are precisely what somebody would need to get in, so a paired
 * phone asking this panel for them is refused - which is why the button is
 * hidden when the page is not being viewed on the computer itself.
 */

const shareState = { open: false, on: false };

const sq = (id) => document.getElementById(id);

function shareEscape(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function renderShare(status) {
  shareState.on = Boolean(status.on);
  sq("share-off").hidden = shareState.on;
  sq("share-on").hidden = !shareState.on;

  const warnings = sq("share-warnings");
  warnings.innerHTML = (status.warnings || []).map(shareEscape).join("<br>");
  warnings.hidden = !(status.warnings || []).length;

  if (!shareState.on) return;

  const addresses = status.addresses || [];
  sq("share-urls").innerHTML = addresses
    .map((a) => `<div class="share-url"><b>${shareEscape(a.url)}</b>`
      + `<span>${shareEscape(a.hint)}</span></div>`)
    .join("") || '<div class="share-url"><span>No address found.</span></div>';

  // Spaced, because it is read off one screen and typed into another.
  const code = status.code || "";
  sq("share-code").textContent = code ? `${code.slice(0, 3)} ${code.slice(3)}` : "";

  if (addresses.length) {
    sq("share-qr-img").src =
      `/api/share/qr?url=${encodeURIComponent(addresses[0].url)}`;
    sq("share-qr-img").hidden = false;
  } else {
    sq("share-qr-img").hidden = true;
  }

  const devices = status.devices || [];
  sq("share-devices").innerHTML = devices.length
    ? devices.map((d) => {
        const when = new Date((d.last_seen || d.paired_at) * 1000).toLocaleString();
        return `<div class="share-device">
          <span class="name">${shareEscape(d.name)}</span>
          <span class="when">last used ${when}</span>
          <button type="button" class="linkish revoke" data-id="${d.id}">Remove</button>
        </div>`;
      }).join("")
    : '<p class="note">None yet.</p>';

  sq("share-devices").querySelectorAll(".revoke").forEach((el) => {
    el.onclick = async () => {
      await fetch(`/api/devices/${el.dataset.id}`, { method: "DELETE" });
      // Immediate, not on expiry: the cookie names this row, so removing it
      // refuses that phone on its very next request.
      notice("That phone will be asked to pair again.", "good");
      loadShare();
    };
  });
}

async function loadShare() {
  const response = await fetch("/api/share");
  if (!response.ok) {
    // 403 means this page is not being viewed on the computer running it,
    // which is the normal case for a paired phone. Nothing to show.
    sq("share-toggle").hidden = true;
    return null;
  }
  const status = await response.json();
  renderShare(status);
  return status;
}

function initShare() {
  const panel = sq("share-panel");
  const toggle = sq("share-toggle");

  const setOpen = async (open) => {
    shareState.open = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) await loadShare();
  };

  toggle.onclick = () => setOpen(!shareState.open);
  sq("share-close").onclick = () => setOpen(false);

  sq("share-start").onclick = async () => {
    sq("share-start").disabled = true;
    try {
      const response = await fetch("/api/share/start", { method: "POST" });
      const body = await response.json();
      if (!response.ok) {
        notice(body.detail || "Could not start sharing.");
        return;
      }
      renderShare(body);
      // Windows will not let a new program listen without being asked once,
      // and the dialog appears behind whatever has focus. Saying so beats
      // somebody concluding the feature is broken.
      notice(
        "If Windows asks whether to allow this on your network, say yes "
        + "&mdash; the phone cannot connect until you do.",
        "good",
      );
    } finally {
      sq("share-start").disabled = false;
    }
  };

  sq("share-stop").onclick = async () => {
    await fetch("/api/share/stop", { method: "POST" });
    notice("Stopped. Paired phones stay paired for next time.", "good");
    loadShare();
  };

  // Hidden until we know this is the computer itself.
  loadShare();
}
