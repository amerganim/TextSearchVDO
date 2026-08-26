/* Zone drawing.
 *
 * Points are stored normalised 0..1 against the frame, never in pixels, so a
 * zone survives the camera being reconfigured or the panel being resized.
 * Everything drawn here converts through the canvas box at paint time.
 */

const zoneState = {
  open: false,
  cameraId: null,
  points: [],
  zones: [],
};

const zq = (id) => document.getElementById(id);

function zoneKind() {
  return zq("zone-kind").value;
}

function requiredPoints() {
  return zoneKind() === "line" ? 2 : 3;
}

/* ---------- painting ---------- */

function fitCanvas() {
  const canvas = zq("zone-canvas");
  const img = zq("zone-frame");
  const rect = img.getBoundingClientRect();
  if (rect.width < 2) return false;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}

function drawZones() {
  const canvas = zq("zone-canvas");
  if (!fitCanvas()) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);

  const toPx = (p) => [p[0] * w, p[1] * h];

  // Saved zones, quiet.
  for (const zone of zoneState.zones) {
    const pts = zone.points.map(toPx);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#4da3ff";
    ctx.fillStyle = "rgba(77,163,255,0.13)";
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
    if (zone.kind === "region") {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();

    if (zone.kind === "line") drawDirection(ctx, pts[0], pts[1], "#4da3ff");

    ctx.fillStyle = "#cfe0f5";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(zone.name, pts[0][0] + 6, pts[0][1] - 6);
  }

  // The zone being drawn, loud.
  const pts = zoneState.points.map(toPx);
  if (!pts.length) return;

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#ffb020";
  ctx.fillStyle = "rgba(255,176,32,0.16)";
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
  if (zoneKind() === "region" && pts.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  if (zoneKind() === "line" && pts.length === 2) {
    drawDirection(ctx, pts[0], pts[1], "#ffb020");
  }

  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffb020";
    ctx.fill();
  }
}

/** An arrow at the midpoint showing which way counts as "in". */
function drawDirection(ctx, a, b, colour) {
  const mx = (a[0] + b[0]) / 2;
  const my = (a[1] + b[1]) / 2;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  // Left of a->b is inbound, matching side_of_line() on the server.
  const nx = dy / len;
  const ny = -dx / len;

  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mx, my);
  ctx.lineTo(mx + nx * 22, my + ny * 22);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(mx + nx * 22, my + ny * 22, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("in", mx + nx * 30, my + ny * 30 + 4);
}

/* ---------- state ---------- */

function updateControls() {
  const enough = zoneState.points.length >= requiredPoints();
  const exact = zoneKind() === "line" ? zoneState.points.length === 2 : enough;
  zq("zone-save").disabled = !(exact && zq("zone-name").value.trim());
  zq("zone-undo").disabled = zoneState.points.length === 0;
  zq("zone-clear").disabled = zoneState.points.length === 0;

  const hint = zq("zone-hint");
  if (!zoneState.cameraId) {
    hint.textContent = "Pick a single camera to draw on.";
  } else if (zoneKind() === "line") {
    hint.textContent = zoneState.points.length < 2
      ? `Click ${2 - zoneState.points.length} more point${zoneState.points.length === 1 ? "" : "s"} to place the line.`
      : "Line placed. The arrow shows which side counts as coming in — clear and redraw the other way round to flip it.";
  } else {
    hint.textContent = zoneState.points.length < 3
      ? `Click ${3 - zoneState.points.length} more point${zoneState.points.length === 2 ? "" : "s"} to close the region.`
      : `${zoneState.points.length} points. Keep clicking to refine, then save.`;
  }
}

function setStatus(text, isError = false) {
  const el = zq("zone-status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

/* ---------- loading ---------- */

async function loadZones() {
  if (!zoneState.cameraId) {
    zoneState.zones = [];
    zq("zone-list").innerHTML = '<div class="note">Pick a single camera.</div>';
    drawZones();
    return;
  }
  zoneState.zones = await fetch(`/api/zones?camera_id=${zoneState.cameraId}`).then((r) => r.json());

  const list = zq("zone-list");
  if (!zoneState.zones.length) {
    list.innerHTML = '<div class="note">None yet.</div>';
  } else {
    const counts = await fetch(`/api/events?camera_id=${zoneState.cameraId}&limit=2000`)
      .then((r) => r.json())
      .then((events) => {
        const by = {};
        for (const e of events) by[e.zone_id] = (by[e.zone_id] || 0) + 1;
        return by;
      });

    list.innerHTML = zoneState.zones
      .map(
        (z) => `<div class="zone-row">
          <span class="name">${z.name}</span>
          <span class="kind">${z.kind}</span>
          <span class="count">${counts[z.id] || 0}</span>
          <button type="button" data-id="${z.id}" title="Delete zone">&times;</button>
        </div>`
      )
      .join("");

    list.querySelectorAll("button").forEach((el) => {
      el.onclick = async () => {
        await fetch(`/api/zones/${el.dataset.id}`, { method: "DELETE" });
        setStatus("Zone deleted.");
        await loadZones();
        await loadEvents();
      };
    });
  }
  drawZones();
}

async function loadEvents() {
  const list = zq("event-list");
  if (!zoneState.cameraId) {
    list.innerHTML = "";
    return;
  }
  const events = await fetch(`/api/events?camera_id=${zoneState.cameraId}&limit=40`)
    .then((r) => r.json());

  if (!events.length) {
    list.innerHTML = '<div class="note">No events yet. Draw a zone, or run <b>analyze</b> first.</div>';
    return;
  }
  list.innerHTML = events
    .map((e) => {
      const when = new Date(e.ts * 1000).toLocaleTimeString();
      const extra = e.duration ? ` ${Math.round(e.duration)}s` : "";
      return `<div class="event-row" data-video="${e.video_id}" data-t="${e.t}">
        <span class="when">${when}</span>
        <span class="name">${e.label}</span>
        <span>${e.zone_name}</span>
        <span class="kind">${e.kind.replace("_", " ")}${extra}</span>
      </div>`;
    })
    .join("");

  list.querySelectorAll(".event-row").forEach((el) => {
    el.onclick = () => {
      const video = document.getElementById("video");
      const src = `/api/media/${el.dataset.video}`;
      const seek = () => {
        video.currentTime = Math.max(0, Number(el.dataset.t) - 2);
        video.play().catch(() => {});
      };
      if (video.dataset.videoId !== el.dataset.video) {
        video.dataset.videoId = el.dataset.video;
        video.src = src;
        video.addEventListener("loadedmetadata", seek, { once: true });
      } else {
        seek();
      }
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    };
  });
}

async function refreshZonePanel(cameraId) {
  zoneState.cameraId = cameraId || null;
  zoneState.points = [];

  const frame = zq("zone-frame");
  if (zoneState.cameraId) {
    frame.src = `/api/frame/${zoneState.cameraId}`;
    frame.hidden = false;
  } else {
    frame.removeAttribute("src");
  }
  await loadZones();
  await loadEvents();
  updateControls();
}

/* ---------- wiring ---------- */

function initZones(getCameraId) {
  const panel = zq("zones-panel");
  const toggle = zq("zones-toggle");

  toggle.onclick = async () => {
    zoneState.open = !zoneState.open;
    panel.hidden = !zoneState.open;
    toggle.setAttribute("aria-expanded", String(zoneState.open));
    if (zoneState.open) await refreshZonePanel(getCameraId());
  };

  zq("zone-canvas").onclick = (e) => {
    if (!zoneState.cameraId) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // A line takes exactly two points; further clicks replace the second
    // rather than silently doing nothing.
    if (zoneKind() === "line" && zoneState.points.length >= 2) {
      zoneState.points = [zoneState.points[0], [x, y]];
    } else {
      zoneState.points.push([x, y]);
    }
    drawZones();
    updateControls();
  };

  zq("zone-undo").onclick = () => {
    zoneState.points.pop();
    drawZones();
    updateControls();
  };

  zq("zone-clear").onclick = () => {
    zoneState.points = [];
    drawZones();
    updateControls();
    setStatus("");
  };

  zq("zone-kind").onchange = () => {
    zoneState.points = [];
    drawZones();
    updateControls();
  };

  zq("zone-name").oninput = updateControls;

  zq("zone-save").onclick = async () => {
    const body = {
      camera_id: Number(zoneState.cameraId),
      name: zq("zone-name").value.trim(),
      kind: zoneKind(),
      points: zoneState.points,
    };
    const response = await fetch("/api/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      setStatus(detail.detail || `Could not save (${response.status}).`, true);
      return;
    }
    const saved = await response.json();
    setStatus(`Saved "${saved.name}" — ${saved.n_events} events found.`);
    zoneState.points = [];
    zq("zone-name").value = "";
    await loadZones();
    await loadEvents();
    updateControls();
  };

  zq("zone-frame").onload = drawZones;
  new ResizeObserver(drawZones).observe(zq("zone-stage"));
}
