/* Drawing the places a question can be about.
 *
 * A doorway is a line: crossing it counts, and which side you came from is
 * the difference between went out and came in. An area is a shape: entering
 * counts, and so does how long you stayed. Nothing else in the index can
 * answer a question about direction, so until one of these exists "when did
 * he go outside" has nothing to measure against.
 *
 * Points are stored normalised 0..1 against the frame, never in pixels, so a
 * place survives the panel being resized or the camera being reconfigured.
 */

const placeState = {
  open: false,
  cameraId: null,
  cameras: [],
  points: [],
  places: [],
};

const lq = (id) => document.getElementById(id);

const placeKind = () =>
  document.querySelector('input[name="place-kind"]:checked').value;

const neededPoints = () => (placeKind() === "line" ? 2 : 3);

function placeEscape(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

/* ---------- painting ---------- */

function fitPlaceCanvas() {
  const canvas = lq("place-canvas");
  const rect = lq("place-frame").getBoundingClientRect();
  if (rect.width < 2) return false;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  canvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  return true;
}

/** An arrow showing which side of a line counts as coming in.
 *
 * The server calls a crossing "in" when side_of_line(end, a, b) is positive.
 * Drawing that side is the only way somebody can tell before they save;
 * otherwise "in" and "out" are a coin toss they discover days later, in a
 * wrong answer about their own front door.
 */
function drawInbound(ctx, a, b, colour) {
  const mx = (a[0] + b[0]) / 2;
  const my = (a[1] + b[1]) / 2;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  // Canvas y grows downward, so the inbound normal is (-dy, dx) - not the
  // (dy, -dx) that the same maths gives on a y-up plane. With the wrong sign
  // the arrow points at the side the server calls cross_out, and every
  // doorway drawn by following it reports entries as exits.
  const nx = -dy / len;
  const ny = dx / len;

  ctx.strokeStyle = colour;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mx, my);
  ctx.lineTo(mx + nx * 24, my + ny * 24);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(mx + nx * 24, my + ny * 24, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("in", mx + nx * 33, my + ny * 33 + 4);
}

function drawPlaces() {
  const canvas = lq("place-canvas");
  if (!fitPlaceCanvas()) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);

  const toPx = (p) => [p[0] * w, p[1] * h];

  // Already saved, quiet.
  for (const place of placeState.places) {
    const pts = place.points.map(toPx);
    if (!pts.length) continue;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#4da3ff";
    ctx.fillStyle = "rgba(77,163,255,0.13)";
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
    if (place.kind === "region") {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
    if (place.kind === "line") drawInbound(ctx, pts[0], pts[1], "#4da3ff");

    ctx.fillStyle = "#cfe0f5";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(place.name, pts[0][0] + 6, pts[0][1] - 6);
  }

  // Being drawn now, loud.
  const pts = placeState.points.map(toPx);
  if (!pts.length) return;

  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#ffb020";
  ctx.fillStyle = "rgba(255,176,32,0.16)";
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
  if (placeKind() === "region" && pts.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();
  if (placeKind() === "line" && pts.length === 2) {
    drawInbound(ctx, pts[0], pts[1], "#ffb020");
  }

  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffb020";
    ctx.fill();
  }
}

/* ---------- what the buttons say ---------- */

function updatePlaceControls() {
  const n = placeState.points.length;
  const enough = placeKind() === "line" ? n === 2 : n >= 3;
  lq("place-save").disabled = !(enough && lq("place-name").value.trim() && placeState.cameraId);
  lq("place-undo").disabled = n === 0;
  lq("place-clear").disabled = n === 0;

  const hint = lq("place-hint");
  if (!placeState.cameraId) {
    hint.textContent = "Add a video first — there is nothing to draw on yet.";
  } else if (placeKind() === "line") {
    hint.textContent = n < 2
      ? `Click ${2 - n} more point${n === 1 ? "" : "s"} across the doorway.`
      : "The arrow marks which side counts as coming in. Start over and draw it "
        + "the other way round to flip that.";
  } else {
    hint.textContent = n < 3
      ? `Click ${3 - n} more point${n === 2 ? "" : "s"} around the area.`
      : `${n} points. Keep clicking to refine the shape, then save.`;
  }
}

function setPlaceStatus(text, isError = false) {
  const el = lq("place-status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

/* ---------- loading ---------- */

async function loadPlaces() {
  const list = lq("place-list");
  if (!placeState.cameraId) {
    placeState.places = [];
    list.innerHTML = '<p class="note">Nothing yet.</p>';
    drawPlaces();
    return;
  }

  placeState.places = await fetch(`/api/zones?camera_id=${placeState.cameraId}`)
    .then((r) => r.json());

  if (!placeState.places.length) {
    list.innerHTML = '<p class="note">None on this camera yet.</p>';
  } else {
    // How many times anything actually crossed or entered. A place that has
    // never fired is usually drawn in the wrong spot, and saying "0" is the
    // only way somebody finds that out before relying on it.
    const counts = await fetch(`/api/events?camera_id=${placeState.cameraId}&limit=2000`)
      .then((r) => r.json())
      .then((events) => {
        const by = {};
        for (const e of events) by[e.zone_id] = (by[e.zone_id] || 0) + 1;
        return by;
      })
      .catch(() => ({}));

    list.innerHTML = placeState.places
      .map((z) => {
        const n = counts[z.id] || 0;
        return `<div class="place-row">
          <span class="name">${placeEscape(z.name)}</span>
          <span class="kind">${z.kind === "line" ? "doorway" : "area"}</span>
          <span class="count" title="Times something crossed or entered">${n}</span>
          <button type="button" class="linkish" data-id="${z.id}">Remove</button>
        </div>`;
      })
      .join("");

    list.querySelectorAll("button").forEach((el) => {
      el.onclick = async () => {
        await fetch(`/api/zones/${el.dataset.id}`, { method: "DELETE" });
        setPlaceStatus("Removed.");
        await loadPlaces();
        if (typeof refreshLibrary === "function") refreshChrome(await refreshLibrary());
      };
    });
  }
  drawPlaces();
}

async function selectCamera(cameraId) {
  placeState.cameraId = cameraId || null;
  placeState.points = [];

  const frame = lq("place-frame");
  if (placeState.cameraId) {
    frame.src = `/api/frame/${placeState.cameraId}`;
    frame.hidden = false;
  } else {
    frame.removeAttribute("src");
    frame.hidden = true;
  }
  await loadPlaces();
  updatePlaceControls();
}

async function loadCameras() {
  placeState.cameras = await fetch("/api/cameras").then((r) => r.json()).catch(() => []);
  const field = lq("place-camera-field");
  const select = lq("place-camera");

  // One camera is the common case and a picker with a single entry is a
  // decision nobody has to make.
  field.hidden = placeState.cameras.length < 2;
  select.innerHTML = placeState.cameras
    .map((c) => `<option value="${c.id}">${placeEscape(c.name)}</option>`)
    .join("");

  const first = placeState.cameras.length ? placeState.cameras[0].id : null;
  select.value = String(first);
  await selectCamera(first);
}

/* ---------- saving ---------- */

async function savePlace() {
  const response = await fetch("/api/zones", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      camera_id: Number(placeState.cameraId),
      name: lq("place-name").value.trim(),
      kind: placeKind(),
      points: placeState.points,
    }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    setPlaceStatus(detail.detail || `Could not save (${response.status}).`, true);
    return;
  }

  // Events are worked out from boxes already in the index, so the count comes
  // back immediately - and it is the honest test of whether the line is in
  // the right place. Zero means redraw it, not wait.
  const saved = await response.json();
  const n = saved.n_events || 0;
  setPlaceStatus(
    n
      ? `Saved "${saved.name}" — ${n} crossing${n === 1 ? "" : "s"} already in your videos.`
      : `Saved "${saved.name}", but nothing has crossed it. If that is wrong, `
        + "remove it and draw it where people actually pass.",
  );

  placeState.points = [];
  lq("place-name").value = "";
  await loadPlaces();
  updatePlaceControls();
  if (typeof refreshLibrary === "function") refreshChrome(await refreshLibrary());
}

/* ---------- wiring ---------- */

function initPlaces() {
  const panel = lq("places-panel");
  const toggle = lq("places-toggle");

  const setOpen = async (open) => {
    placeState.open = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) {
      await loadCameras();
      updatePlaceControls();
    }
  };

  toggle.onclick = () => setOpen(!placeState.open);
  lq("places-close").onclick = () => setOpen(false);

  lq("place-canvas").onclick = (event) => {
    if (!placeState.cameraId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;

    // A doorway takes exactly two points. A third click moves the second one
    // rather than being silently ignored.
    if (placeKind() === "line" && placeState.points.length >= 2) {
      placeState.points = [placeState.points[0], [x, y]];
    } else {
      placeState.points.push([x, y]);
    }
    drawPlaces();
    updatePlaceControls();
  };

  lq("place-undo").onclick = () => {
    placeState.points.pop();
    drawPlaces();
    updatePlaceControls();
  };

  lq("place-clear").onclick = () => {
    placeState.points = [];
    drawPlaces();
    updatePlaceControls();
    setPlaceStatus("");
  };

  document.querySelectorAll('input[name="place-kind"]').forEach((el) => {
    el.onchange = () => {
      placeState.points = [];
      drawPlaces();
      updatePlaceControls();
    };
  });

  lq("place-name").oninput = updatePlaceControls;
  lq("place-save").onclick = savePlace;
  lq("place-camera").onchange = (e) => selectCamera(Number(e.target.value));
  lq("place-frame").onload = drawPlaces;
  new ResizeObserver(drawPlaces).observe(lq("place-stage"));

  window.openPlaces = () => setOpen(true);
}
