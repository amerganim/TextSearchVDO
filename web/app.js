const $ = (id) => document.getElementById(id);
const api = (path) => fetch(path).then((r) => r.json());

const state = {
  day: null,
  cameraId: "",
  timeline: null,
  segments: [],
  selected: null,
};

const pad = (n) => String(n).padStart(2, "0");
const clockOf = (ts) => {
  const d = new Date(ts * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};
const hms = (s) => {
  s = Math.round(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${pad(m)}m` : `${m}m ${pad(s % 60)}s`;
};

/* ---------- timeline ---------- */

function drawTimeline() {
  const canvas = $("timeline");
  const data = state.timeline;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;

  // The canvas can measure zero while the page is laid out - loading in a
  // background tab is the common way to hit this. Bail rather than sizing the
  // backing store to nothing; the ResizeObserver redraws once it has a box.
  if (width < 2) return;

  canvas.width = width * dpr;
  canvas.height = 88 * dpr;
  canvas.style.height = "88px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, 88);

  if (!data) return;

  const n = data.activity.length;
  const barW = width / n;
  const top = 10;
  const bandH = 58;

  // Recording coverage first: the reader has to be able to tell "the camera
  // saw nothing" apart from "the camera was not recording".
  for (let i = 0; i < n; i++) {
    const covered = data.coverage[i] / data.bucket_seconds;
    if (covered <= 0.01) continue;
    ctx.fillStyle = `rgba(51, 65, 92, ${0.35 + 0.65 * Math.min(1, covered)})`;
    ctx.fillRect(i * barW, top, Math.max(barW, 0.6), bandH);
  }

  // Activity on top, height scaled by score so a busy minute reads louder.
  const peak = Math.max(...data.activity, 1e-6);
  for (let i = 0; i < n; i++) {
    const score = data.activity[i];
    if (score <= 0) continue;
    const frac = Math.min(1, Math.sqrt(score / peak));
    const h = Math.max(4, frac * bandH);
    ctx.fillStyle = "#ffb020";
    ctx.fillRect(i * barW, top + bandH - h, Math.max(barW, 1.2), h);
  }

  // Hour gridlines.
  ctx.strokeStyle = "rgba(230,234,242,0.10)";
  ctx.lineWidth = 1;
  for (let hour = 0; hour <= 24; hour += 1) {
    const x = Math.round((hour / 24) * width) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, top - 4);
    ctx.lineTo(x, top + bandH + (hour % 6 === 0 ? 8 : 4));
    ctx.stroke();
  }

  if (state.selected) {
    const frac = (state.selected.ts_start - data.day_start) / 86400;
    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(frac * width, top - 6);
    ctx.lineTo(frac * width, top + bandH + 8);
    ctx.stroke();
  }
}

function renderHours() {
  $("hours").innerHTML = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    .map((h) => `<span>${pad(h % 24)}:00</span>`)
    .join("");
}

/* ---------- filmstrip ---------- */

function renderFilmstrip() {
  const strip = $("filmstrip");
  if (!state.segments.length) {
    strip.innerHTML = '<div class="empty">No activity indexed for this day.</div>';
    return;
  }
  strip.innerHTML = state.segments
    .map((s) => {
      const on = state.selected && state.selected.id === s.id ? " selected" : "";
      return `<button class="card${on}" data-id="${s.id}">
        <img loading="lazy" src="/api/thumb/${s.id}" alt="">
        <span class="meta">
          <span class="time">${clockOf(s.ts_start)}</span>
          <span class="sub">${hms(s.ts_end - s.ts_start)} &middot; ${(s.activity_score * 100).toFixed(1)}%</span>
        </span>
      </button>`;
    })
    .join("");

  strip.querySelectorAll(".card").forEach((el) => {
    el.onclick = () => selectSegment(state.segments.find((s) => s.id === Number(el.dataset.id)));
  });
}

/* ---------- playback ---------- */

function selectSegment(segment) {
  if (!segment) return;
  state.selected = segment;
  const video = $("video");
  const src = `/api/media/${segment.video_id}`;

  const seek = () => {
    video.currentTime = segment.t_start;
    video.play().catch(() => {});
  };

  if (video.dataset.videoId !== String(segment.video_id)) {
    video.dataset.videoId = String(segment.video_id);
    video.src = src;
    video.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    seek();
  }

  $("now").textContent =
    `${clockOf(segment.ts_start)} - ${clockOf(segment.ts_end)}  ` +
    `(${hms(segment.ts_end - segment.ts_start)} of activity)`;

  renderFilmstrip();
  drawTimeline();
}

function jumpToFraction(frac) {
  if (!state.timeline || !state.segments.length) return;
  const ts = state.timeline.day_start + frac * 86400;
  // Nearest segment by start time - clicking empty timeline should still land
  // somewhere useful rather than do nothing.
  let best = state.segments[0];
  let bestGap = Infinity;
  for (const s of state.segments) {
    const gap = ts >= s.ts_start && ts <= s.ts_end ? 0 : Math.min(Math.abs(s.ts_start - ts), Math.abs(s.ts_end - ts));
    if (gap < bestGap) {
      bestGap = gap;
      best = s;
    }
  }
  selectSegment(best);
}

/* ---------- loading ---------- */

async function loadDay() {
  if (!state.day) {
    state.timeline = null;
    state.segments = [];
    renderFilmstrip();
    drawTimeline();
    return;
  }
  const query = new URLSearchParams({ day: state.day });
  if (state.cameraId) query.set("camera_id", state.cameraId);
  state.timeline = await api(`/api/timeline?${query}`);
  state.segments = state.timeline.segments;
  state.selected = null;
  renderFilmstrip();
  drawTimeline();
}

async function loadDays() {
  const query = state.cameraId ? `?camera_id=${state.cameraId}` : "";
  const days = await api(`/api/days${query}`);
  const select = $("day");
  select.innerHTML = days
    .map((d) => `<option value="${d.day}">${d.day} — ${d.n_segments} segments</option>`)
    .join("");
  state.day = days.length ? days[0].day : null;
  if (state.day) select.value = state.day;
  await loadDay();
}

async function boot() {
  renderHours();

  const summary = await api("/api/summary");
  $("stats").innerHTML = summary.n_videos
    ? `<b>${hms(summary.duration)}</b> of footage &rarr; <b>${hms(summary.active_seconds)}</b> worth watching
       in <b>${summary.n_segments}</b> segments &middot; <b>${(summary.reduction * 100).toFixed(1)}%</b> needs no review`
    : "Nothing indexed yet — run <b>python -m tsv ingest &lt;folder&gt;</b>";

  const cameras = await api("/api/cameras");
  $("camera").innerHTML =
    '<option value="">All cameras</option>' +
    cameras.map((c) => `<option value="${c.id}">${c.name}</option>`).join("");

  $("camera").onchange = async (e) => {
    state.cameraId = e.target.value;
    await loadDays();
  };
  $("day").onchange = async (e) => {
    state.day = e.target.value;
    await loadDay();
  };

  $("timeline").onclick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    jumpToFraction((e.clientX - rect.left) / rect.width);
  };

  // Covers window resizes, the sidebar collapsing, and the first time the
  // canvas gets a non-zero box after being laid out hidden.
  new ResizeObserver(drawTimeline).observe($("timeline"));

  await loadDays();
}

boot();
