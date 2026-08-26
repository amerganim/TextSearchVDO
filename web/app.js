const $ = (id) => document.getElementById(id);
const api = (path) => fetch(path).then((r) => r.json());

const state = {
  day: null,
  cameraId: "",
  label: "",
  timeline: null,
  segments: [],
  selected: null,
  // Search results replace the day's segments in the filmstrip while active.
  results: null,
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

/** Detection labels for a segment, e.g. {person: 2, dog: 1}. */
function labelsOf(segment) {
  if (!segment.labels) return null;
  try {
    return JSON.parse(segment.labels);
  } catch {
    return null;
  }
}

function matchesFilter(segment) {
  if (!state.label) return true;
  const labels = labelsOf(segment);
  return Boolean(labels && labels[state.label]);
}

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

  // With a label filter on, non-matching activity recedes rather than
  // disappearing, so "when was the dog here" reads against the rest of the
  // day instead of hiding it.
  const matching = new Set();
  if (state.label) {
    for (const segment of state.segments) {
      if (!matchesFilter(segment)) continue;
      const from = Math.floor((segment.ts_start - data.day_start) / data.bucket_seconds);
      const to = Math.ceil((segment.ts_end - data.day_start) / data.bucket_seconds);
      for (let i = from; i <= to; i++) matching.add(i);
    }
  }

  // Activity, height scaled by score so a busy minute reads louder.
  const peak = Math.max(...data.activity, 1e-6);
  for (let i = 0; i < n; i++) {
    const score = data.activity[i];
    if (score <= 0) continue;
    const frac = Math.min(1, Math.sqrt(score / peak));
    const h = Math.max(4, frac * bandH);
    ctx.fillStyle = !state.label || matching.has(i) ? "#ffb020" : "rgba(255,176,32,0.22)";
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
  const showing = state.results || state.segments;

  if (!showing.length) {
    strip.innerHTML = state.results
      ? '<div class="empty">Nothing matched that search.</div>'
      : '<div class="empty">No activity indexed for this day.</div>';
    return;
  }

  strip.innerHTML = showing
    .map((s) => {
      const on = state.selected && state.selected.id === s.id ? " selected" : "";
      const dim = state.results || matchesFilter(s) ? "" : " dimmed";
      let why = "";
      if (s.sources) {
        const sim = s.semantic_score != null ? `<b>${s.semantic_score.toFixed(3)}</b>` : "";
        why = `<span class="why">${sim}<span>${s.sources.join(" + ")}</span></span>`;
      }
      const labels = labelsOf(s);
      let tags = "";
      if (labels) {
        tags =
          '<span class="tags">' +
          Object.entries(labels)
            .map(([name, n]) => `<span class="tag">${n > 1 ? n + " " : ""}${name}</span>`)
            .join("") +
          "</span>";
      } else if (s.analyzed_at) {
        tags = '<span class="tags"><span class="tag">nothing recognised</span></span>';
      }
      return `<button class="card${on}${dim}" data-id="${s.id}">
        <img loading="lazy" src="/api/thumb/${s.id}" alt="">
        <span class="meta">
          <span class="time">${clockOf(s.ts_start)}</span>
          <span class="sub">${hms(s.ts_end - s.ts_start)} &middot; ${(s.activity_score * 100).toFixed(1)}%</span>
          ${tags}
          ${why}
        </span>
      </button>`;
    })
    .join("");

  strip.querySelectorAll(".card").forEach((el) => {
    el.onclick = () =>
      selectSegment(showing.find((s) => s.id === Number(el.dataset.id)));
  });
}

/* ---------- objects ---------- */

async function loadObjects(segment) {
  const box = $("objects");
  box.innerHTML = "";
  if (!segment || !segment.n_tracklets) return;

  const objects = await api(`/api/objects?segment_id=${segment.id}`);
  box.innerHTML = objects
    .map(
      (o) => `<button class="object" data-t="${o.t_start}" title="${o.label}">
        <img loading="lazy" src="/api/crop/${o.id}" alt="${o.label}">
        <span class="cap"><b>${o.label}</b>${clockOf(o.ts_start)}</span>
      </button>`
    )
    .join("");

  box.querySelectorAll(".object").forEach((el) => {
    el.onclick = () => {
      const video = $("video");
      // Land slightly before the object appeared rather than exactly on it.
      video.currentTime = Math.max(0, Number(el.dataset.t) - 1);
      video.play().catch(() => {});
    };
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
  loadObjects(segment);
}

function jumpToFraction(frac) {
  if (!state.timeline || !state.segments.length) return;
  const ts = state.timeline.day_start + frac * 86400;
  // Prefer segments matching the active filter, so clicking near a dog
  // sighting does not land on an unrelated one.
  const pool = state.segments.filter(matchesFilter);
  const candidates = pool.length ? pool : state.segments;

  let best = candidates[0];
  let bestGap = Infinity;
  for (const s of candidates) {
    const gap =
      ts >= s.ts_start && ts <= s.ts_end
        ? 0
        : Math.min(Math.abs(s.ts_start - ts), Math.abs(s.ts_end - ts));
    if (gap < bestGap) {
      bestGap = gap;
      best = s;
    }
  }
  selectSegment(best);
}

/* ---------- search ---------- */

function renderAnswer(body) {
  const panel = $("answer");
  const answer = body.answer;

  if (!answer) {
    // Clear as well as hide: a stale headline flashing on the next question
    // would read as an answer to it.
    panel.hidden = true;
    $("answer-headline").textContent = "";
    $("answer-understood").innerHTML = "";
    $("answer-rows").innerHTML = "";
    $("answer-caveat").hidden = true;
    return;
  }
  panel.hidden = false;
  $("answer-headline").textContent = answer.headline;

  const u = body.understood;
  const chips = u.matched.map((m) => `<span class="chip">${m.kind}: <b>${m.value}</b></span>`);
  if (u.semantic_text) {
    chips.push(`<span class="chip">also looking for: <b>${u.semantic_text}</b></span>`);
  }
  $("answer-understood").innerHTML = chips.join("");

  $("answer-rows").innerHTML = answer.rows
    .map((r) => {
      const when = new Date(r.ts * 1000).toLocaleString();
      const who = r.who || r.label;
      const detail = [
        r.kind ? r.kind.replace("_", " ") : "",
        r.zone ? `at ${r.zone}` : "",
        r.duration ? `for ${Math.round(r.duration)}s` : "",
      ].filter(Boolean).join(" ");
      return `<div class="answer-row" data-video="${r.video_id}" data-t="${r.t}">
        <span>${when}</span><span class="who">${who}</span>
        <span class="detail">${detail}</span>
      </div>`;
    })
    .join("");

  $("answer-rows").querySelectorAll(".answer-row").forEach((el) => {
    el.onclick = () => playAt(Number(el.dataset.video), Number(el.dataset.t));
  });

  const caveat = $("answer-caveat");
  caveat.hidden = !body.caveat;
  caveat.textContent = body.caveat || "";
}

function playAt(videoId, t) {
  const video = $("video");
  const seek = () => {
    video.currentTime = Math.max(0, t - 2);
    video.play().catch(() => {});
  };
  if (video.dataset.videoId !== String(videoId)) {
    video.dataset.videoId = String(videoId);
    video.src = `/api/media/${videoId}`;
    video.addEventListener("loadedmetadata", seek, { once: true });
  } else {
    seek();
  }
}

async function runSearch() {
  const text = $("q").value.trim();
  if (!text) return clearSearch();

  const query = new URLSearchParams({ q: text, limit: "60" });

  $("search-note").textContent = "thinking\u2026";
  const body = await api(`/api/ask?${query}`);
  renderAnswer(body);

  // Reshape hits into the segment shape the filmstrip already renders.
  state.results = body.results.map((r) => ({
    id: r.segment_id,
    video_id: r.video_id,
    camera_id: r.camera_id,
    t_start: r.t_start,
    t_end: r.t_end,
    ts_start: r.ts_start,
    ts_end: r.ts_end,
    activity_score: 0,
    labels: r.labels,
    analyzed_at: 1,
    n_tracklets: r.tracklet_id ? 1 : 0,
    sources: r.sources,
    semantic_score: r.semantic_score,
  }));

  state.selected = null;
  $("objects").innerHTML = "";
  $("search-clear").hidden = false;
  const n = body.results.length;
  if (body.mode === "answer") {
    $("search-note").textContent = "answered from the index";
  } else if (n) {
    $("search-note").textContent = `${n} closest match${n === 1 ? "" : "es"}`;
  } else {
    $("search-note").textContent = "nothing matched";
  }

  renderFilmstrip();
  drawTimeline();
}

function clearSearch() {
  state.results = null;
  $("q").value = "";
  $("search-clear").hidden = true;
  $("answer").hidden = true;
  $("search-note").textContent = "";
  renderFilmstrip();
  drawTimeline();
}

/* ---------- loading ---------- */

async function loadLabels() {
  const query = new URLSearchParams();
  if (state.day) query.set("day", state.day);
  if (state.cameraId) query.set("camera_id", state.cameraId);
  const labels = await api(`/api/labels?${query}`);

  const select = $("label");
  const previous = state.label;
  select.innerHTML =
    '<option value="">Everything</option>' +
    labels.map((l) => `<option value="${l.label}">${l.label} (${l.n})</option>`).join("");

  // Keep the filter across day changes, but only while it still applies.
  state.label = labels.some((l) => l.label === previous) ? previous : "";
  select.value = state.label;
  select.disabled = labels.length === 0;
}

async function loadDay() {
  if (!state.day) {
    state.timeline = null;
    state.segments = [];
    $("objects").innerHTML = "";
    renderFilmstrip();
    drawTimeline();
    return;
  }
  const query = new URLSearchParams({ day: state.day });
  if (state.cameraId) query.set("camera_id", state.cameraId);
  state.timeline = await api(`/api/timeline?${query}`);
  state.segments = state.timeline.segments;
  state.selected = null;
  state.results = null;
  $("search-clear").hidden = true;
  $("answer").hidden = true;
  $("objects").innerHTML = "";
  await loadLabels();
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
  let objectsLine = "";
  if (summary.n_analyzed) {
    objectsLine = ` &middot; <b>${summary.n_tracklets}</b> objects tracked`;
  } else if (summary.n_segments) {
    objectsLine = " &middot; not yet analysed — run <b>python -m tsv analyze</b>";
  }
  $("stats").innerHTML = summary.n_videos
    ? `<b>${hms(summary.duration)}</b> of footage &rarr; <b>${hms(summary.active_seconds)}</b> worth watching
       in <b>${summary.n_segments}</b> segments &middot; <b>${(summary.reduction * 100).toFixed(1)}%</b> needs no review${objectsLine}`
    : "Nothing indexed yet — run <b>python -m tsv ingest &lt;folder&gt;</b>";

  const cameras = await api("/api/cameras");
  $("camera").innerHTML =
    '<option value="">All cameras</option>' +
    cameras.map((c) => `<option value="${c.id}">${c.name}</option>`).join("");

  $("camera").onchange = async (e) => {
    state.cameraId = e.target.value;
    await loadDays();
    if (typeof refreshZonePanel === "function" && zoneState.open) {
      await refreshZonePanel(state.cameraId);
    }
  };
  $("day").onchange = async (e) => {
    state.day = e.target.value;
    await loadDay();
  };
  $("search-go").onclick = runSearch;
  $("search-clear").onclick = clearSearch;
  $("q").onkeydown = (e) => {
    if (e.key === "Enter") runSearch();
    if (e.key === "Escape") clearSearch();
  };

  $("label").onchange = (e) => {
    state.label = e.target.value;
    renderFilmstrip();
    drawTimeline();
  };

  $("timeline").onclick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    jumpToFraction((e.clientX - rect.left) / rect.width);
  };

  // Covers window resizes, the sidebar collapsing, and the first time the
  // canvas gets a non-zero box after being laid out hidden.
  new ResizeObserver(drawTimeline).observe($("timeline"));

  if (typeof initZones === "function") initZones(() => state.cameraId);

  await loadDays();
}

boot();
