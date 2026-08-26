/* The simple face of the app.
 *
 * One page that is always usable. Importing runs in a strip at the top rather
 * than taking the app over, because a long recording can take minutes and
 * being locked out of a search box while you wait is worse than the wait.
 * Motion segmentation finishes early, so there is usually something to search
 * long before analysis ends.
 *
 * Two environments, one flow. In the desktop window pywebview hands us real
 * paths and nothing is copied. In a browser there is only a File, so it
 * uploads. Dropping files works in a browser and *not* in the desktop window,
 * which has no drop support at all - so a drop that yields nothing says so
 * instead of silently doing nothing.
 */

const $ = (id) => document.getElementById(id);
const api = (path, options) => fetch(path, options).then((r) => r.json());

const state = { poll: null, indexed: 0, importing: false };

const EXAMPLES = [
  "a person",
  "someone carrying something",
  "a car",
  "a dog",
];

const inDesktop = () => Boolean(window.pywebview && window.pywebview.api);

/* ---------- chrome ---------- */

function fmtDuration(seconds) {
  seconds = Math.round(seconds || 0);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function notice(message, kind = "error") {
  const box = document.createElement("div");
  box.className = `notice ${kind}`;
  box.innerHTML = `<span>${message}</span>`;
  const close = document.createElement("button");
  close.className = "ghost tiny";
  close.textContent = "Dismiss";
  close.onclick = () => box.remove();
  box.appendChild(close);
  $("notices").prepend(box);
}

/** The search box is live whenever there is anything at all to search. */
function refreshChrome() {
  const has = state.indexed > 0;
  $("searchbar").hidden = !has;
  $("examples").hidden = !has;
  $("stage-empty").hidden = has;
  $("q").disabled = false;
}

/* ---------- importing ---------- */

async function importPaths(paths) {
  for (const path of paths) {
    const body = await api("/api/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (body.detail) {
      notice(body.detail);
      continue;
    }
    watchJob(body.id);
  }
}

async function importFiles(files) {
  if (!files || !files.length) return false;
  showStrip("Copying " + files[0].name, 0.02, "");

  let last = null;
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      const body = await api("/api/import/upload", { method: "POST", body: form });
      if (body.detail) {
        notice(body.detail);
        continue;
      }
      last = body.id;
    } catch (err) {
      notice(`Could not read ${file.name}: ${err}`);
    }
  }
  if (last) watchJob(last);
  else hideStrip();
  return true;
}

async function chooseFiles() {
  if (inDesktop()) {
    const paths = await window.pywebview.api.pick_videos();
    if (paths && paths.length) importPaths(paths);
    return;
  }
  $("filepick").click();
}

/* ---------- progress strip ---------- */

function showStrip(title, progress, detail) {
  state.importing = true;
  $("strip").hidden = false;
  $("strip-title").textContent = title;
  $("strip-detail").textContent = detail || "";
  $("strip-pct").textContent = `${Math.round((progress || 0) * 100)}%`;
  $("strip-fill").style.width = `${Math.max(2, (progress || 0) * 100).toFixed(1)}%`;
}

function hideStrip() {
  state.importing = false;
  $("strip").hidden = true;
}

function watchJob(jobId) {
  clearInterval(state.poll);
  showStrip("Reading the video", 0.02, "");

  state.poll = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch {
      return;                       // transient; keep polling
    }
    showStrip(
      job.title ? `Reading ${job.title}` : "Working",
      job.progress,
      [job.stage, job.message].filter(Boolean).join(" — "),
    );

    // Whatever has been read so far is already searchable.
    await refreshLibrary();
    refreshChrome();

    if (job.status === "done") {
      clearInterval(state.poll);
      hideStrip();
      const r = job.result || {};
      const bits = [];
      if (r.duration) {
        bits.push(`${fmtDuration(r.duration)} of video`);
        bits.push(`${fmtDuration(r.active)} worth looking at`);
      }
      if (r.tracklets) bits.push(`${r.tracklets} object(s) found`);
      if (r.skipped) bits.push(`${r.skipped} already indexed`);
      notice(bits.join(" &middot; ") || "Ready.", "good");
      if ((r.failed || []).length) notice(r.failed.join("; "));
      $("q").focus();
    } else if (job.status === "failed") {
      clearInterval(state.poll);
      hideStrip();
      notice(job.error || "That video could not be read.");
    }
  }, 400);
}

/* ---------- searching ---------- */

function renderAnswer(body) {
  const answer = body.answer;
  if (!answer || !answer.found) {
    $("answer").hidden = true;
    $("answer-headline").textContent = "";
    $("answer-caveat").hidden = true;
    return;
  }
  $("answer").hidden = false;
  $("answer-headline").textContent = answer.headline;
  $("answer-caveat").hidden = !body.caveat;
  $("answer-caveat").textContent = body.caveat || "";
}

function renderHits(hits) {
  $("results").innerHTML = hits
    .map((h) => {
      const when = new Date(h.ts_start * 1000).toLocaleString();
      let objects = "";
      try {
        const labels = h.labels ? JSON.parse(h.labels) : null;
        if (labels) {
          objects = Object.entries(labels)
            .map(([k, n]) => (n > 1 ? `${n} ${k}s` : k))
            .join(", ");
        }
      } catch { /* labels are optional */ }

      const why = h.semantic_score != null
        ? `${(h.semantic_score * 100).toFixed(0)}% match`
        : (h.sources || []).join(" + ");

      return `<button class="hit" data-video="${h.video_id}" data-t="${h.t_start}" data-when="${when}">
        <img src="/api/thumb/${h.segment_id}" alt="">
        <span class="cap">
          <span class="when">${when}</span>
          <span class="sub">${objects || "movement"}</span>
          <span class="why">${why}</span>
        </span>
      </button>`;
    })
    .join("");

  $("results").querySelectorAll(".hit").forEach((el) => {
    el.onclick = () => openPlayer(el.dataset.video, Number(el.dataset.t), el.dataset.when);
  });
}

async function runSearch() {
  const text = $("q").value.trim();
  if (!text) return;

  $("stage-results").hidden = false;
  $("status").textContent = "Looking…";
  $("nothing").hidden = true;
  $("results").innerHTML = "";

  let body;
  try {
    body = await api(`/api/ask?q=${encodeURIComponent(text)}&limit=48`);
  } catch (err) {
    $("status").textContent = "";
    notice(`Search failed: ${err}`);
    return;
  }
  renderAnswer(body);

  const rows = body.results || [];
  const answered = body.answer && body.answer.found;

  if (answered) {
    const seen = new Set();
    const fromAnswer = body.answer.rows
      .filter((r) => !seen.has(r.segment_id) && seen.add(r.segment_id))
      .map((r) => ({
        segment_id: r.segment_id, video_id: r.video_id, t_start: r.t,
        ts_start: r.ts,
        labels: JSON.stringify({ [r.who || r.label]: 1 }),
        sources: ["answer"], semantic_score: null,
      }));
    renderHits(fromAnswer);
    $("status").textContent = `${fromAnswer.length} moment(s)`;
    return;
  }

  if (!rows.length) {
    $("status").textContent = "";
    $("nothing").hidden = false;
    $("nothing-why").textContent = body.answer
      ? body.answer.headline
      : state.importing
        ? "Nothing matching so far — the video is still being read, so try again in a moment."
        : "Nothing in the video looks like that. Try simpler words, or describe "
          + "what is visible rather than what happened.";
    return;
  }

  renderHits(rows);
  $("status").textContent = `${rows.length} possible match(es), closest first`;
}

/* ---------- player ---------- */

function openPlayer(videoId, t, caption) {
  const video = $("video");
  $("player-caption").textContent = caption || "";
  $("player-backdrop").hidden = false;

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

function closePlayer() {
  $("player-backdrop").hidden = true;
  $("video").pause();
}

/* ---------- library ---------- */

async function refreshLibrary() {
  const summary = await api("/api/summary");
  state.indexed = summary.n_videos || 0;
  $("library").innerHTML = state.indexed
    ? `<b>${state.indexed}</b> video(s) &middot; <b>${fmtDuration(summary.duration)}</b> indexed`
    : "";
  return summary;
}

/* ---------- boot ---------- */

async function boot() {
  $("examples").innerHTML = EXAMPLES.map(
    (e) => `<button type="button" data-q="${e}">${e}</button>`
  ).join("");
  $("examples").querySelectorAll("button").forEach((el) => {
    el.onclick = () => {
      $("q").value = el.dataset.q;
      runSearch();
    };
  });

  $("go").onclick = runSearch;
  $("q").onkeydown = (e) => {
    if (e.key === "Enter") runSearch();
  };
  $("add").onclick = chooseFiles;
  $("choose").onclick = chooseFiles;
  $("filepick").onchange = (e) => {
    importFiles(e.target.files);
    e.target.value = "";           // let the same file be picked again
  };

  $("player-close").onclick = closePlayer;
  $("player-backdrop").onclick = (e) => {
    if (e.target === $("player-backdrop")) closePlayer();
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePlayer();
  });

  // Dragging works in a browser. The desktop window has no drop support at
  // all, so say so rather than appearing to ignore the file.
  if (inDesktop()) {
    $("or-drop").textContent = "";
  }

  const zone = $("dropzone");
  for (const name of ["dragenter", "dragover"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      if (zone) zone.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      if (zone) zone.classList.remove("over");
    });
  }
  document.addEventListener("drop", async (e) => {
    e.preventDefault();
    const files = e.dataTransfer ? e.dataTransfer.files : null;
    const handled = await importFiles(files);
    if (!handled) {
      notice(
        "This window cannot read dropped files. Use <b>Add video</b> to pick "
        + "one instead — it opens a normal file chooser.",
        "warn",
      );
    }
  });

  await refreshLibrary();
  refreshChrome();

  const running = (await api("/api/jobs")).find(
    (j) => j.status === "running" || j.status === "queued"
  );
  if (running) watchJob(running.id);
  if (state.indexed) $("q").focus();
}

boot();
