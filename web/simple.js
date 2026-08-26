/* The simple face of the app: drop a video, wait, search it.
 *
 * Runs in two environments and has to behave in both. Inside the desktop
 * window the file is already on disk and pywebview hands us its path, so
 * nothing is copied. In a plain browser there is no path - only a File - so it
 * uploads. The rest of the flow is identical.
 */

const $ = (id) => document.getElementById(id);
const api = (path, options) => fetch(path, options).then((r) => r.json());

const state = { poll: null, ready: false };

const EXAMPLES = [
  "a person carrying something",
  "someone at the door",
  "a car",
  "a dog",
];

/* ---------- stages ---------- */

function show(stage) {
  for (const id of ["stage-empty", "stage-working", "stage-ready"]) {
    $(id).hidden = id !== stage;
  }
  $("add-more").hidden = stage !== "stage-ready";
}

function fmtDuration(seconds) {
  seconds = Math.round(seconds || 0);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function showError(message) {
  const box = document.createElement("div");
  box.className = "error";
  box.textContent = message;
  const stage = $("stage-empty").hidden ? $("stage-ready") : $("stage-empty");
  stage.querySelectorAll(".error").forEach((e) => e.remove());
  stage.appendChild(box);
}

/* ---------- importing ---------- */

async function startImportFromPath(path) {
  const body = await api("/api/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (body.detail) return showError(body.detail);
  watchJob(body.id);
}

async function startImportFromFiles(files) {
  if (!files || !files.length) return;
  show("stage-working");
  $("work-title").textContent = "Copying the video";
  $("work-stage").textContent = files[0].name;
  $("work-fill").style.width = "4%";

  let last = null;
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    const body = await api("/api/import/upload", { method: "POST", body: form });
    if (body.detail) return showError(body.detail);
    last = body.id;
  }
  if (last) watchJob(last);
}

function watchJob(jobId) {
  show("stage-working");
  clearInterval(state.poll);

  state.poll = setInterval(async () => {
    const job = await api(`/api/jobs/${jobId}`);
    $("work-title").textContent = job.title ? `Reading ${job.title}` : "Working";
    $("work-stage").textContent = job.stage || "";
    $("work-message").textContent = job.message || "";
    $("work-fill").style.width = `${Math.max(3, job.progress * 100).toFixed(1)}%`;

    if (job.status === "done") {
      clearInterval(state.poll);
      await refreshLibrary();
      show("stage-ready");
      const r = job.result || {};
      const bits = [];
      if (r.duration) {
        bits.push(`${fmtDuration(r.duration)} of video`);
        bits.push(`${fmtDuration(r.active)} worth looking at`);
      }
      if (r.tracklets) bits.push(`${r.tracklets} object(s) found`);
      $("status").textContent = bits.join(" · ") || "Ready.";
      if ((r.failed || []).length) showError(r.failed.join("; "));
      $("q").focus();
    } else if (job.status === "failed") {
      clearInterval(state.poll);
      show(state.ready ? "stage-ready" : "stage-empty");
      showError(job.error || "Import failed.");
    }
  }, 500);
}

/* ---------- searching ---------- */

function renderAnswer(body) {
  const panel = $("answer");
  const answer = body.answer;
  if (!answer || !answer.found) {
    panel.hidden = true;
    $("answer-headline").textContent = "";
    $("answer-caveat").hidden = true;
    return;
  }
  panel.hidden = false;
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
        <img loading="lazy" src="/api/thumb/${h.segment_id}" alt="">
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

  $("status").textContent = "Looking…";
  $("nothing").hidden = true;
  $("results").innerHTML = "";

  const body = await api(`/api/ask?q=${encodeURIComponent(text)}&limit=48`);
  renderAnswer(body);

  const answered = body.answer && body.answer.found;
  const rows = body.results || [];

  if (answered) {
    // An exact answer already lists its moments; show those as the frames.
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
      : "Nothing in the indexed video looks like that. Try simpler words, "
        + "or describe what is visible rather than what happened.";
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
    // Start slightly before, so the moment is not already over.
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
  state.ready = summary.n_videos > 0;
  $("library").innerHTML = state.ready
    ? `<b>${summary.n_videos}</b> video(s) &middot; <b>${fmtDuration(summary.duration)}</b> indexed`
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
  $("player-close").onclick = closePlayer;
  $("player-backdrop").onclick = (e) => {
    if (e.target === $("player-backdrop")) closePlayer();
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePlayer();
  });

  // Choosing a file: the desktop window has a native dialog and real paths.
  const pickNatively = async () => {
    const paths = await window.pywebview.api.pick_videos();
    if (paths && paths.length) {
      show("stage-working");
      for (const path of paths) await startImportFromPath(path);
    }
  };
  const inDesktop = () => Boolean(window.pywebview && window.pywebview.api);

  $("browse").onclick = () => (inDesktop() ? pickNatively() : $("filepick").click());
  $("add-more").onclick = () => (inDesktop() ? pickNatively() : $("filepick").click());
  $("filepick").onchange = (e) => startImportFromFiles(e.target.files);

  const zone = $("dropzone");
  for (const name of ["dragenter", "dragover"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      zone.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      zone.classList.remove("over");
    });
  }
  document.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) {
      startImportFromFiles(e.dataTransfer.files);
    }
  });

  const summary = await refreshLibrary();
  show(summary.n_videos ? "stage-ready" : "stage-empty");

  // An import may already be running from a previous window.
  const running = (await api("/api/jobs")).find(
    (j) => j.status === "running" || j.status === "queued"
  );
  if (running) watchJob(running.id);
}

boot();
