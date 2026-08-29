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

const state = {
  poll: null, indexed: 0, importing: false, setupNeeded: false,
  // Captioning is part of indexing now, so this is normally zero. It is not
  // zero for anything indexed before that was true, and searching those
  // videos for what somebody was doing would silently find nothing.
  toCaption: 0, captionReady: false, captionStarted: false, captioning: false,
  zones: 0,
};

/** Captions come from a model, so they are text of unknown shape. */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

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

/** A standing note when parts of the app are not installed.
 *
 * Shown once and kept, rather than raised per failed action: a missing model
 * is a state of the installation, and finding out about it only when a search
 * returns nothing is how it used to feel broken.
 */
function showSetupBanner(summary) {
  const existing = document.getElementById("setup-note");
  if (!summary || !summary.setup_needed) {
    if (existing) existing.remove();
    return;
  }
  if (existing) return;

  const box = document.createElement("div");
  box.className = "notice warn";
  box.id = "setup-note";
  box.innerHTML =
    `<span>${summary.setup_needed.replace(
      "Run: python -m tsv setup",
      "Run <code>python -m tsv setup</code> to finish installing.",
    )}</span>`;
  $("notices").appendChild(box);
}

/** The search box is live whenever there is anything at all to search. */
function refreshChrome(summary) {
  showSetupBanner(summary);
  const has = state.indexed > 0;
  $("searchbar").hidden = !has;
  $("examples").hidden = !has;
  $("stage-empty").hidden = has;
  $("q").disabled = false;

  // Naming appears only once there is somebody to name, and carries the count
  // still unnamed - the whole point of the panel is the work outstanding.
  // Places can be drawn as soon as there is a camera to draw on, which is as
  // soon as anything has been indexed.
  $("places-toggle").hidden = !has;
  $("videos-toggle").hidden = !has;

  const people = $("people-toggle");
  const unnamed = summary ? summary.n_people_unnamed || 0 : 0;
  people.hidden = !summary || !summary.n_people;
  people.textContent = unnamed ? `People (${unnamed} to name)` : "People";
  if (summary && summary.n_people && !summary.faces_ready) {
    people.title = "Face matching is not installed, so a name covers one "
      + "sighting rather than every appearance.";
  } else {
    people.title = "Name someone once and every sighting of them is found";
  }

  // "Describe people" says how much work it is, because it is the one slow
  // thing in the app and starting it blind would be unkind.
  const describe = $("describe");
  if (!summary || !has) {
    describe.hidden = true;
    return;
  }
  const todo = summary.n_to_caption || 0;
  const done = summary.n_captioned || 0;   // only to decide whether to warn
  state.toCaption = todo;
  state.captionReady = Boolean(summary.caption_ready);

  if (!summary.caption_ready) {
    describe.hidden = done === 0;
    describe.textContent = "Descriptions unavailable";
    describe.disabled = true;
    describe.title = "The captioning model has not been fetched.";
    return;
  }
  // Only shown when there is work outstanding. Importing describes what it
  // finds, so with nothing pending this was a permanently disabled button
  // reading "Described 22" - a status wearing a control's clothes.
  describe.hidden = todo === 0;
  describe.disabled = state.importing;
  describe.title = `About ${Math.round((todo * 6) / 60) || 1} minute(s) of work`;
  describe.textContent = `Describe people (${todo})`;
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

/** What a running job should be called while it works. */
function jobTitle(job) {
  if (job.kind === "caption") return "Describing people";
  return job.title ? `Reading ${job.title}` : "Working";
}

function watchJob(jobId, kind) {
  clearInterval(state.poll);
  state.captioning = kind === "caption";
  showStrip(kind === "caption" ? "Describing people" : "Reading the video", 0.02, "");

  state.poll = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch {
      return;                       // transient; keep polling
    }
    showStrip(
      jobTitle(job),
      job.progress,
      [job.stage, job.message].filter(Boolean).join(" — "),
    );

    // Whatever has been read so far is already searchable.
    refreshChrome(await refreshLibrary());

    state.captioning = job.kind === "caption";

    if (job.status === "done") {
      clearInterval(state.poll);
      state.captioning = false;
      hideStrip();
      const r = job.result || {};
      const bits = [];
      if (r.duration) {
        bits.push(`${fmtDuration(r.duration)} of video`);
        bits.push(`${fmtDuration(r.active)} worth looking at`);
      }
      if (r.tracklets) bits.push(`${r.tracklets} object(s) found`);
      if (r.skipped) bits.push(`${r.skipped} already indexed`);
      // Say so plainly. "Ready." after adding a video that added nothing is
      // how the library quietly ended up counting the same hours twice.
      for (const line of r.duplicates || []) {
        notice(`Already in your library: ${escapeHtml(line)}`, "warn");
      }
      if (job.kind === "caption") {
        const described = r.captioned || 0;
        notice(
          described
            ? `Described ${described} ${described === 1 ? "person" : "people"} &middot; `
              + `now searchable by what they are doing`
            : "Nothing new to describe.",
          "good",
        );
      } else {
        const nothing = !bits.length && (r.duplicates || []).length;
        if (!nothing) notice(bits.join(" &middot; ") || "Ready.", "good");
      }
      if (r.utterances) {
        notice(
          `Heard ${r.utterances} line(s) of speech — now searchable by what `
          + "was said.",
          "good",
        );
      }
      if ((r.failed || []).length) notice(r.failed.join("; "));
      refreshChrome(await refreshLibrary());
      if (typeof reloadPeople === "function") reloadPeople();
      // Somebody who has just waited for a video to be read is looking for
      // something in that video, so that is what the next search covers.
      if (job.kind !== "caption" && job.kind !== "listen") {
        if (typeof reloadVideos === "function") await reloadVideos();
        if (r.files && typeof scopeToNewest === "function") scopeToNewest();
      }
      $("q").focus();
    } else if (job.status === "failed") {
      clearInterval(state.poll);
      state.captioning = false;
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

      // The description, where captioning has run. Without it a word match is
      // unexplained: "bag" matched, but the reader cannot see what was said.
      const said = h.caption
        ? `<span class="said">${escapeHtml(h.caption)}</span>`
        : "";

      return `<button class="hit" data-video="${h.video_id}" data-t="${h.t_start}" data-when="${when}">
        <img src="/api/thumb/${h.segment_id}" alt="">
        <span class="cap">
          <span class="when">${when}</span>
          <span class="sub">${objects || "movement"}</span>
          ${said}
          <span class="why">${why}</span>
        </span>
      </button>`;
    })
    .join("");

  $("results").querySelectorAll(".hit").forEach((el) => {
    el.onclick = () => openPlayer(el.dataset.video, Number(el.dataset.t), el.dataset.when);
  });
}

/** Finish describing anything left over, once, on the first search.
 *
 * A description that has not been written yet is a search that fails without
 * saying why: "carrying a bag" matches nothing, and the result looks the same
 * as the moment genuinely not being in the video. New imports describe
 * themselves, so this only ever fires on a library indexed before that.
 */
function catchUpOnCaptions() {
  if (state.captionStarted || state.importing) return;
  if (!state.captionReady || !state.toCaption) return;

  state.captionStarted = true;
  const todo = state.toCaption;
  api("/api/caption", { method: "POST" }).then((body) => {
    if (body.detail) return;
    notice(
      `Describing ${todo} ${todo === 1 ? "person that was" : "people that were"} `
      + "indexed before descriptions were switched on. Search again once it "
      + "finishes to include what they were doing.",
      "good",
    );
    watchJob(body.id, "caption");
  });
}

async function runSearch() {
  const text = $("q").value.trim();
  if (!text) return;

  catchUpOnCaptions();

  $("stage-results").hidden = false;
  $("status").textContent = "Looking…";
  $("nothing").hidden = true;
  $("draw-a-place").hidden = true;
  $("results").innerHTML = "";

  const scope = typeof searchScope === "function" ? searchScope() : null;

  let body;
  try {
    body = await api(
      `/api/ask?q=${encodeURIComponent(text)}&limit=48`
      + (scope ? `&video_id=${scope}` : "")
    );
  } catch (err) {
    $("status").textContent = "";
    notice(`Search failed: ${err}`);
    return;
  }
  renderAnswer(body);

  const rows = body.results || [];

  // An exact answer wins only when it used the whole question. "a person
  // carrying something" grounds "a person" and drops the rest, so answering
  // it exactly lists every person in the recording - while the ranked search
  // right beside it has already found the one holding a bag. The API returns
  // both; picking the answer regardless threw the better one away.
  const leftover = (body.understood && body.understood.semantic_text) || "";
  const answered = body.answer && body.answer.found && !leftover;

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

  // A direction question against no drawn place is not "nothing found", it
  // is a question the index cannot answer yet. Reporting it as absent is a
  // wrong answer that looks like a right one, and leaves the reader with no
  // idea that a two-click setup step would fix it.
  const wantsDirection = Boolean(body.understood && body.understood.event_kind);
  if (wantsDirection && !state.zones) {
    $("status").textContent = "";
    $("nothing").hidden = false;
    $("nothing-what").textContent = "Nothing to measure that against";
    $("nothing-why").textContent =
      "You asked about going in or out, but no doorway has been drawn yet. "
      + "Two clicks across one in the camera view is enough, and it reads the "
      + "videos you already have.";
    $("draw-a-place").hidden = false;
    return;
  }

  if (!rows.length) {
    $("status").textContent = "";
    $("nothing").hidden = false;
    $("nothing-what").textContent = "Nothing found";
    $("draw-a-place").hidden = true;
    if (body.answer) {
      $("nothing-why").textContent = body.answer.headline;
    } else if (state.captioning) {
      // Not the same as absent. Until a sighting is described, nothing about
      // what somebody was *doing* is searchable, and saying "nothing found"
      // would be a wrong answer that looks like a right one.
      $("nothing-why").textContent =
        "Nothing matching yet — what people were doing is still being described. "
        + "Try this search again once that finishes.";
    } else if (state.importing) {
      $("nothing-why").textContent =
        "Nothing matching so far — the video is still being read, so try again in a moment.";
    } else if (state.setupNeeded) {
      // Distinguish "not in the video" from "this app cannot look for that
      // yet", which are the same empty screen but entirely different problems.
      $("nothing-why").textContent =
        "This copy is not fully set up, so searching by description is limited. "
        + "Run python -m tsv setup, then try again.";
    } else if (scope) {
      // The most likely reason for an empty screen once scoping exists.
      $("nothing-why").textContent =
        "Nothing in this video looks like that. It only searched the one "
        + "video named above — widen it to everything, or try simpler words.";
    } else {
      $("nothing-why").textContent =
        "Nothing in the video looks like that. Try simpler words, or describe "
        + "what is visible rather than what happened.";
    }
    return;
  }

  renderHits(rows);
  $("status").textContent = leftover && body.answer && body.answer.found
    ? `${rows.length} possible match(es) for "${escapeHtml(leftover)}", closest first`
    : `${rows.length} possible match(es), closest first`;
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
  state.setupNeeded = Boolean(summary.setup_needed);
  state.zones = summary.n_zones || 0;
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
  $("describe").onclick = async () => {
    const body = await api("/api/caption", { method: "POST" });
    if (body.detail) return notice(body.detail);
    watchJob(body.id, "caption");
  };
  $("choose").onclick = chooseFiles;
  $("draw-a-place").onclick = () => window.openPlaces && window.openPlaces();
  $("filepick").onchange = (e) => {
    importFiles(e.target.files);
    e.target.value = "";           // let the same file be picked again
  };

  initPeople();
  initPlaces();
  initVideos();

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

  for (const name of ["dragenter", "dragover"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      document.body.classList.add("dragging");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    document.addEventListener(name, (e) => {
      e.preventDefault();
      document.body.classList.remove("dragging");
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

  refreshChrome(await refreshLibrary());

  const running = (await api("/api/jobs")).find(
    (j) => j.status === "running" || j.status === "queued"
  );
  if (running) watchJob(running.id);
  if (state.indexed) $("q").focus();
}

boot();
