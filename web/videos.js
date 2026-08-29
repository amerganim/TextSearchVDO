/* The library, and what a search actually covers.
 *
 * The complaint this answers: "it seems the app always loaded with previous
 * videos". It did, and it was right to - a library you can search across days
 * is the point of a CCTV tool, and wiping it on every launch would throw away
 * the archive the product exists to search. What was missing was any way to
 * see what was in there, take something out, or tell which recording a result
 * came from.
 *
 * So: the library persists, and the scope is made visible instead. A search
 * covers the most recently added recording by default, because "I just added
 * this, find something in it" is what somebody is nearly always doing, and one
 * click widens it to everything. Both states are on screen; neither is a
 * silent default.
 */

const videoState = {
  open: false,
  rows: [],
  // null means every video. Set to an id, the search box says so.
  scope: null,
};

const vq = (id) => document.getElementById(id);

function videoEscape(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function videoDuration(seconds) {
  seconds = Math.round(seconds || 0);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/* ---------- scope ---------- */

/** Show which recording a search covers, or hide the strip when it is all. */
function renderScope() {
  const strip = vq("scope");
  const row = videoState.rows.find((v) => v.id === videoState.scope);
  if (!row) {
    videoState.scope = null;
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  vq("scope-name").textContent = row.name;
}

function setScope(videoId) {
  videoState.scope = videoId;
  renderScope();
  renderVideos();
}

/** Scope to a newly imported recording.
 *
 * Called when an import finishes. Somebody who has just waited for a video to
 * be read is looking for something in *that* video, and showing them a hit
 * from last week's footage without saying so reads as the app being wrong.
 */
function scopeToNewest() {
  const newest = videoState.rows[0];
  if (newest) setScope(newest.id);
}

/* ---------- the list ---------- */

function renderVideos() {
  const list = vq("video-list");
  const empty = vq("videos-empty");

  if (!videoState.rows.length) {
    list.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;

  list.innerHTML = videoState.rows
    .map((row) => {
      const when = new Date(row.start_ts * 1000).toLocaleString();
      const scoped = row.id === videoState.scope;
      const bits = [`${row.n_segments} moment${row.n_segments === 1 ? "" : "s"}`];
      if (row.n_tracklets) bits.push(`${row.n_tracklets} object(s)`);
      if (row.n_utterances) bits.push(`${row.n_utterances} line(s) heard`);
      if (!row.present) bits.push("file has moved");

      return `<div class="video-row${scoped ? " scoped" : ""}${row.present ? "" : " missing"}" data-id="${row.id}">
        <div class="video-main">
          <span class="video-name">${videoEscape(row.name)}</span>
          <span class="video-meta">${when} &middot; ${videoDuration(row.duration)} &middot; ${bits.join(" &middot; ")}</span>
        </div>
        <div class="video-actions">
          ${scoped
            ? '<span class="video-badge">searching this</span>'
            : `<button type="button" class="linkish only" data-id="${row.id}">Search only this</button>`}
          <button type="button" class="linkish remove" data-id="${row.id}">Remove</button>
        </div>
      </div>`;
    })
    .join("");

  list.querySelectorAll(".only").forEach((el) => {
    el.onclick = () => setScope(Number(el.dataset.id));
  });
  list.querySelectorAll(".remove").forEach((el) => {
    el.onclick = () => removeVideo(Number(el.dataset.id));
  });
}

async function loadVideos() {
  videoState.rows = await fetch("/api/videos").then((r) => r.json()).catch(() => []);
  // A scope pointing at something no longer here is worse than none.
  if (videoState.scope && !videoState.rows.some((v) => v.id === videoState.scope)) {
    videoState.scope = null;
  }
  renderScope();
  renderVideos();
  return videoState.rows;
}

/* ---------- removal ---------- */

async function removeVideo(videoId) {
  const row = videoState.rows.find((v) => v.id === videoId);
  const name = row ? row.name : "this video";
  // Removal takes moments out of the index for good. Cheap to redo by adding
  // the file again, but not something to do on a stray click.
  if (!window.confirm(
    `Remove "${name}" from the library?\n\n`
    + "Its moments, objects and transcript are deleted from the index. "
    + "The recording itself is left alone unless this app made the copy."
  )) {
    return;
  }

  const response = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
  if (!response.ok) {
    notice("That video could not be removed.");
    return;
  }
  const body = await response.json();
  notice(
    `Removed ${videoEscape(name)}`
    + (body.file_removed ? " and the copy this app had made of it." : "."),
    "good",
  );
  await loadVideos();
  if (typeof refreshLibrary === "function") refreshChrome(await refreshLibrary());
}

/* ---------- wiring ---------- */

function initVideos() {
  const panel = vq("videos-panel");
  const toggle = vq("videos-toggle");

  const setOpen = async (open) => {
    videoState.open = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) await loadVideos();
  };

  toggle.onclick = () => setOpen(!videoState.open);
  vq("videos-close").onclick = () => setOpen(false);
  vq("scope-all").onclick = () => setScope(null);

  window.openVideos = () => setOpen(true);
  window.reloadVideos = loadVideos;
  window.scopeToNewest = scopeToNewest;
  window.searchScope = () => videoState.scope;
}
