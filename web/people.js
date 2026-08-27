/* Naming the people in a video.
 *
 * This is what turns "a person went out the front door" into "Rafi went out
 * the front door". Naming one sighting adds it to a gallery; matching then
 * finds the same face elsewhere, so the work is one name rather than one name
 * per appearance.
 *
 * Deliberately shows the unnamed first and never invents a name. Getting this
 * wrong is worse than leaving it undone: a wrong name is a confident, wrong
 * answer to every later question about that person.
 */

const peopleState = {
  open: false,
  loading: false,
  rows: [],
  editing: null,
};

const pq = (id) => document.getElementById(id);

function peopleEscape(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function peopleClock(ts) {
  return new Date(ts * 1000).toLocaleString();
}

/* ---------- loading ---------- */

async function loadPeople() {
  peopleState.loading = true;
  const grid = pq("people-grid");
  const empty = pq("people-empty");

  const [objects, identities, summary] = await Promise.all([
    fetch("/api/objects?label=person&limit=200").then((r) => r.json()),
    fetch("/api/identities").then((r) => r.json()),
    fetch("/api/summary").then((r) => r.json()),
  ]);
  peopleState.loading = false;

  // Only sightings with a crop can be shown; naming a blank square is asking
  // somebody to guess.
  peopleState.rows = objects.filter((o) => o.thumb_path);

  if (!peopleState.rows.length) {
    grid.innerHTML = "";
    empty.hidden = false;
    empty.innerHTML = summary.n_tracklets
      ? "No people found in what has been read so far."
      : "Nothing has been read yet. Add a video first.";
    return;
  }
  empty.hidden = true;

  // Unnamed first: those are the ones needing a decision.
  const rows = [...peopleState.rows].sort((a, b) => {
    const named = Number(Boolean(a.identity_name)) - Number(Boolean(b.identity_name));
    return named !== 0 ? named : a.ts_start - b.ts_start;
  });

  grid.innerHTML = rows
    .map((row) => {
      const named = Boolean(row.identity_name);
      const auto = row.identity_source === "auto";
      return `<figure class="person${named ? " named" : ""}" data-id="${row.id}">
        <img loading="lazy" src="/api/crop/${row.id}" alt="A person seen at ${peopleClock(row.ts_start)}">
        <figcaption>
          <span class="person-when">${peopleClock(row.ts_start)}</span>
          ${named
            ? `<span class="person-name">${peopleEscape(row.identity_name)}${
                auto ? '<span class="auto" title="A guess, matched against someone you named">guessed</span>' : ""
              }</span>
               <span class="person-actions">
                 <button type="button" class="linkish rename" data-id="${row.id}">Rename</button>
                 <button type="button" class="linkish wrong" data-id="${row.id}">Not them</button>
               </span>`
            : `<button type="button" class="name-it" data-id="${row.id}">Name this person</button>`}
        </figcaption>
      </figure>`;
    })
    .join("");

  grid.querySelectorAll(".name-it, .rename").forEach((el) => {
    el.onclick = () => startNaming(Number(el.dataset.id));
  });
  grid.querySelectorAll(".wrong").forEach((el) => {
    el.onclick = () => unname(Number(el.dataset.id));
  });

  const known = identities.map((i) => i.name);
  // Naming carries across sightings by face and by nothing else. Where that
  // cannot work - the model is absent, or the faces are too small - the panel
  // says so, rather than letting somebody name thirty crops expecting the
  // first one to have covered them.
  const howMatched = summary.faces_ready
    ? ""
    : " Face recognition is not installed, so a name applies to the one "
      + "sighting you put it on.";
  pq("people-note").innerHTML = (known.length
    ? `Known so far: <b>${known.map(peopleEscape).join("</b>, <b>")}</b>. `
      + "Naming another sighting teaches it a new face."
    : "Name someone once and every other sighting of them is found too. "
      + "Then you can ask when they arrived or left.") + howMatched;
}

/* ---------- naming ---------- */

function startNaming(trackletId) {
  const figure = document.querySelector(`.person[data-id="${trackletId}"]`);
  if (!figure || peopleState.editing === trackletId) return;
  peopleState.editing = trackletId;

  const caption = figure.querySelector("figcaption");
  const row = peopleState.rows.find((r) => r.id === trackletId);
  caption.innerHTML = `
    <form class="name-form">
      <label class="visually-hidden" for="name-${trackletId}">Name</label>
      <input id="name-${trackletId}" type="text" maxlength="64" autocomplete="off"
             placeholder="Their name" value="${peopleEscape(row && row.identity_name)}">
      <div class="name-actions">
        <button type="submit" class="tiny">Save</button>
        <button type="button" class="ghost tiny cancel">Cancel</button>
      </div>
    </form>`;

  const input = caption.querySelector("input");
  input.focus();
  input.select();

  caption.querySelector(".cancel").onclick = () => {
    peopleState.editing = null;
    loadPeople();
  };
  caption.querySelector("form").onsubmit = async (event) => {
    event.preventDefault();
    await saveName(trackletId, input.value.trim());
  };
}

async function saveName(trackletId, name) {
  if (!name) return;

  const response = await fetch("/api/identities/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracklet_id: trackletId, name }),
  });
  const body = await response.json();
  peopleState.editing = null;

  if (!response.ok) {
    notice(body.detail || "That name could not be saved.");
    return loadPeople();
  }

  // Naming teaches the gallery; matching then finds the same person
  // elsewhere. Doing it here means one name covers every sighting.
  //
  // Which kind was learned decides what can honestly be claimed. A face is
  // still that person next week; an appearance vector is a jacket. Nothing
  // writes appearance vectors today - see STORED_KIND in identity.py for why
  // the obvious candidate is not one - so this only ever takes the face
  // branch. The body branch stays because a real re-identification model
  // would land here, and its wording already says "a guess".
  const kinds = body.kinds || [];
  const kind = kinds.includes("face") ? "face" : (kinds.includes("body") ? "body" : null);
  const named = peopleEscape(name);

  let matched = 0;
  if (kind) {
    const assigned = await fetch(`/api/identities/assign?kind=${kind}`, { method: "POST" })
      .then((r) => r.json())
      .catch(() => ({}));
    matched = assigned.assigned || 0;
  }

  if (!kind) {
    // A face is the only thing that carries a name to another sighting, so
    // without one this labels the crop in front of you and nothing else.
    // Saying "done" here would promise matching that will never happen.
    notice(
      `Named ${named} for this sighting only &mdash; no clear face in this crop, `
      + "so their other appearances will not be found automatically.",
      "warn",
    );
  } else if (!matched) {
    notice(`Named ${named}. No other sighting looked like the same person.`, "good");
  } else if (kind === "face") {
    notice(
      `Named ${named} &middot; recognised their face in ${matched} other `
      + `sighting${matched === 1 ? "" : "s"}`,
      "good",
    );
  } else {
    notice(
      `Named ${named} &middot; ${matched} other sighting${matched === 1 ? "" : "s"} `
      + "look like the same person. No face was clear enough here, so this is a "
      + "guess from their overall appearance &mdash; correct any that are wrong.",
      "warn",
    );
  }

  await loadPeople();
  if (typeof refreshLibrary === "function") refreshChrome(await refreshLibrary());
}

/** Take a wrong name off a sighting.
 *
 * The counterpart to guessing. Matching by appearance will sometimes be
 * wrong, and a wrong name is a confident wrong answer to every later question
 * about that person, so it has to be one click to undo.
 */
async function unname(trackletId) {
  const response = await fetch(`/api/objects/${trackletId}/identity`, { method: "DELETE" });
  if (!response.ok) {
    notice("That name could not be removed.");
  }
  await loadPeople();
  if (typeof refreshLibrary === "function") refreshChrome(await refreshLibrary());
}

/* ---------- wiring ---------- */

function initPeople() {
  const panel = pq("people-panel");
  const toggle = pq("people-toggle");

  const setOpen = async (open) => {
    peopleState.open = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
    if (open) await loadPeople();
  };

  toggle.onclick = () => setOpen(!peopleState.open);
  pq("people-close").onclick = () => setOpen(false);
  window.openPeople = () => setOpen(true);
  window.reloadPeople = () => (peopleState.open ? loadPeople() : Promise.resolve());
}
