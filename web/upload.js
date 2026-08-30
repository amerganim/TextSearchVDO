/* Sending a video from a phone, a chunk at a time.
 *
 * The single POST this replaces asked a phone to hold one connection open for
 * the whole transfer. At the ~12 MB/s a laptop manages over WiFi, a four
 * gigabyte recording is five minutes of that, and everything a phone does
 * normally - locking the screen, switching apps, walking towards the door -
 * ends it. The user's reward was starting again from zero.
 *
 * Here the connection only has to survive one chunk. Anything longer is
 * retried, and the *server* says where to carry on, so this never has to
 * guess: a reply that was lost after the write landed comes back as "you are
 * further along than you think" rather than as a duplicate.
 */

const UPLOAD_RETRIES = 6;

/** Wait between attempts, backing off but never past a few seconds.
 *
 * A phone that has walked out of range comes back suddenly, so the ceiling
 * matters more than the curve: waiting a minute would mean the transfer sits
 * idle long after the WiFi returned.
 */
const backoff = (attempt) => Math.min(500 * 2 ** attempt, 4000);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class UploadCancelled extends Error {}

/** Send one file, resuming whatever is already on the server.
 *
 * `onProgress(sent, total)` is called per chunk. `signal` cancels it.
 */
async function uploadFile(file, { onProgress, signal } = {}) {
  const throwIfCancelled = () => {
    if (signal && signal.aborted) throw new UploadCancelled();
  };

  throwIfCancelled();
  const begun = await fetch("/api/upload/begin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: file.name, size: file.size }),
  });

  if (!begun.ok) {
    const body = await begun.json().catch(() => ({}));
    // 507 is out of disk. Worth its own words: nothing the user does to the
    // file or the network will fix it.
    throw new Error(body.detail || `Could not start the upload (${begun.status}).`);
  }

  let state = await begun.json();
  const chunkBytes = state.chunk_bytes || 8 << 20;
  if (onProgress) onProgress(state.offset, file.size);

  while (state.offset < file.size) {
    throwIfCancelled();
    const start = state.offset;
    const slice = file.slice(start, Math.min(start + chunkBytes, file.size));

    let sent = null;
    for (let attempt = 0; attempt < UPLOAD_RETRIES; attempt += 1) {
      throwIfCancelled();
      try {
        const response = await fetch(`/api/upload/${state.id}`, {
          method: "PUT",
          headers: { "X-Upload-Offset": String(start) },
          body: slice,
          signal,
        });
        if (response.status === 404) {
          throw new Error("The computer forgot this upload. Start it again.");
        }
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `Upload failed (${response.status}).`);
        }
        sent = await response.json();
        break;
      } catch (err) {
        if (err instanceof UploadCancelled || (signal && signal.aborted)) {
          throw new UploadCancelled();
        }
        // A dropped connection is the expected case, not an exception: keep
        // trying, and let the last failure through if it never recovers.
        if (attempt === UPLOAD_RETRIES - 1) throw err;
        await sleep(backoff(attempt));
      }
    }

    if (sent.offset <= state.offset && sent.offset < file.size) {
      // No forward progress despite a success. Rather than spin, stop and say
      // so - continuing would be an infinite loop wearing a progress bar.
      throw new Error("The upload stopped making progress. Try again.");
    }
    state = sent;
    if (onProgress) onProgress(state.offset, file.size);
  }

  throwIfCancelled();
  const finished = await fetch(`/api/upload/${state.id}/finish`, { method: "POST" });
  if (!finished.ok) {
    const body = await finished.json().catch(() => ({}));
    throw new Error(body.detail || "The computer could not finish the upload.");
  }
  return finished.json();
}

/** Send several files in turn, reporting one overall progress.
 *
 * In turn rather than at once on purpose: phones and home routers both do
 * worse with parallel large transfers, and a queue that finishes the first
 * video quickly means indexing can start while the second is still arriving.
 */
async function uploadFiles(files, { onProgress, onFileDone, signal } = {}) {
  const list = [...files];
  const total = list.reduce((sum, file) => sum + file.size, 0);
  let done = 0;
  const jobs = [];

  for (const [index, file] of list.entries()) {
    const job = await uploadFile(file, {
      signal,
      onProgress: (sent) => {
        if (onProgress) {
          onProgress({
            file, index, count: list.length,
            sent: done + sent, total,
            fraction: total ? (done + sent) / total : 0,
          });
        }
      },
    });
    done += file.size;
    jobs.push(job);
    if (onFileDone) onFileDone(job, file);
  }
  return jobs;
}
