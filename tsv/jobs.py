"""Background work, with progress a person can watch.

Importing a video means motion segmentation, then detection, then embedding -
minutes of work on a long recording. Doing that inside a request would leave
the window frozen with no idea whether anything is happening, so jobs run on a
thread and report where they are.

Progress is reported per stage rather than as one number, because the stages
have wildly different costs and a single bar that crawls then leaps is worse
than no bar. Each stage gets a slice of the whole proportional to roughly what
it costs, and reports within it.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

JobStatus = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: JobStatus = "queued"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.created

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
        }


class Reporter:
    """Handed to a job so it can say where it has got to.

    `stage(name, share)` opens a slice of the overall bar; `step(fraction)`
    moves within it. The job never has to know how it fits into the whole.
    """

    def __init__(self, job: Job, lock: threading.Lock) -> None:
        self._job = job
        self._lock = lock
        self._base = 0.0
        self._share = 1.0

    def stage(self, name: str, share: float, message: str = "") -> None:
        with self._lock:
            self._base += self._share if self._job.stage else 0.0
            self._share = share
            self._job.stage = name
            self._job.message = message
            self._job.progress = min(1.0, self._base)

    def step(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self._job.progress = min(1.0, self._base + self._share * max(0.0, min(1.0, fraction)))
            if message:
                self._job.message = message

    def say(self, message: str) -> None:
        with self._lock:
            self._job.message = message


class JobRunner:
    """A handful of background jobs and their progress.

    Deliberately serial: the work is CPU-bound and already uses every core, so
    running two imports at once would make both slower and the progress
    meaningless. Later submissions queue.
    """

    def __init__(self, keep: int = 20) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._keep = keep

    def submit(self, kind: str, title: str, fn: Callable[[Reporter], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            for stale in self._order[: -self._keep]:
                self._jobs.pop(stale, None)
            self._order = self._order[-self._keep :]

        thread = threading.Thread(target=self._run, args=(job, fn), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, fn: Callable[[Reporter], dict]) -> None:
        # One at a time; the work saturates the machine as it is.
        with self._worker_lock:
            with self._lock:
                job.status = "running"
            reporter = Reporter(job, self._lock)
            try:
                result = fn(reporter)
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                traceback.print_exc()
                # The terminal state is written under one lock. Setting the
                # status and the finish time separately lets a poller see a
                # job that has failed but never finished, whose elapsed time
                # then counts upward forever.
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.message = job.error
                    job.finished = time.time()
            else:
                with self._lock:
                    job.result = result or {}
                    job.status = "done"
                    job.progress = 1.0
                    job.stage = "finished"
                    job.finished = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._order[-1]) if self._order else None

    def active(self) -> list[Job]:
        with self._lock:
            return [
                self._jobs[i] for i in self._order
                if self._jobs[i].status in ("queued", "running")
            ]

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)]
