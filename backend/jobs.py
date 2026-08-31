"""
jobs.py

In-process job manager that decouples video processing from any single
WebSocket connection. Fixes the "closed the tab and lost an hour of
processing" problem:

- A job starts running the instant it's created (on upload), as a
  background asyncio task - not when a WebSocket connects.
- Every event (progress, chunk_result, done, error) is checkpointed to
  disk immediately, so a completed job survives a server restart too.
- Any number of WebSocket clients can attach/reattach to *watch* a job
  without affecting whether it keeps running. A late-joining client
  gets a backlog replay of everything that already happened, then live
  updates from there.
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from pipeline.orchestrator import process_video

JOBS_DIR = Path(__file__).resolve().parent / "job_storage"
JOBS_DIR.mkdir(exist_ok=True)


@dataclass
class JobState:
    job_id: str
    video_path: str
    status: str = "running"  # running | done | error
    total: int = 0
    stage: str = "starting"
    current_chunk: int = 0
    chunks: list = field(default_factory=list)  # list of {chunk, total, summary, frames_analyzed}
    final_summary: Optional[str] = None
    error: Optional[str] = None


class JobManager:
    def __init__(self):
        self._jobs: dict[str, JobState] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def _checkpoint_path(self, job_id: str) -> Path:
        return JOBS_DIR / f"{job_id}.json"

    def _save_checkpoint(self, state: JobState) -> None:
        self._checkpoint_path(state.job_id).write_text(
            json.dumps(asdict(state), indent=2), encoding="utf-8"
        )

    def start_job(self, job_id: str, video_path: str, disable_context: bool = True) -> None:
        """Kick off processing immediately - does NOT wait for a WebSocket."""
        state = JobState(job_id=job_id, video_path=video_path)
        self._jobs[job_id] = state
        self._subscribers[job_id] = []
        self._save_checkpoint(state)
        asyncio.create_task(self._run(state, disable_context))

    async def _run(self, state: JobState, disable_context: bool) -> None:
        async def send_progress(event: dict):
            self._apply_event(state, event)
            self._save_checkpoint(state)
            await self._broadcast(state.job_id, event)

        try:
            await process_video(
                state.video_path, send_progress,
                disable_context=disable_context, job_id=state.job_id,
            )
        except Exception as e:
            state.status = "error"
            state.error = str(e)
            self._save_checkpoint(state)
            await self._broadcast(state.job_id, {"type": "error", "message": str(e)})

    def _apply_event(self, state: JobState, event: dict) -> None:
        etype = event.get("type")
        if etype == "progress":
            state.current_chunk = event.get("chunk", state.current_chunk)
            state.total = event.get("total", state.total)
            state.stage = event.get("stage", state.stage)
        elif etype == "chunk_result":
            state.chunks.append({
                "chunk": event["chunk"],
                "total": event["total"],
                "summary": event["summary"],
                "frames_analyzed": event["frames_analyzed"],
            })
        elif etype == "done":
            state.status = "done"
            state.final_summary = event.get("final_summary")
        elif etype == "error":
            state.status = "error"
            state.error = event.get("message")

    async def _broadcast(self, job_id: str, event: dict) -> None:
        for q in list(self._subscribers.get(job_id, [])):
            await q.put(event)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)

    def get_state(self, job_id: str) -> Optional[JobState]:
        """Look up job state - checks memory first, falls back to the
        on-disk checkpoint (e.g. after a server restart)."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        path = self._checkpoint_path(job_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            state = JobState(**data)
            self._jobs[job_id] = state
            self._subscribers.setdefault(job_id, [])
            return state
        return None


job_manager = JobManager()
