"""
chunk_logger.py

Writes per-chunk transcript + model response to disk for inspection,
scoped per job_id (job_storage/{job_id}/chunks/...) so multiple videos
processed back-to-back don't collide on the same debug folder. Useful
for diffing runs, catching regressions, or feeding test_isolation.py.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

DEBUG_BASE = Path(__file__).resolve().parent.parent / "job_storage"


def log_chunk(job_id: str, chunk_index: int, transcript, result) -> None:
    chunk_dir = DEBUG_BASE / job_id / "chunks" / f"chunk_{chunk_index:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    (chunk_dir / "transcript.json").write_text(
        json.dumps([{"start": s.start, "end": s.end, "text": s.text} for s in transcript], indent=2),
        encoding="utf-8",
    )
    (chunk_dir / "response.txt").write_text(result.raw_response, encoding="utf-8")
    (chunk_dir / "meta.json").write_text(
        json.dumps({"chunk_index": chunk_index, "logged_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
