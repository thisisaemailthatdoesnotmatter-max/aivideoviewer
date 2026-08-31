"""
AIVideoViewer v2 - FastAPI backend entrypoint.

Serves:
- REST endpoint to upload a video, which starts processing IMMEDIATELY
  as a background task (not tied to any WebSocket connection)
- WebSocket endpoint to watch a job's progress live - purely a viewer,
  can attach/detach/reattach without affecting the job itself
- REST endpoint to poll a job's current state without a WebSocket
- REST endpoint to chat about a finished (or in-progress) video
- Static frontend (HTML/JS) for the localhost UI

Run with:
    uvicorn main:app --reload --port 8000
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jobs import job_manager
import chat as chat_module

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AIVideoViewer v2")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), disable_context: bool = Form(True)):
    """Accept a video upload and start processing right away. The job
    runs as a background task independent of any WebSocket - closing
    the browser tab does NOT stop it. Returns job_id immediately;
    connect a WebSocket to /ws/{job_id} to watch it, or poll
    /jobs/{job_id}."""
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    job_manager.start_job(job_id, str(dest), disable_context=disable_context)

    return {"job_id": job_id, "filename": file.filename}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll a job's current state without needing a WebSocket - useful
    for reconnecting after closing the tab, or checking on a job from
    a fresh page load."""
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@app.websocket("/ws/{job_id}")
async def ws_watch(websocket: WebSocket, job_id: str):
    """Attach to an already-running (or already-finished) job and
    stream its events. Does NOT start or stop the job - purely a
    viewer. On connect, replays everything that already happened
    (backlog), then streams live updates until the job finishes or
    the client disconnects."""
    await websocket.accept()

    state = job_manager.get_state(job_id)
    if state is None:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    # Replay backlog so a late-joining or reconnecting client catches up.
    await websocket.send_json({"type": "progress", "chunk": state.current_chunk, "total": state.total, "stage": state.stage})
    for c in state.chunks:
        await websocket.send_json({"type": "chunk_result", **c})

    if state.status == "done":
        await websocket.send_json({"type": "done", "final_summary": state.final_summary})
        return
    if state.status == "error":
        await websocket.send_json({"type": "error", "message": state.error})
        return

    # Job still running - subscribe for live updates.
    queue = job_manager.subscribe(job_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
            if event.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, queue)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat/{job_id}")
async def chat_endpoint(job_id: str, req: ChatRequest):
    state = job_manager.get_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    answer = await chat_module.answer_question(state, req.question)
    return {"answer": answer}
