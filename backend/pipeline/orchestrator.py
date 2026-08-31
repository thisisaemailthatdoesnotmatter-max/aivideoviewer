"""
Orchestrator: wires together whisper_service, frame_extractor, vlm_client,
and the memory store into a single per-chunk processing loop.

Kept deliberately thin - all real logic lives in the individual modules
so each piece can be tested/swapped independently. This is the module
that fixes v1's biggest structural problem: memory compression is no
longer entangled with the analysis loop, it's a discrete step you can
disable or inspect on its own.
"""

from typing import Awaitable, Callable

from pipeline.whisper_service import transcribe_chunks
from pipeline.frame_extractor import extract_chunks
from pipeline.vlm_client import VLMClient
from memory.store import MemoryStore
from debug.chunk_logger import log_chunk

ProgressCallback = Callable[[dict], Awaitable[None]]

# Chunk length in seconds. Kept short so the VLM gets a manageable
# number of frames per call and so hallucination is easier to isolate
# to a specific chunk when it happens.
CHUNK_SECONDS = 30


async def process_video(video_path: str, send_progress: ProgressCallback, disable_context: bool = True, job_id: str = "_unscoped"):
    memory = MemoryStore()
    vlm = VLMClient(model="qwen2.5vl:7b")

    chunks = extract_chunks(video_path, chunk_seconds=CHUNK_SECONDS)
    total = len(chunks)

    await send_progress({"type": "progress", "chunk": 0, "total": total, "stage": "starting"})

    for i, chunk in enumerate(chunks, start=1):
        await send_progress({"type": "progress", "chunk": i, "total": total, "stage": "transcribing"})
        transcript = transcribe_chunks(chunk)

        await send_progress({"type": "progress", "chunk": i, "total": total, "stage": "analyzing"})
        # Per-chunk analysis defaults to NO memory_context. Isolation
        # testing confirmed the model anchors on prior-chunk text and
        # echoes it back instead of extracting new details from the
        # current transcript, even with an explicit "this is new, that is
        # old" prompt structure. Continuity is reconstructed later via
        # compress()/get_final_summary(), which only ever sees chunk
        # results, never re-feeds its own output back into a fresh
        # analysis call. The "enable context" checkbox exists for
        # regression testing an improved context strategy later
        # (e.g. a short rolling recap instead of full prior text).
        context = "" if disable_context else memory.get_context()
        result = await vlm.analyze_chunk(
            frames=chunk.frames,
            transcript=transcript,
            memory_context=context,
        )

        memory.add_short_term(chunk_index=i, result=result)
        log_chunk(job_id=job_id, chunk_index=i, transcript=transcript, result=result)

        await send_progress({
            "type": "chunk_result",
            "chunk": i,
            "total": total,
            "summary": result.summary,
            "frames_analyzed": len(chunk.frames),
        })

        # Compression is an explicit, isolated step - easy to disable
        # or swap for a no-op while debugging model behavior.
        if memory.should_compress():
            await memory.compress(vlm)

    final_summary = memory.get_final_summary()
    await send_progress({"type": "done", "final_summary": final_summary})
