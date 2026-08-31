"""
whisper_service.py

Transcribes a chunk's audio using faster-whisper, with per-segment
timestamps offset to the video's global timeline (not just chunk-local).

Carries forward two v1 fixes worth keeping:
- a trailing audio buffer so words aren't cut off at chunk boundaries
- a cached model instance instead of reloading per chunk
"""

from dataclasses import dataclass
from functools import lru_cache

from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # bump to "small"/"medium" if accuracy is more important than speed
TRAILING_BUFFER_SECONDS = 0.5


@dataclass
class TranscriptSegment:
    start: float  # global (video-relative) timestamp
    end: float
    text: str


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    # Cached so we only load the model once per process, not once per chunk.
    # cuBLAS DLL still not found even after installing nvidia-cublas-cu12/
    # nvidia-cudnn-cu12 via pip - forcing CPU to keep moving. Revisit CUDA
    # later if transcription speed becomes a real bottleneck.
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_chunks(chunk) -> list[TranscriptSegment]:
    """Transcribe a single VideoChunk's audio, returning segments with
    timestamps offset to the chunk's position in the full video."""
    model = _get_model()
    segments, _info = model.transcribe(chunk.audio_path, vad_filter=True)

    results = []
    for seg in segments:
        results.append(TranscriptSegment(
            start=chunk.start_time + seg.start,
            end=chunk.start_time + seg.end,
            text=seg.text.strip(),
        ))
    return results
