"""
frame_extractor.py

Splits a video into fixed-length chunks and pulls evenly-spaced frames
from each, scaled down to keep VLM image-token costs low (this was the
fix that solved Qwen's token-cost blowup in v1 - keep it here).

Requires ffmpeg/ffprobe on PATH.
"""

import subprocess
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

FRAME_WIDTH = 512  # matches the v1 fix for Qwen image token costs
FRAMES_PER_CHUNK = 5


@dataclass
class VideoChunk:
    index: int
    start_time: float
    end_time: float
    audio_path: str
    frames: list[str] = field(default_factory=list)  # image file paths


def _probe_duration(video_path: str) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_chunks(video_path: str, chunk_seconds: int = 30) -> list[VideoChunk]:
    duration = _probe_duration(video_path)
    n_chunks = math.ceil(duration / chunk_seconds)

    work_dir = Path(video_path).parent / f"{Path(video_path).stem}_chunks"
    work_dir.mkdir(exist_ok=True)

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_seconds
        end = min(start + chunk_seconds, duration)
        chunk_dir = work_dir / f"chunk_{i:04d}"
        chunk_dir.mkdir(exist_ok=True)

        audio_path = chunk_dir / "audio.wav"
        _extract_audio(video_path, start, end - start, str(audio_path))

        frame_paths = _extract_frames(
            video_path, start, end - start, chunk_dir, FRAMES_PER_CHUNK
        )

        chunks.append(VideoChunk(
            index=i, start_time=start, end_time=end,
            audio_path=str(audio_path), frames=frame_paths,
        ))

    return chunks


def _extract_audio(video_path: str, start: float, length: float, out_path: str):
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
            "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", out_path,
        ],
        capture_output=True, check=True,
    )


def _extract_frames(video_path: str, start: float, length: float, out_dir: Path, count: int) -> list[str]:
    if count <= 0 or length <= 0:
        return []

    interval = length / count
    paths = []
    for j in range(count):
        t = start + j * interval
        frame_path = out_dir / f"frame_{j:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-1",
                str(frame_path),
            ],
            capture_output=True, check=True,
        )
        paths.append(str(frame_path))
    return paths
