"""
test_isolation.py

Standalone script to isolate whether growing memory context is causing
the model to give vague/generic summaries instead of specific ones.

Usage:
    python test_isolation.py <chunk_dir> [--with-context "prior summary text"]

Example - test chunk 3 with ZERO prior context:
    python test_isolation.py debug_output/chunk_0002

Example - test chunk 3 WITH the same prior context it had in the real run:
    python test_isolation.py debug_output/chunk_0002 --with-context "..."

This reads transcript.json from the given debug_output chunk folder and
re-runs analyze_chunk() directly against Ollama, bypassing the rest of
the pipeline entirely. Compare the two outputs to see whether context
size/content is what's degrading specificity.

Note: this test uses debug_output's saved transcript/response, not the
original frame images (those live in the uploads/*_chunks folder, not
debug_output). Point --frames-dir at the matching uploads chunk folder
if you want frames included; otherwise this runs text-only.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.vlm_client import VLMClient


@dataclass
class _Segment:
    start: float
    end: float
    text: str


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_dir", help="Path to a debug_output/chunk_XXXX folder")
    parser.add_argument("--frames-dir", default=None, help="Optional path to matching frame images (uploads/*_chunks/chunk_XXXX)")
    parser.add_argument("--with-context", default="", help="Prior context text to include (default: none)")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    transcript_data = json.loads((chunk_dir / "transcript.json").read_text())
    transcript = [_Segment(**s) for s in transcript_data]

    frames = []
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames = sorted(str(p) for p in frames_dir.glob("frame_*.jpg"))
        print(f"Using {len(frames)} frames from {frames_dir}")
    else:
        print("No --frames-dir given, running TEXT-ONLY (transcript only, no images)")

    vlm = VLMClient(model="qwen2.5vl:7b")

    print("\n=== Running with context ===")
    print(f"'{args.with_context}'" if args.with_context else "(none)")
    print("\n=== Transcript ===")
    for s in transcript:
        print(f"  [{s.start:.1f}s] {s.text}")

    result = await vlm.analyze_chunk(
        frames=frames,
        transcript=transcript,
        memory_context=args.with_context,
    )

    print("\n=== MODEL OUTPUT ===")
    print(result.summary)


if __name__ == "__main__":
    asyncio.run(main())
