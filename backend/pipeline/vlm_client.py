"""
vlm_client.py

Thin wrapper around Ollama's chat API for vision-language analysis of a
chunk (frames + transcript + prior memory context).

Deliberately model-agnostic: swap `model=` to A/B test different VLMs
against the same video, since model behavior (e.g. the v1 Gemma 3 4B
narrative-momentum hallucination bug) was the actual crux of v1's
problems, not the surrounding pipeline code.
"""

import base64
from dataclasses import dataclass

import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"

# Grounding-focused system prompt - keep this explicit and boring.
# The Gemma hallucination bug got worse the more "creative" prompts got.
SYSTEM_PROMPT = (
    "You are analyzing real video footage. Describe only what is "
    "directly visible in the provided frames and audible in the "
    "transcript. Do not invent scenes, objects, or events that are not "
    "clearly present. If uncertain, say so explicitly rather than "
    "guessing."
)


@dataclass
class ChunkAnalysis:
    summary: str
    raw_response: str


class VLMClient:
    def __init__(self, model: str = "qwen2.5vl:7b", num_ctx: int = 8192):
        self.model = model
        self.num_ctx = num_ctx  # kept at 8192 for VRAM safety on a 3060

    async def analyze_chunk(self, frames: list[str], transcript, memory_context: str) -> ChunkAnalysis:
        images_b64 = [self._encode_image(p) for p in frames]
        transcript_text = "\n".join(f"[{s.start:.1f}s] {s.text}" for s in transcript) or "(no speech)"

        user_content = (
            "=== PRIOR CONTEXT (already summarized, already reported - "
            "do NOT repeat this, it is only background) ===\n"
            f"{memory_context or '(none yet)'}\n\n"
            "=== CURRENT TRANSCRIPT (this is the NEW segment you must "
            "describe - this is what is happening NOW) ===\n"
            f"{transcript_text}\n\n"
            "Describe what happens in THIS segment only, grounded strictly "
            "in the frames and CURRENT TRANSCRIPT above. Your description "
            "MUST explicitly cover both: (1) what is visually shown, and "
            "(2) what is said in the CURRENT TRANSCRIPT, if any dialogue "
            "is present there. Be specific - use actual names, phrases, "
            "and details from the CURRENT TRANSCRIPT rather than general "
            "descriptions like 'political figures' or 'expressing "
            "frustration'.\n\n"
            "If the visual framing (e.g. camera angle, subject, setting) "
            "is similar to what was already described in prior context, "
            "do not re-describe it at length - note briefly that it's "
            "unchanged and focus your description on what is NEW in this "
            "segment. In a talking-head style video, the dialogue usually "
            "changes far more than the visuals between segments, so "
            "dialogue content should typically dominate the summary "
            "unless something visually new actually happens."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content, "images": images_b64},
            ],
            "options": {"num_ctx": self.num_ctx},
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        return ChunkAnalysis(summary=text, raw_response=text)

    async def summarize(self, long_text: str) -> str:
        """Dedicated compression call - kept separate from analyze_chunk
        so memory compression never shares a prompt/context with frame
        analysis (this entanglement was what fed the v1 hallucination
        feedback loop)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Summarize the following notes concisely and factually. Do not add new details."},
                {"role": "user", "content": long_text},
            ],
            "options": {"num_ctx": self.num_ctx},
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
