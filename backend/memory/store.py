"""
store.py

Short-term + long-term memory for the analysis loop.

Key structural fix vs v1: compression is an explicit, separately-callable
step (`compress()`) rather than something baked into the analysis loop.
That entanglement is what let the Gemma hallucination bug become
self-reinforcing - a hallucinated detail would get folded into a summary,
which then became "prior context" for the next chunk, and the model
treated its own earlier hallucination as ground truth.

To guard against that here:
- compression only ever summarizes what chunks actually reported, via a
  dedicated summarizer prompt that's told not to add new details
  (see vlm_client.summarize)
- short-term memory is capped and FIFO, so a single bad chunk result
  ages out rather than permanently anchoring later context
"""

from dataclasses import dataclass, field

SHORT_TERM_CAP = 5  # chunks kept verbatim before compression is triggered


@dataclass
class ChunkRecord:
    chunk_index: int
    summary: str


class MemoryStore:
    def __init__(self):
        self.short_term: list[ChunkRecord] = []
        self.long_term_summary: str = ""

    def add_short_term(self, chunk_index: int, result) -> None:
        self.short_term.append(ChunkRecord(chunk_index=chunk_index, summary=result.summary))

    def should_compress(self) -> bool:
        return len(self.short_term) >= SHORT_TERM_CAP

    async def compress(self, vlm) -> None:
        """Fold short-term memory into the long-term summary via a
        dedicated summarizer call, then clear short-term memory."""
        notes = "\n".join(f"Chunk {r.chunk_index}: {r.summary}" for r in self.short_term)
        combined = f"{self.long_term_summary}\n\n{notes}".strip()
        self.long_term_summary = await vlm.summarize(combined)
        self.short_term = []

    def get_context(self) -> str:
        """Context handed to the VLM before analyzing the next chunk."""
        recent = "\n".join(f"Chunk {r.chunk_index}: {r.summary}" for r in self.short_term)
        parts = [p for p in [self.long_term_summary, recent] if p]
        return "\n\n".join(parts)

    def get_final_summary(self) -> str:
        recent = "\n".join(f"Chunk {r.chunk_index}: {r.summary}" for r in self.short_term)
        parts = [p for p in [self.long_term_summary, recent] if p]
        return "\n\n".join(parts) or "(no content processed)"
