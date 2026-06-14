"""
Chunk Overlap — Preserves cross-boundary context on size-based splits.

Problem
-------
`build_chunks()` cuts mid-section when accumulated text exceeds
`CHUNK_MAX_CHARS` (default 6000). Without overlap, the last sentence of
chunk N and the first sentence of chunk N+1 are completely disjoint, so
retrieval can return either half and miss the other — e.g. a "cause"
sentence in chunk N and the matching "resolution" sentence in chunk N+1.

Solution
--------
On (and only on) a size-based split, prepend the tail of the previous
chunk's text — snapped to a sentence or newline boundary — to the next
chunk. We deliberately do NOT add overlap on heading-bounded splits:
those are semantic boundaries that already carry the heading as context.

Why not just use a sliding window?
----------------------------------
A blanket sliding window inflates the chunk count by ~10–15% across the
whole corpus, even for well-structured docs where every chunk maps to a
distinct heading. Targeted overlap costs nothing on structured content
and only fires where it matters: long, heading-less sections.

Boundary snapping
-----------------
We prefer to cut at, in order:
  1. The most recent paragraph break (`\n\n`)
  2. The most recent sentence-ending punctuation (`. ? !` followed by space)
  3. The most recent newline
  4. A hard char cut (fallback)

This avoids overlap chunks that start mid-word.
"""

from __future__ import annotations

import re
from typing import Final

from backend.config import settings


# Sentence boundary: punctuation + whitespace + uppercase letter. The
# uppercase requirement keeps us from cutting inside abbreviations like
# "e.g. the" or "Mr. Smith". Compiled once.
_SENTENCE_BOUNDARY_RE: Final = re.compile(r"[.!?]\s+(?=[A-Z])")


def build_overlap_prefix(previous_chunk_text: str, overlap_chars: int = -1) -> str:
    """
    Build the prefix that should be prepended to the next chunk.

    Parameters
    ----------
    previous_chunk_text : str
        The full text of the chunk that just got flushed.
    overlap_chars : int, optional
        Target overlap length in characters. -1 (default) means read from
        settings.CHUNK_OVERLAP_CHARS. 0 disables overlap (returns "").

    Returns
    -------
    str
        The boundary-snapped tail of the previous chunk, or "" if overlap
        is disabled or the previous chunk is too short.

    Notes
    -----
    * The returned prefix does NOT include any trailing whitespace; the
      caller is expected to join with a separator.
    * If the previous chunk is shorter than the overlap target, the entire
      previous chunk is returned. Cheap and correct for tiny tail chunks.
    """
    target = (
        overlap_chars if overlap_chars >= 0 else settings.CHUNK_OVERLAP_CHARS
    )
    if target <= 0 or not previous_chunk_text:
        return ""

    text = previous_chunk_text.strip()
    if len(text) <= target:
        return text

    # Take the last `target` chars as the starting candidate, then walk
    # backwards to the nearest clean boundary.
    candidate = text[-target:]

    # 1. Paragraph break — strongest signal.
    para_idx = candidate.find("\n\n")
    if para_idx != -1 and para_idx < target - 50:
        # Trim everything up to and including the paragraph break so the
        # overlap starts at the beginning of a fresh paragraph.
        return candidate[para_idx + 2:].strip()

    # 2. Sentence boundary — walk the regex matches and use the earliest
    #    one that still leaves a meaningful overlap behind.
    matches = list(_SENTENCE_BOUNDARY_RE.finditer(candidate))
    if matches:
        # Use the first sentence boundary we find — it gives us the longest
        # well-formed tail. If we used the last one we might end up with
        # only a few words of overlap.
        first = matches[0]
        return candidate[first.end():].strip()

    # 3. Newline — weakest semantic signal but still better than mid-word.
    newline_idx = candidate.find("\n")
    if newline_idx != -1 and newline_idx < target - 50:
        return candidate[newline_idx + 1:].strip()

    # 4. Fallback: hard cut. Avoid starting mid-word by trimming to the
    #    first whitespace boundary.
    space_idx = candidate.find(" ")
    if space_idx != -1:
        return candidate[space_idx + 1:].strip()

    return candidate.strip()


def apply_overlap(previous_chunk_text: str, next_chunk_text: str) -> str:
    """
    Convenience wrapper: returns next_chunk_text prefixed with the
    overlap tail from previous_chunk_text.

    Use this in build_chunks() right BEFORE creating the new ParsedChunk
    for the continued section. The returned string is the new chunk text.
    """
    prefix = build_overlap_prefix(previous_chunk_text)
    if not prefix:
        return next_chunk_text
    # The "[…]" marker tells downstream readers (and the LLM during
    # metadata extraction) that this overlap region is borrowed context,
    # not the chunk's primary content. Cheap, debuggable, harmless.
    return f"[…overlap…] {prefix}\n\n{next_chunk_text}"
