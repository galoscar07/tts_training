"""Shared helper for regex-driven text transformations that record a trace."""

from __future__ import annotations

import re
from typing import Callable

from expressive_tts.preprocess.schemas import Provenance, TraceEntry


def apply_substitution(
    text: str,
    pattern: re.Pattern,
    replace: Callable[[re.Match], str],
    *,
    stage: str,
    operation: str,
    producer: str,
    provenance: Provenance = Provenance.RULE,
    confidence: float | None = 1.0,
) -> tuple[str, list[TraceEntry]]:
    """Apply `pattern` across `text`, replacing each match via `replace`.

    Returns the transformed text and one `TraceEntry` per match whose
    replacement differs from the original span. `start`/`end` on each entry
    are offsets into `text` as passed in (before this call).
    """
    entries: list[TraceEntry] = []
    chunks: list[str] = []
    last_end = 0

    for match in pattern.finditer(text):
        chunks.append(text[last_end : match.start()])
        original = match.group(0)
        replacement = replace(match)
        if replacement != original:
            entries.append(
                TraceEntry(
                    stage=stage,
                    operation=operation,
                    original=original,
                    replacement=replacement,
                    start=match.start(),
                    end=match.end(),
                    producer=producer,
                    provenance=provenance,
                    confidence=confidence,
                )
            )
        chunks.append(replacement)
        last_end = match.end()

    chunks.append(text[last_end:])
    return "".join(chunks), entries
