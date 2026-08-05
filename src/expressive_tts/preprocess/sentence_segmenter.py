"""Rule-based sentence segmentation (v0).

A minimal deterministic splitter used until full Stanza-based sentence
analysis (preprocess/objectives.md Phase 3) is implemented. Splits on runs
of terminal punctuation while respecting protected spans (URLs, e-mails,
dates, decimals, abbreviations) from `protected_spans.py`, per readme.md
section 2 stages 3-4.
"""

from __future__ import annotations

import re

from expressive_tts.preprocess.protected_spans import find_protected_spans, is_protected
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan

PRODUCER = "sentence_segmenter_rule_based_v0"

_TERMINAL_PATTERN = re.compile(r"[.!?…]+")
_TRAILING_CLOSERS = "\"')]"


def segment(text: str) -> list[SentenceSpan]:
    if not text:
        return []

    protected = find_protected_spans(text)
    boundaries: list[int] = []
    for match in _TERMINAL_PATTERN.finditer(text):
        if is_protected(match.start(), protected):
            continue
        end = match.end()
        while end < len(text) and text[end] in _TRAILING_CLOSERS:
            end += 1
        boundaries.append(end)

    spans: list[SentenceSpan] = []
    start = 0
    for end in boundaries + [len(text)]:
        chunk = text[start:end]
        stripped = chunk.strip()
        if stripped:
            offset = start + (len(chunk) - len(chunk.lstrip()))
            spans.append(SentenceSpan(text=stripped, start=offset, end=offset + len(stripped)))
        start = end

    return spans


class SentenceSegmenterProcessor:
    name = "sentence_segmenter"
    version = PRODUCER
    provides = {"sentences"}
    requires = {"clean"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        document.sentence_spans = segment(document.clean_text or "")
