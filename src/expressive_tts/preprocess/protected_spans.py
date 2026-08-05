"""Detection of spans that must not be split on during sentence segmentation:
URLs, e-mail addresses, dates, decimal numbers, and known abbreviations.
See readme.md section 2 (stage 3) and preprocess/objectives.md Phase 2.
"""

from __future__ import annotations

import re

_URL_PATTERN = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DATE_PATTERN = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_DECIMAL_PATTERN = re.compile(r"\b\d+[.,]\d+\b")

# Minimal set of abbreviations that end in a period but never end a sentence.
# Intentionally separate from configs/preprocess/abbreviations.yaml (which
# drives spoken-form expansion in normalizer.py) since this list only needs
# to protect periods, not know how the abbreviation is pronounced.
DEFAULT_ABBREVIATIONS = {
    "dr",
    "d-na",
    "d-nul",
    "prof",
    "ing",
    "nr",
    "art",
    "pag",
    "str",
    "bd",
    "sec",
    "min",
    "max",
    "etc",
    "ex",
    "cf",
    "vol",
    "cap",
    "fig",
}


def _abbreviation_pattern(abbreviations: set[str]) -> re.Pattern:
    escaped = sorted((re.escape(a) for a in abbreviations), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\.", re.IGNORECASE)


def find_protected_spans(
    text: str, abbreviations: set[str] | None = None
) -> list[tuple[int, int]]:
    """Return merged, non-overlapping (start, end) spans that must not be
    treated as containing a sentence boundary, sorted by start offset."""
    abbreviation_pattern = _abbreviation_pattern(abbreviations or DEFAULT_ABBREVIATIONS)
    spans: list[tuple[int, int]] = []
    for pattern in (
        _URL_PATTERN,
        _EMAIL_PATTERN,
        _DATE_PATTERN,
        _DECIMAL_PATTERN,
        abbreviation_pattern,
    ):
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def is_protected(offset: int, spans: list[tuple[int, int]]) -> bool:
    """True if `offset` falls strictly inside one of `spans`."""
    return any(start <= offset < end for start, end in spans)
