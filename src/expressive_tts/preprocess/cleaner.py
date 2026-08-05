"""Unicode cleanup: NFC normalization, legacy Romanian diacritics, whitespace,
and quote/apostrophe normalization. See readme.md section 2 (stage 2) and
preprocess/objectives.md Phase 2.
"""

from __future__ import annotations

import re
import unicodedata

from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.trace_utils import apply_substitution
from expressive_tts.preprocess.schemas import TraceEntry

PRODUCER = "cleaner_v1"

# Legacy pre-Unicode-3.0 Romanian letters (cedilla) -> correct comma-below forms.
_LEGACY_DIACRITICS = {
    "ş": "ș",  # ş -> ș
    "Ş": "Ș",  # Ş -> Ș
    "ţ": "ț",  # ţ -> ț
    "Ţ": "Ț",  # Ţ -> Ț
}
_LEGACY_PATTERN = re.compile("[" + "".join(_LEGACY_DIACRITICS) + "]")

_QUOTE_MAP = {
    "“": '"',  # “
    "”": '"',  # ”
    "„": '"',  # „
    "«": '"',  # «
    "»": '"',  # »
    "‘": "'",  # ‘
    "’": "'",  # ’
    "ʼ": "'",  # ʼ
}
_QUOTE_PATTERN = re.compile("[" + "".join(_QUOTE_MAP) + "]")

_WHITESPACE_PATTERN = re.compile(r"\s+")


class CleanerProcessor:
    name = "cleaner"
    version = PRODUCER
    provides = {"clean"}
    requires: set[str] = set()

    def process(self, document: PipelineDocument, config: dict) -> None:
        text = unicodedata.normalize("NFC", document.original_text)

        text, entries = apply_substitution(
            text,
            _LEGACY_PATTERN,
            lambda m: _LEGACY_DIACRITICS[m.group(0)],
            stage=self.name,
            operation="legacy_diacritics",
            producer=PRODUCER,
        )
        document.trace.extend(entries)

        text, entries = apply_substitution(
            text,
            _QUOTE_PATTERN,
            lambda m: _QUOTE_MAP[m.group(0)],
            stage=self.name,
            operation="quote_normalization",
            producer=PRODUCER,
        )
        document.trace.extend(entries)

        text, entries = apply_substitution(
            text,
            _WHITESPACE_PATTERN,
            lambda m: " ",
            stage=self.name,
            operation="whitespace",
            producer=PRODUCER,
        )
        document.trace.extend(entries)

        stripped = text.strip()
        if stripped != text:
            document.trace.append(
                TraceEntry(
                    stage=self.name,
                    operation="strip",
                    original=text,
                    replacement=stripped,
                    start=0,
                    end=len(text),
                    producer=PRODUCER,
                    confidence=1.0,
                )
            )

        document.clean_text = stripped
