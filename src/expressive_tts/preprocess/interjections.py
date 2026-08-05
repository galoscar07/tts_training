"""Controlled interjection suggestions, and document-style detection to
gate them. See readme.md section 1.9/2.4 and preprocess/objectives.md
Phase 10.

Distinct from Phase 6, which *detects* interjections already present in
the input (`Token.is_interjection`, used as emotion evidence) — this phase
*proposes* new ones, and never applies them to `Sentence.text` itself
(objectives.md: "Preserve the unmodified text").

Modes (objectives.md): "disabled" (default — no suggestions at all),
"suggest" (populate `Sentence.interjection_suggestions`), "insert" (also
populate `Sentence.text_with_interjections`, an enriched variant).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from expressive_tts.preprocess.lexicons import default_interjection_emotions
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import InterjectionSuggestion

PRODUCER = "interjections_rules_v1"

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"

MIN_EMOTION_CONFIDENCE = 0.6
MAX_SUGGESTIONS = 2

# Document-style detection: a deliberately coarse formal/conversational
# binary (not the full academic/legal/technical/formal taxonomy
# objectives.md's prose mentions — without labeled formality data, a finer
# split would be unfounded guessing, and the actual feature requirement
# only needs a binary gate). See `detect_document_style` for the scoring
# rule (a formal-marker hit is authoritative; the other three signals need
# unanimous agreement).
LONG_SENTENCE_TOKEN_COUNT = 12


@lru_cache(maxsize=None)
def default_formal_markers() -> tuple[str, ...]:
    path = _CONFIG_DIR / "formal_markers.yaml"
    with path.open(encoding="utf-8") as handle:
        markers = yaml.safe_load(handle) or []
    return tuple(m.lower() for m in markers)


def _sentence_has_existing_interjection(span: SentenceSpan, interjection_emotions: dict) -> bool:
    return any(
        (token.lemma or token.text).lower() in interjection_emotions
        for token in (span.tokens or [])
        if token.upos != "PUNCT"
    )


def detect_document_style(
    sentence_spans: list[SentenceSpan],
    formal_markers: tuple[str, ...],
    interjection_emotions: dict,
) -> tuple[str, float, list[str]]:
    """Coarse formal/conversational classification, aggregated across the
    whole document (not per-sentence). Returns (style, score, reasons).

    `formal_markers` hits are treated as authoritative on their own (a
    legal/administrative phrase is strong, unambiguous evidence). The other
    three signals are individually weak/fragile — e.g. Romanian is
    pro-drop and Stanza can mis-tag syncretic verb forms like "sunt" as
    3rd-person-plural instead of 1st-person-singular with no explicit
    subject to disambiguate (verified directly on "Sunt extrem de
    fericit!") — so they only count toward "formal" if *all three* agree;
    a flat equal-weighted average let one fragile signal combine with
    "no existing interjection" (true for most short sentences regardless
    of register) to misclassify that clearly casual sentence.
    """
    if not sentence_spans:
        return "conversational", 0.0, []

    reasons: list[str] = []
    weak_signals = 0

    lengths = [
        sum(1 for t in (span.tokens or []) if t.upos != "PUNCT") for span in sentence_spans
    ]
    avg_length = sum(lengths) / len(lengths)
    if avg_length > LONG_SENTENCE_TOKEN_COUNT:
        weak_signals += 1
        reasons.append(f"long_average_sentence_length:{avg_length:.1f}")

    has_person_markers = any(
        token.feats.get("Person") in ("1", "2")
        for span in sentence_spans
        for token in (span.tokens or [])
    )
    if not has_person_markers:
        weak_signals += 1
        reasons.append("no_first_or_second_person_markers")

    has_interjection = any(
        _sentence_has_existing_interjection(span, interjection_emotions) for span in sentence_spans
    )
    if not has_interjection:
        weak_signals += 1
        reasons.append("no_existing_interjections")

    full_text = " ".join((span.normalized_text or span.text) for span in sentence_spans).lower()
    marker_hits = [marker for marker in formal_markers if marker in full_text]
    if marker_hits:
        reasons.append(f"formal_markers:{','.join(marker_hits)}")

    is_formal = bool(marker_hits) or weak_signals == 3
    score = 1.0 if marker_hits else weak_signals / 3
    style = "formal" if is_formal else "conversational"
    return style, score, reasons


def _ranked_candidates(label: str, interjection_emotions: dict) -> list[tuple[str, float]]:
    matches = [
        (word, entry["weight"]) for word, entry in interjection_emotions.items() if entry["label"] == label
    ]
    matches.sort(key=lambda pair: -pair[1])
    return matches


def suggest_for_sentence(
    span: SentenceSpan,
    *,
    mode: str,
    document_style: str,
    interjection_emotions: dict,
    min_confidence: float = MIN_EMOTION_CONFIDENCE,
    max_suggestions: int = MAX_SUGGESTIONS,
) -> None:
    """Populate `span.interjection_suggestions` (and, in "insert" mode,
    `span.text_with_interjections`), in place. No-op unless every gate
    passes."""
    if mode == "disabled":
        return
    if document_style == "formal":
        return
    if _sentence_has_existing_interjection(span, interjection_emotions):
        return

    emotion = span.emotion
    if emotion is None or emotion.confidence < min_confidence:
        return

    candidates = _ranked_candidates(emotion.label, interjection_emotions)
    if not candidates:
        return

    suggestions = [
        InterjectionSuggestion(
            text=word.capitalize(),
            position=0,
            reason=f"emotion_match:{emotion.label}",
            confidence=round(weight * emotion.confidence, 3),
            matched_emotion=emotion.label,
        )
        for word, weight in candidates[:max_suggestions]
    ]
    span.interjection_suggestions = suggestions

    if mode == "insert":
        original = span.normalized_text or span.text
        rest = original[0].lower() + original[1:] if original else original
        span.text_with_interjections = f"{suggestions[0].text}, {rest}"


class InterjectionProcessor:
    name = "interjections"
    version = PRODUCER
    provides = {"interjections"}
    requires = {"emotion"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        mode = config.get("interjection_mode", "disabled")
        interjection_emotions = config.get("interjection_emotions", default_interjection_emotions())
        formal_markers = config.get("formal_markers", default_formal_markers())
        min_confidence = config.get("min_interjection_emotion_confidence", MIN_EMOTION_CONFIDENCE)
        max_suggestions = config.get("max_interjection_suggestions", MAX_SUGGESTIONS)

        style, _score, _reasons = detect_document_style(
            document.sentence_spans, formal_markers, interjection_emotions
        )
        document.document_style = style

        for span in document.sentence_spans:
            suggest_for_sentence(
                span,
                mode=mode,
                document_style=style,
                interjection_emotions=interjection_emotions,
                min_confidence=min_confidence,
                max_suggestions=max_suggestions,
            )
