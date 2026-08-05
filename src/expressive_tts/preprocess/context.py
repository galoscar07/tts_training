"""Context-aware sentence smoothing: avoid implausible emotion swings
between adjacent sentences. See preprocess/objectives.md Phase 11.

The raw local prediction (`Sentence.emotion`, Phase 6) is never modified —
this processor only ever writes a separate `Sentence.context_emotion`
field ("Store both local and contextual results", "Local predictions are
never discarded"). Operates over `document.sentence_spans` in order,
within a single `process()` call — a real multi-paragraph *document* (as
opposed to a multi-sentence one) isn't a concept the current text-only
reader constructs, so the practical paragraph-boundary signal used here is
a blank line inside the source text itself, which does reset context
propagation for real when present.

Baseline formula (objectives.md): `context_score = α·current + β·previous`,
α > β. Implemented over `EmotionAnnotation.distribution` (Phase 6's
per-class score dict, already computed) rather than a single scalar, so
the blend can still produce a full label decision via argmax.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import ContextAdjustment, EmotionAnnotation

PRODUCER = "context_rules_v1"

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"

# Not gold-tuned — selected by inspection against dev.jsonl and the 3
# context.jsonl paragraphs available at the time this was written (a
# small sample; see scripts/evaluate_context.py's honest framing of what
# that can and can't establish).
ALPHA = 0.7
BETA = 0.3

# "Prevent a neutral sentence from erasing a strong explicit emotion":
# skip context blending entirely once the local prediction is already
# this confident, regardless of what a neighboring sentence says.
HIGH_CONFIDENCE_SKIP = 0.75

# "Smooth intensity only when confidence is low."
INTENSITY_SMOOTH_CONFIDENCE_THRESHOLD = 0.5


@lru_cache(maxsize=None)
def default_discourse_markers() -> dict[str, tuple[str, ...]]:
    path = _CONFIG_DIR / "discourse_markers.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {category: tuple(w.lower() for w in words) for category, words in data.items()}


_SENTENCE_END_RE = re.compile(r"[.!?…]+")
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def _paragraph_boundaries(document_text: str, spans: list[SentenceSpan]) -> set[int]:
    """Indices of spans that start a new paragraph — index 0 always does.

    `SentenceSpan.start`/`.end` are offsets into `document.clean_text`,
    which has already collapsed all whitespace — including blank lines —
    to single spaces by the time this runs (`cleaner.CleanerProcessor`),
    so a blank line can't be located by offset anymore. Instead: split the
    *original* source text on blank lines, count each chunk's approximate
    sentence count from terminal punctuation, and walk the cumulative
    counts to find which span index starts each chunk. This is a
    heuristic, not an exact offset mapping — unusual punctuation (nested
    quotes, abbreviations inside a chunk) can shift a boundary by a
    sentence — but no cumulative offset-mapping layer exists through the
    cleaning stages today, and building one is out of scope for this
    phase.
    """
    if not spans:
        return set()
    paragraphs = _BLANK_LINE_RE.split(document_text)
    boundaries = {0}
    cumulative = 0
    for paragraph in paragraphs[:-1]:
        cumulative += len(_SENTENCE_END_RE.findall(paragraph)) or 1
        if cumulative < len(spans):
            boundaries.add(cumulative)
    return boundaries


def _leading_discourse_category(span: SentenceSpan, markers: dict[str, tuple[str, ...]]) -> str | None:
    tokens = [t for t in (span.tokens or []) if t.upos != "PUNCT"]
    if not tokens:
        return None
    first = (tokens[0].lemma or tokens[0].text).lower()
    for category, words in markers.items():
        if first in words:
            return category
    return None


def _blend_distribution(current: dict, previous: dict, alpha: float, beta: float) -> dict:
    labels = set(current) | set(previous)
    blended = {label: alpha * current.get(label, 0.0) + beta * previous.get(label, 0.0) for label in labels}
    total = sum(blended.values())
    return {label: value / total for label, value in blended.items()} if total > 0 else current


def compute_context_adjustment(
    span: SentenceSpan,
    previous_emotion: EmotionAnnotation | None,
    *,
    alpha: float = ALPHA,
    beta_default: float = BETA,
    high_confidence_skip: float = HIGH_CONFIDENCE_SKIP,
    intensity_smooth_threshold: float = INTENSITY_SMOOTH_CONFIDENCE_THRESHOLD,
    discourse_category: str | None = None,
    producer: str = PRODUCER,
) -> ContextAdjustment | None:
    """Pure function: no local mutation. Returns `None` if `span.emotion`
    hasn't been computed (this layer requires `emotion`)."""
    emotion = span.emotion
    if emotion is None:
        return None

    if emotion.confidence >= high_confidence_skip or previous_emotion is None:
        return ContextAdjustment(
            label=emotion.label,
            confidence=emotion.confidence,
            valence=emotion.valence,
            arousal=emotion.arousal,
            intensity=emotion.intensity,
            changed=False,
            reason=None,
            producer=producer,
        )

    beta = beta_default
    marker_reason = None
    if discourse_category == "contrast":
        beta = 0.0  # a deliberate tonal break — don't pull toward the previous sentence at all
        marker_reason = "discourse_marker:contrast"
    elif discourse_category == "consequence":
        beta = min(1.0 - alpha, beta_default * 1.5)  # continuation of tone — slightly more context weight
        marker_reason = "discourse_marker:consequence"

    if beta <= 0.0:
        blended_label, blended_confidence = emotion.label, emotion.confidence
    else:
        local_dist = emotion.distribution or {emotion.label: 1.0}
        previous_dist = previous_emotion.distribution or {previous_emotion.label: 1.0}
        blended = _blend_distribution(local_dist, previous_dist, alpha, beta)
        blended_label = max(blended, key=blended.get)
        blended_confidence = round(blended[blended_label], 3)

    label_changed = blended_label != emotion.label

    intensity = emotion.intensity
    intensity_changed = False
    if emotion.confidence < intensity_smooth_threshold and previous_emotion.intensity != emotion.intensity:
        intensity = previous_emotion.intensity
        intensity_changed = True

    changed = label_changed or intensity_changed
    reasons = []
    if changed and marker_reason:
        reasons.append(marker_reason)
    if label_changed:
        reasons.append(f"context_blend:{emotion.label}->{blended_label}")
    if intensity_changed:
        reasons.append("intensity_smoothed")

    return ContextAdjustment(
        label=blended_label,
        confidence=blended_confidence,
        valence=emotion.valence,
        arousal=emotion.arousal,
        intensity=intensity,
        changed=changed,
        reason=";".join(reasons) if changed else None,
        producer=producer,
    )


class ContextProcessor:
    name = "context"
    version = PRODUCER
    provides = {"context"}
    requires = {"emotion"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        alpha = config.get("context_alpha", ALPHA)
        beta = config.get("context_beta", BETA)
        high_confidence_skip = config.get("context_high_confidence_skip", HIGH_CONFIDENCE_SKIP)
        intensity_threshold = config.get(
            "context_intensity_smooth_threshold", INTENSITY_SMOOTH_CONFIDENCE_THRESHOLD
        )
        markers = config.get("discourse_markers", default_discourse_markers())

        spans = document.sentence_spans
        boundaries = _paragraph_boundaries(document.original_text, spans)

        # Chained against each sentence's *local* prediction, not the
        # smoothed one — otherwise smoothing error would compound across a
        # long paragraph instead of each sentence being pulled toward what
        # was actually locally observed next to it.
        previous_emotion: EmotionAnnotation | None = None
        for index, span in enumerate(spans):
            if index in boundaries:
                previous_emotion = None
            category = _leading_discourse_category(span, markers)
            span.context_emotion = compute_context_adjustment(
                span,
                previous_emotion,
                alpha=alpha,
                beta_default=beta,
                high_confidence_skip=high_confidence_skip,
                intensity_smooth_threshold=intensity_threshold,
                discourse_category=category,
                producer=PRODUCER,
            )
            previous_emotion = span.emotion
