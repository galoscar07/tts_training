"""Prosody and pause prediction: converts punctuation, syntax, emotion, and
focus into model-independent prosodic control values. See readme.md
section 1.5/2.4 and preprocess/objectives.md Phase 9.

Deliberately model-independent: these are intermediate control values (a
speaking-rate multiplier, a relative-pitch multiplier, a pause duration in
ms, ...), not guaranteed acoustic outputs (objectives.md: "these are
intermediate control values, not guaranteed acoustic outputs").
"""

from __future__ import annotations

from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import ProsodyAnnotation, Provenance

PRODUCER = "prosody_rules_v1"

SPEAKING_RATE_RANGE = (0.80, 1.20)
RELATIVE_PITCH_RANGE = (0.85, 1.20)
RELATIVE_ENERGY_RANGE = (0.80, 1.20)
PAUSE_RANGE_MS = (0, 1000)

# Base pause duration, in ms, by punctuation mark (objectives.md Phase 9:
# "Define pause values for commas, semicolons, colons, periods, question
# marks, exclamation marks, and ellipses").
_PUNCTUATION_PAUSE_MS = {
    ",": 150,
    ";": 250,
    ":": 250,
    ".": 400,
    "!": 400,
    "?": 400,
    "…": 600,  # hesitation
}
_ELLIPSIS_TEXTS = {"...", "…"}
_TERMINAL_MARKS = {".", "!", "?", "…"}

# "Modify pause values using sentence length": longer sentences get a
# small extra terminal pause (more recovery time needed).
_LONG_SENTENCE_TOKEN_COUNT = 15
_LONG_SENTENCE_EXTRA_MS = 100

# Dependency relations marking a subordinate/adverbial/complement clause
# boundary — "detect clause boundaries from dependency parsing"
# (objectives.md), using real parse structure, not just punctuation.
_CLAUSE_BOUNDARY_DEPRELS = {"mark", "advcl", "ccomp"}
_CLAUSE_BOUNDARY_PAUSE_MS = 100

_CONTOUR_BY_SENTENCE_TYPE = {
    "interrogative": "rising",
    "declarative": "falling",
    "exclamative": "falling",
    "imperative": "falling",
    "incomplete": "continuation",
}
_DEFAULT_CONTOUR = "falling"

# value = 1.0 + (arousal - 0.5) * _AROUSAL_SENSITIVITY, then clamped.
# "Increase energy and pitch range for high-arousal emotions. Reduce rate
# and energy for low-arousal emotions." — one symmetric formula does both.
_AROUSAL_SENSITIVITY = 0.4
_FOCUS_BOOST = 0.15  # additional local pitch/energy boost, scaled by focus_score


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pause_key(text: str) -> str | None:
    if text in _ELLIPSIS_TEXTS:
        return "…"
    if text in _PUNCTUATION_PAUSE_MS:
        return text
    return None


def _rate_pitch_energy(arousal: float | None) -> tuple[float, float, float, str]:
    if arousal is None:
        return 1.0, 1.0, 1.0, "no_emotion_data"
    delta = (arousal - 0.5) * _AROUSAL_SENSITIVITY
    rate = clamp(1.0 + delta, *SPEAKING_RATE_RANGE)
    pitch = clamp(1.0 + delta, *RELATIVE_PITCH_RANGE)
    energy = clamp(1.0 + delta, *RELATIVE_ENERGY_RANGE)
    return rate, pitch, energy, "arousal_driven"


def score_sentence(span: SentenceSpan, *, user_overrides: dict | None = None) -> None:
    """Compute `ProsodyAnnotation` for `span` and set `pause_*`/
    `local_relative_*` on its tokens, in place."""
    tokens = span.tokens or []
    content_token_count = sum(1 for t in tokens if t.upos != "PUNCT")
    long_sentence = content_token_count > _LONG_SENTENCE_TOKEN_COUNT

    arousal = span.emotion.arousal if span.emotion else None
    rate, pitch, energy, rate_rule = _rate_pitch_energy(arousal)
    contour = _CONTOUR_BY_SENTENCE_TYPE.get(span.sentence_type, _DEFAULT_CONTOUR)

    sentence_pause_after_ms = 0

    for token in tokens:
        key = _pause_key(token.text)
        if key is not None:
            ms = _PUNCTUATION_PAUSE_MS[key]
            rules = [f"punctuation_{key}"]
            if long_sentence and key in _TERMINAL_MARKS:
                ms += _LONG_SENTENCE_EXTRA_MS
                rules.append("long_sentence")
            ms = int(clamp(ms, *PAUSE_RANGE_MS))
            token.pause_after_ms = ms
            token.prosody_rules = list(token.prosody_rules) + rules
            token.prosody_provenance = Provenance.RULE
            token.prosody_producer = PRODUCER
            sentence_pause_after_ms = ms

        if token.deprel in _CLAUSE_BOUNDARY_DEPRELS:
            token.pause_before_ms = int(clamp(_CLAUSE_BOUNDARY_PAUSE_MS, *PAUSE_RANGE_MS))
            token.prosody_rules = list(token.prosody_rules) + ["clause_boundary"]
            token.prosody_provenance = Provenance.RULE
            token.prosody_producer = PRODUCER

        if token.is_focus:
            boost = _FOCUS_BOOST * (token.focus_score or 0.0)
            token.local_relative_pitch = round(clamp(pitch + boost, *RELATIVE_PITCH_RANGE), 3)
            token.local_relative_energy = round(clamp(energy + boost, *RELATIVE_ENERGY_RANGE), 3)
            token.prosody_rules = list(token.prosody_rules) + ["local_focus_boost"]
            token.prosody_provenance = Provenance.RULE
            token.prosody_producer = PRODUCER

    overrides = user_overrides or {}
    rules = [rate_rule, f"terminal_contour_from_{span.sentence_type or 'unknown'}"]
    rules += [f"user_override_{field}" for field in overrides]

    span.prosody = ProsodyAnnotation(
        speaking_rate=overrides.get("speaking_rate", round(rate, 3)),
        relative_pitch=overrides.get("relative_pitch", round(pitch, 3)),
        relative_energy=overrides.get("relative_energy", round(energy, 3)),
        terminal_contour=overrides.get("terminal_contour", contour),
        pause_after_ms=overrides.get("pause_after_ms", sentence_pause_after_ms),
        rules=rules,
        provenance=Provenance.RULE,
        producer=PRODUCER,
    )


class ProsodyProcessor:
    name = "prosody"
    version = PRODUCER
    provides = {"prosody"}
    requires = {"linguistic"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        user_prosody_overrides = config.get("user_prosody_overrides") or {}
        for index, sentence in enumerate(document.sentence_spans):
            score_sentence(sentence, user_overrides=user_prosody_overrides.get(index))
