"""Transformer-based Romanian emotion classifier. See readme.md section 2.4
and preprocess/objectives.md Phase 6/7.

Replaces the earlier rule-based NRC-lexicon baseline as the pipeline's
emotion source. It runs a multilingual XLM-RoBERTa emotion model
(`tabularisai/multilingual-emotion-classification`) locally through
HuggingFace `transformers` — free, offline after a one-time download. The
model's label set (anger, contempt, disgust, fear, frustration, gratitude,
joy, love, neutral, sadness, surprise) is aggregated down to this project's
label set (`objectives.md` section "Initial label set"):

    happy   ← joy, gratitude, love
    angry   ← anger, frustration, contempt, disgust
    sad     ← sadness
    fear    ← fear
    surprise← surprise
    neutral ← neutral

Per sentence: softmax over the model's classes → sum probabilities into the
project labels → argmax. `neutral` and `unspecified` stay distinct
(objectives.md section 2.4): the model can predict `neutral` as a positive
reading, whereas `unspecified` is *our* abstention when the top label's
aggregated probability is below `MIN_CONFIDENCE` or ties another label —
low-confidence predictions are marked rather than silently accepted.

Valence and arousal come from a per-emotion VAD prototype table (there is no
lexical VAD lookup any more), taken as the distribution-weighted average so
they vary smoothly with the model's uncertainty. Intensity is derived from
arousal, nudged — never decided — by repeated punctuation and capitalization,
per Phase 6's "punctuation as supporting, not decisive, evidence".

The model runs behind an injectable `predictor` (config key
`emotion_predictor`): a callable `str -> dict[label, probability]` over the
project labels. Tests inject a deterministic fake so they neither download
nor run the ~1.1GB transformer.
"""

from __future__ import annotations

from pathlib import Path

from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.schemas import EmotionAnnotation, EmotionEvidence, Provenance, Token

MODEL_ID = "tabularisai/multilingual-emotion-classification"
PRODUCER = "emotion_xlmr_v1"
MODEL_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "models" / "emotion_transformer"

# The transformer's raw class labels -> this project's label set. Several
# fine-grained negative categories fold into `angry` (the only high-arousal
# negative label we model); `disgust`/`contempt` have no closer target.
RAW_LABEL_MAP = {
    "joy": "happy",
    "gratitude": "happy",
    "love": "happy",
    "anger": "angry",
    "frustration": "angry",
    "contempt": "angry",
    "disgust": "angry",
    "sadness": "sad",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
}

PROJECT_LABELS = ("happy", "angry", "sad", "fear", "surprise", "neutral")

# (valence, arousal) prototypes on the NRC 0-1 scale (0.5 = neutral), used to
# turn a categorical distribution into continuous valence/arousal. Kept
# deliberately coarse — these are category anchors, not per-word lexicon
# values.
VAD_PROTOTYPES = {
    "happy": (0.85, 0.65),
    "angry": (0.15, 0.80),
    "sad": (0.15, 0.30),
    "fear": (0.20, 0.75),
    "surprise": (0.60, 0.75),
    "neutral": (0.50, 0.40),
}

# Below this aggregated top-label probability we abstain (`unspecified`),
# matching Phase 6's requirement to return `unspecified` when confidence is
# low rather than committing to a weak label.
MIN_CONFIDENCE = 0.40
SECONDARY_SHARE = 0.5  # a runner-up this close to the top is reported as secondary

_classifier_instance = None  # lazy singleton — loading the transformer is expensive


class ModelNotAvailableError(RuntimeError):
    """The transformer emotion model isn't installed/downloaded yet."""


def _get_classifier():
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise ModelNotAvailableError(
            "transformers is not installed; run `pip install -e '.[ai]'`"
        ) from exc

    try:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(MODEL_CACHE_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, cache_dir=str(MODEL_CACHE_DIR)
        )
        _classifier_instance = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, top_k=None
        )
    except Exception as exc:
        raise ModelNotAvailableError(
            "transformer emotion model not available; run "
            "`./.venv/bin/python scripts/download_emotion_model.py` first "
            "(needs network on the first run)"
        ) from exc

    return _classifier_instance


def _default_predict(text: str) -> dict[str, float]:
    """Run the transformer and aggregate its class probabilities into a
    project-label distribution (normalized to sum to 1 over mapped mass)."""
    classifier = _get_classifier()
    raw = classifier(text, truncation=True)
    # `top_k=None` yields a list of {label, score}; some transformers wrap it
    # in an extra list when given a single string.
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    distribution = {label: 0.0 for label in PROJECT_LABELS}
    for entry in raw:
        project_label = RAW_LABEL_MAP.get(str(entry["label"]).lower())
        if project_label is not None:
            distribution[project_label] += float(entry["score"])

    total = sum(distribution.values())
    if total > 0:
        distribution = {label: score / total for label, score in distribution.items()}
    return distribution


def _punctuation_and_caps(tokens: list[Token]) -> tuple[int, int, int]:
    repeated_bangs = 0
    repeated_questions = 0
    caps_count = 0
    for token in tokens:
        if token.upos == "PUNCT":
            if token.text == "!":
                repeated_bangs += 1
            elif token.text == "?":
                repeated_questions += 1
        elif len(token.text) > 1 and token.text.isalpha() and token.text.isupper():
            caps_count += 1
    return repeated_bangs, repeated_questions, caps_count


def _weighted_vad(distribution: dict[str, float]) -> tuple[float | None, float | None]:
    total = sum(distribution.values())
    if total <= 0:
        return None, None
    valence = sum(distribution[label] * VAD_PROTOTYPES[label][0] for label in distribution) / total
    arousal = sum(distribution[label] * VAD_PROTOTYPES[label][1] for label in distribution) / total
    return valence, arousal


def _decide_label(distribution: dict[str, float]) -> tuple[str, float, str | None]:
    if not any(distribution.values()):
        return "unspecified", 0.0, None

    ranked = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ranked[0]
    runner_up_label, runner_up_prob = ranked[1] if len(ranked) > 1 else (None, 0.0)

    secondary = runner_up_label if runner_up_prob >= SECONDARY_SHARE * top_prob else None

    # An exact tie has no principled winner — abstain rather than let sort
    # order pick one arbitrarily.
    if runner_up_prob == top_prob:
        return "unspecified", top_prob, secondary
    if top_prob < MIN_CONFIDENCE:
        return "unspecified", top_prob, secondary
    return top_label, top_prob, secondary


def _decide_intensity(
    label: str, arousal: float | None, bangs: int, questions: int, caps_count: int
) -> str:
    if label == "unspecified":
        return "unspecified"
    base = arousal if arousal is not None else 0.5
    # Punctuation/capitalization are *supporting* evidence only (objectives.md
    # Phase 6): they nudge intensity, never which category wins.
    boost = min(0.15, 0.05 * bangs + 0.03 * questions + 0.05 * caps_count)
    score = base + boost
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def score_sentence(text: str, tokens: list[Token], predictor) -> EmotionAnnotation:
    distribution = predictor(text)
    label, confidence, secondary_label = _decide_label(distribution)
    valence, arousal = _weighted_vad(distribution)

    bangs, questions, caps_count = _punctuation_and_caps(tokens)
    intensity = _decide_intensity(label, arousal, bangs, questions, caps_count)

    evidence: list[EmotionEvidence] = []
    if label != "unspecified":
        evidence.append(EmotionEvidence(span=text.strip()[:80], rule=f"transformer:{MODEL_ID}"))

    return EmotionAnnotation(
        label=label,
        confidence=round(confidence, 3),
        valence=round(valence, 3) if valence is not None else None,
        arousal=round(arousal, 3) if arousal is not None else None,
        intensity=intensity,
        secondary_label=secondary_label,
        distribution={k: round(v, 3) for k, v in distribution.items()},
        evidence=evidence,
        provenance=Provenance.PREDICTED,
        producer=PRODUCER,
    )


class EmotionProcessor:
    name = "emotion"
    version = PRODUCER
    provides = {"emotion", "intensity"}
    requires = {"linguistic"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        predictor = config.get("emotion_predictor") or _default_predict

        for sentence in document.sentence_spans:
            tokens = sentence.tokens or []
            text = sentence.normalized_text or sentence.text
            sentence.emotion = score_sentence(text, tokens, predictor)
