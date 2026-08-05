import pytest

from expressive_tts.preprocess.emotion import (
    PROJECT_LABELS,
    EmotionProcessor,
    _decide_intensity,
    _decide_label,
    _default_predict,
    _punctuation_and_caps,
    _weighted_vad,
    score_sentence,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import Provenance, Token


def word(text, upos="ADJ"):
    return Token(text=text, start=0, end=len(text), lemma=text.lower(), upos=upos)


def punct(text):
    return Token(text=text, start=0, end=len(text), lemma=text, upos="PUNCT")


def dist(**scores) -> dict[str, float]:
    """A full project-label distribution, zero-filled for unspecified keys."""
    full = {label: 0.0 for label in PROJECT_LABELS}
    full.update(scores)
    return full


def predictor_for(distribution):
    return lambda _text: distribution


# --- _decide_label ---------------------------------------------------------


def test_confident_top_label_wins():
    label, confidence, secondary = _decide_label(dist(happy=0.8, neutral=0.2))
    assert label == "happy"
    assert confidence == pytest.approx(0.8)
    assert secondary is None


def test_low_confidence_is_unspecified():
    # top prob below MIN_CONFIDENCE (0.40)
    label, confidence, _ = _decide_label(dist(happy=0.3, sad=0.25, fear=0.25, angry=0.2))
    assert label == "unspecified"
    assert confidence == pytest.approx(0.3)


def test_exact_tie_is_unspecified():
    label, _, _ = _decide_label(dist(happy=0.5, sad=0.5))
    assert label == "unspecified"


def test_substantial_runner_up_becomes_secondary():
    label, _, secondary = _decide_label(dist(happy=0.55, surprise=0.35, neutral=0.1))
    assert label == "happy"
    assert secondary == "surprise"  # 0.35 >= 0.5 * 0.55


def test_empty_distribution_is_unspecified():
    label, confidence, _ = _decide_label(dist())
    assert label == "unspecified"
    assert confidence == 0.0


# --- _weighted_vad ---------------------------------------------------------


def test_weighted_vad_pure_label_matches_prototype():
    valence, arousal = _weighted_vad(dist(happy=1.0))
    assert valence == pytest.approx(0.85)
    assert arousal == pytest.approx(0.65)


def test_weighted_vad_blends_labels():
    valence, arousal = _weighted_vad(dist(happy=0.5, sad=0.5))
    assert valence == pytest.approx((0.85 + 0.15) / 2)
    assert arousal == pytest.approx((0.65 + 0.30) / 2)


def test_weighted_vad_empty_is_none():
    assert _weighted_vad(dist()) == (None, None)


# --- intensity & punctuation ----------------------------------------------


def test_unspecified_label_has_unspecified_intensity():
    assert _decide_intensity("unspecified", 0.9, 0, 0, 0) == "unspecified"


def test_intensity_scales_with_arousal():
    assert _decide_intensity("angry", 0.8, 0, 0, 0) == "high"
    assert _decide_intensity("neutral", 0.5, 0, 0, 0) == "medium"
    assert _decide_intensity("sad", 0.3, 0, 0, 0) == "low"


def test_punctuation_only_nudges_intensity_upward():
    plain = _decide_intensity("fear", 0.6, 0, 0, 0)
    excited = _decide_intensity("fear", 0.6, 3, 0, 0)
    order = {"low": 0, "medium": 1, "high": 2, "unspecified": -1}
    assert order[excited] >= order[plain]


def test_punctuation_and_caps_counts():
    tokens = [word("FRICĂ", upos="NOUN"), punct("!"), punct("!"), punct("?")]
    bangs, questions, caps = _punctuation_and_caps(tokens)
    assert (bangs, questions, caps) == (2, 1, 1)


# --- score_sentence --------------------------------------------------------


def test_score_sentence_confident():
    result = score_sentence("Sunt fericit", [word("fericit")], predictor_for(dist(happy=0.9, neutral=0.1)))
    assert result.label == "happy"
    assert result.confidence == 0.9
    assert result.valence is not None and result.valence > 0.5
    assert result.provenance == Provenance.PREDICTED
    assert result.producer == "emotion_xlmr_v1"
    assert result.evidence and result.evidence[0].rule.startswith("transformer:")


def test_score_sentence_unspecified_has_no_evidence():
    result = score_sentence("ceva", [word("ceva")], predictor_for(dist(happy=0.3, sad=0.3, fear=0.2, angry=0.2)))
    assert result.label == "unspecified"
    assert result.intensity == "unspecified"
    assert result.evidence == []


def test_score_sentence_punctuation_raises_intensity_not_label():
    base = score_sentence("frică", [word("frică", upos="NOUN")], predictor_for(dist(fear=0.9, neutral=0.1)))
    loud = score_sentence(
        "frică!!!",
        [word("frică", upos="NOUN"), punct("!"), punct("!"), punct("!")],
        predictor_for(dist(fear=0.9, neutral=0.1)),
    )
    order = {"low": 0, "medium": 1, "high": 2, "unspecified": -1}
    assert loud.label == base.label == "fear"
    assert order[loud.intensity] >= order[base.intensity]


# --- EmotionProcessor with injected predictor ------------------------------


def test_processor_uses_injected_predictor():
    span = SentenceSpan(text="Sunt fericit", start=0, end=12, tokens=[word("fericit")])
    document = PipelineDocument(original_text="Sunt fericit", sentence_spans=[span])
    EmotionProcessor().process(document, {"emotion_predictor": predictor_for(dist(happy=1.0))})
    emotion = document.sentence_spans[0].emotion
    assert emotion is not None
    assert emotion.label == "happy"
    assert emotion.provenance == Provenance.PREDICTED
    assert emotion.producer == "emotion_xlmr_v1"


def test_processor_prefers_normalized_text():
    captured = {}

    def spy(text):
        captured["text"] = text
        return dist(neutral=1.0)

    span = SentenceSpan(text="orig", start=0, end=4, normalized_text="normalized form", tokens=[])
    document = PipelineDocument(original_text="orig", sentence_spans=[span])
    EmotionProcessor().process(document, {"emotion_predictor": spy})
    assert captured["text"] == "normalized form"


# --- _default_predict aggregation (fake classifier, no download) -----------


def test_default_predict_aggregates_raw_labels(monkeypatch):
    import expressive_tts.preprocess.emotion as emotion_module

    def fake_classifier(text, truncation=True):
        # joy + gratitude + love all map to "happy"; anger -> "angry"
        return [
            {"label": "joy", "score": 0.4},
            {"label": "gratitude", "score": 0.1},
            {"label": "love", "score": 0.1},
            {"label": "anger", "score": 0.2},
            {"label": "neutral", "score": 0.2},
        ]

    monkeypatch.setattr(emotion_module, "_get_classifier", lambda: fake_classifier)
    distribution = _default_predict("orice text")
    assert distribution["happy"] == pytest.approx(0.6)
    assert distribution["angry"] == pytest.approx(0.2)
    assert distribution["neutral"] == pytest.approx(0.2)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_default_predict_handles_nested_list(monkeypatch):
    import expressive_tts.preprocess.emotion as emotion_module

    def fake_classifier(text, truncation=True):
        return [[{"label": "sadness", "score": 1.0}]]  # some pipelines nest the result

    monkeypatch.setattr(emotion_module, "_get_classifier", lambda: fake_classifier)
    distribution = _default_predict("trist")
    assert distribution["sad"] == pytest.approx(1.0)
