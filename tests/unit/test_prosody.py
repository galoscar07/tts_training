import pytest

from expressive_tts.preprocess.prosody import (
    PAUSE_RANGE_MS,
    RELATIVE_ENERGY_RANGE,
    RELATIVE_PITCH_RANGE,
    SPEAKING_RATE_RANGE,
    ProsodyProcessor,
    clamp,
    score_sentence,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import EmotionAnnotation, Provenance, Token


def tok(text, upos="NOUN", deprel=None, is_focus=False, focus_score=None):
    return Token(
        text=text,
        start=0,
        end=len(text),
        upos=upos,
        deprel=deprel,
        is_focus=is_focus,
        focus_score=focus_score,
    )


def emotion(arousal, label="happy"):
    return EmotionAnnotation(
        label=label, confidence=0.9, arousal=arousal, valence=0.7, intensity="medium", producer="x"
    )


def span(tokens, sentence_type="declarative", emotion_annotation=None):
    return SentenceSpan(
        text="x", start=0, end=1, sentence_type=sentence_type, tokens=tokens, emotion=emotion_annotation
    )


def test_clamp():
    assert clamp(0.5, 0.8, 1.2) == 0.8
    assert clamp(1.5, 0.8, 1.2) == 1.2
    assert clamp(1.0, 0.8, 1.2) == 1.0


@pytest.mark.parametrize(
    "text, expected_ms",
    [(",", 150), (";", 250), (":", 250), (".", 400), ("!", 400), ("?", 400), ("…", 600), ("...", 600)],
)
def test_punctuation_pause_values(text, expected_ms):
    s = span([tok(text, upos="PUNCT")])
    score_sentence(s)
    assert s.tokens[0].pause_after_ms == expected_ms


def test_non_punctuation_token_gets_no_pause():
    s = span([tok("masă")])
    score_sentence(s)
    assert s.tokens[0].pause_after_ms is None


def test_long_sentence_adds_extra_terminal_pause():
    short = span([tok(f"w{i}") for i in range(5)] + [tok(".", upos="PUNCT")])
    long = span([tok(f"w{i}") for i in range(20)] + [tok(".", upos="PUNCT")])
    score_sentence(short)
    score_sentence(long)
    assert long.tokens[-1].pause_after_ms > short.tokens[-1].pause_after_ms
    assert "long_sentence" in long.tokens[-1].prosody_rules


def test_clause_boundary_pause_from_deprel():
    s = span([tok("că", deprel="mark"), tok("plouă")])
    score_sentence(s)
    assert s.tokens[0].pause_before_ms is not None
    assert "clause_boundary" in s.tokens[0].prosody_rules
    assert s.tokens[1].pause_before_ms is None


@pytest.mark.parametrize(
    "sentence_type, expected_contour",
    [
        ("interrogative", "rising"),
        ("declarative", "falling"),
        ("exclamative", "falling"),
        ("imperative", "falling"),
        ("incomplete", "continuation"),
    ],
)
def test_terminal_contour_from_sentence_type(sentence_type, expected_contour):
    s = span([tok("x")], sentence_type=sentence_type)
    score_sentence(s)
    assert s.prosody.terminal_contour == expected_contour


def test_no_emotion_data_gives_baseline():
    s = span([tok("x")])
    score_sentence(s)
    assert s.prosody.speaking_rate == 1.0
    assert s.prosody.relative_pitch == 1.0
    assert s.prosody.relative_energy == 1.0
    assert "no_emotion_data" in s.prosody.rules


def test_high_arousal_increases_rate_pitch_energy():
    s = span([tok("x")], emotion_annotation=emotion(arousal=0.9))
    score_sentence(s)
    assert s.prosody.speaking_rate > 1.0
    assert s.prosody.relative_pitch > 1.0
    assert s.prosody.relative_energy > 1.0


def test_low_arousal_decreases_rate_pitch_energy():
    s = span([tok("x")], emotion_annotation=emotion(arousal=0.1))
    score_sentence(s)
    assert s.prosody.speaking_rate < 1.0
    assert s.prosody.relative_pitch < 1.0
    assert s.prosody.relative_energy < 1.0


def test_extreme_arousal_stays_within_safe_ranges():
    s = span([tok("x")], emotion_annotation=emotion(arousal=1.0))
    score_sentence(s)
    assert SPEAKING_RATE_RANGE[0] <= s.prosody.speaking_rate <= SPEAKING_RATE_RANGE[1]
    assert RELATIVE_PITCH_RANGE[0] <= s.prosody.relative_pitch <= RELATIVE_PITCH_RANGE[1]
    assert RELATIVE_ENERGY_RANGE[0] <= s.prosody.relative_energy <= RELATIVE_ENERGY_RANGE[1]

    s2 = span([tok("x")], emotion_annotation=emotion(arousal=0.0))
    score_sentence(s2)
    assert SPEAKING_RATE_RANGE[0] <= s2.prosody.speaking_rate <= SPEAKING_RATE_RANGE[1]


def test_focused_token_gets_local_boost_applied_to_sentence_baseline():
    s = span(
        [tok("fericit", is_focus=True, focus_score=1.0)],
        emotion_annotation=emotion(arousal=0.9),
    )
    score_sentence(s)
    token = s.tokens[0]
    assert token.local_relative_pitch is not None
    assert token.local_relative_pitch >= s.prosody.relative_pitch
    assert token.local_relative_pitch <= RELATIVE_PITCH_RANGE[1]
    assert "local_focus_boost" in token.prosody_rules


def test_non_focused_token_has_no_local_override():
    s = span([tok("masă", is_focus=False)], emotion_annotation=emotion(arousal=0.9))
    score_sentence(s)
    assert s.tokens[0].local_relative_pitch is None
    assert s.tokens[0].local_relative_energy is None


def test_pause_never_exceeds_safe_range():
    s = span([tok(".", upos="PUNCT")] * 3)
    score_sentence(s)
    for token in s.tokens:
        assert PAUSE_RANGE_MS[0] <= token.pause_after_ms <= PAUSE_RANGE_MS[1]


def test_user_overrides_preserved():
    s = span([tok(".", upos="PUNCT")], emotion_annotation=emotion(arousal=0.9))
    score_sentence(s, user_overrides={"terminal_contour": "continuation", "speaking_rate": 0.95})
    assert s.prosody.terminal_contour == "continuation"
    assert s.prosody.speaking_rate == 0.95
    assert "user_override_terminal_contour" in s.prosody.rules
    assert "user_override_speaking_rate" in s.prosody.rules
    # non-overridden fields still come from the rule engine
    assert s.prosody.relative_pitch != 1.0


def test_processor_end_to_end_with_indexed_overrides():
    span0 = SentenceSpan(text="a", start=0, end=1, sentence_type="declarative", tokens=[tok(".", upos="PUNCT")])
    span1 = SentenceSpan(text="b", start=0, end=1, sentence_type="declarative", tokens=[tok(".", upos="PUNCT")])
    document = PipelineDocument(original_text="a b", sentence_spans=[span0, span1])
    ProsodyProcessor().process(
        document, {"user_prosody_overrides": {1: {"terminal_contour": "rising"}}}
    )
    assert document.sentence_spans[0].prosody.terminal_contour == "falling"
    assert document.sentence_spans[1].prosody.terminal_contour == "rising"


def test_prosody_provenance_is_rule():
    s = span([tok(".", upos="PUNCT")])
    score_sentence(s)
    assert s.prosody.provenance == Provenance.RULE
    assert s.prosody.producer == "prosody_rules_v1"
