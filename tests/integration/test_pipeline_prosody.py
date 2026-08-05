from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.prosody import (
    PAUSE_RANGE_MS,
    RELATIVE_ENERGY_RANGE,
    RELATIVE_PITCH_RANGE,
    SPEAKING_RATE_RANGE,
)


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(
    not _model_available(), reason="Stanza Romanian model not downloaded"
)


@requires_model
def test_full_chain_emotion_focus_prosody():
    # prosody.ProsodyProcessor.requires == {"linguistic"} only — it degrades
    # gracefully without emotion/focus, so both must be requested explicitly
    # to exercise the full chain here.
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit!", include={"prosody", "emotion", "focus"}
    )
    sentence = result.sentences[0]

    assert sentence.prosody is not None
    assert sentence.prosody.terminal_contour == "falling"
    assert sentence.prosody.speaking_rate > 1.0  # happy -> higher arousal -> faster

    focused = [t for t in sentence.tokens if t.is_focus]
    assert focused
    assert any(t.local_relative_pitch is not None for t in focused)


@requires_model
def test_interrogative_gets_rising_contour():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Ce faci?", include={"prosody"})
    assert result.sentences[0].prosody.terminal_contour == "rising"


@requires_model
def test_prosody_alone_does_not_require_phonemes_or_espeak():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Nu pot să cred! Am reușit.", include={"prosody"})
    assert result.phoneme_text is None
    for sentence in result.sentences:
        assert sentence.prosody is not None


@requires_model
def test_range_safety_over_real_sentences():
    pipeline = PreprocessPipeline()
    sentences = [
        "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.",
        "Uau! Chiar am reușit?",
        "Nu pot să cred! Am reușit.",
        "Sunt extrem de fericit azi, dar mâine trebuie să plec devreme.",
        "Ce panică, nu mai suport deloc situația asta îngrozitoare!",
    ]
    for text in sentences:
        result = pipeline.process(text, include={"prosody"})
        for sentence in result.sentences:
            p = sentence.prosody
            assert SPEAKING_RATE_RANGE[0] <= p.speaking_rate <= SPEAKING_RATE_RANGE[1]
            assert RELATIVE_PITCH_RANGE[0] <= p.relative_pitch <= RELATIVE_PITCH_RANGE[1]
            assert RELATIVE_ENERGY_RANGE[0] <= p.relative_energy <= RELATIVE_ENERGY_RANGE[1]
            assert PAUSE_RANGE_MS[0] <= p.pause_after_ms <= PAUSE_RANGE_MS[1]
            for token in sentence.tokens:
                if token.pause_after_ms is not None:
                    assert PAUSE_RANGE_MS[0] <= token.pause_after_ms <= PAUSE_RANGE_MS[1]
                if token.pause_before_ms is not None:
                    assert PAUSE_RANGE_MS[0] <= token.pause_before_ms <= PAUSE_RANGE_MS[1]
                if token.local_relative_pitch is not None:
                    assert RELATIVE_PITCH_RANGE[0] <= token.local_relative_pitch <= RELATIVE_PITCH_RANGE[1]
                if token.local_relative_energy is not None:
                    assert RELATIVE_ENERGY_RANGE[0] <= token.local_relative_energy <= RELATIVE_ENERGY_RANGE[1]


@requires_model
def test_user_prosody_overrides_end_to_end():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Plouă.", include={"prosody"}, user_prosody_overrides={0: {"terminal_contour": "rising"}}
    )
    assert result.sentences[0].prosody.terminal_contour == "rising"
    assert "user_override_terminal_contour" in result.sentences[0].prosody.rules
