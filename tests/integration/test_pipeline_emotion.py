from functools import lru_cache

import pytest

from expressive_tts.preprocess.emotion import ModelNotAvailableError as EmotionModelNotAvailableError
from expressive_tts.preprocess.emotion import _get_classifier
from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.schemas import Provenance

VALID_LABELS = {"happy", "angry", "sad", "fear", "surprise", "neutral", "unspecified"}
VALID_INTENSITIES = {"low", "medium", "high", "unspecified"}


@lru_cache(maxsize=None)
def _stack_available() -> bool:
    try:
        _get_pipeline()
        _get_classifier()
    except (ModelNotAvailableError, EmotionModelNotAvailableError):
        return False
    return True


requires_stack = pytest.mark.skipif(
    not _stack_available(),
    reason="requires the Trankit Romanian model and the transformer emotion model",
)


@requires_stack
def test_emotion_is_well_formed_end_to_end():
    pipeline = PreprocessPipeline.from_profile("expressive")
    result = pipeline.process("Sunt foarte fericit astăzi!")
    emotion = result.sentences[0].emotion
    assert emotion is not None
    assert emotion.label in VALID_LABELS
    assert emotion.intensity in VALID_INTENSITIES
    assert emotion.producer == "emotion_xlmr_v1"
    assert emotion.provenance == Provenance.PREDICTED
    # distribution is a normalized probability vector over the project labels
    assert set(emotion.distribution) <= (VALID_LABELS - {"unspecified"})
    assert abs(sum(emotion.distribution.values()) - 1.0) < 0.01


@requires_stack
def test_default_profile_includes_emotion():
    pipeline = PreprocessPipeline.from_profile("default")
    result = pipeline.process("Nu pot să cred! Am reușit.")
    for sentence in result.sentences:
        assert sentence.emotion is not None
        assert sentence.emotion.producer == "emotion_xlmr_v1"
        assert sentence.emotion.label in VALID_LABELS
