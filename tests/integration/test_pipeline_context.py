from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(not _model_available(), reason="Stanza Romanian model not downloaded")


@requires_model
def test_context_layer_preserves_local_emotion():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit azi! E o zi obișnuită, nimic special.", include={"emotion", "context"}
    )
    for sentence in result.sentences:
        assert sentence.emotion is not None
        assert sentence.context_emotion is not None
        # local prediction is a real value regardless of what context did
        assert sentence.emotion.label


@requires_model
def test_context_never_discards_local_even_when_it_changes():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Am câștigat marele premiu la concurs! E ceva obișnuit, se întâmplă des.",
        include={"emotion", "context"},
    )
    for sentence in result.sentences:
        assert sentence.emotion is not None  # never None/overwritten just because context ran


@requires_model
def test_context_traceable_when_changed():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit azi! E o zi obișnuită, nimic special.", include={"emotion", "context"}
    )
    for sentence in result.sentences:
        adjustment = sentence.context_emotion
        if adjustment.changed:
            assert adjustment.reason is not None
        else:
            assert adjustment.reason is None


@requires_model
def test_context_requires_only_emotion_not_prosody_or_focus():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Plouă azi.", include={"context"})
    assert result.sentences[0].context_emotion is not None
    assert result.phoneme_text is None


@requires_model
def test_paragraph_boundary_resets_context_in_real_pipeline():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit azi!\n\nE o zi obișnuită, nimic special.",
        include={"emotion", "context"},
    )
    assert len(result.sentences) == 2
    second = result.sentences[1]
    # with the paragraph boundary respected, the low-confidence second
    # sentence should NOT be pulled toward the first sentence's strong
    # "happy" — it should keep (or stay close to) its own local prediction.
    assert second.context_emotion.label == second.emotion.label
