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


requires_model = pytest.mark.skipif(
    not _model_available(), reason="Stanza Romanian model not downloaded"
)


@requires_model
def test_disabled_by_default_even_with_strong_emotion():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Sunt extrem de fericit!", include={"interjections"})
    assert result.sentences[0].interjection_suggestions == []
    assert result.sentences[0].text_with_interjections is None


@requires_model
def test_suggest_mode_on_strong_emotion_conversational_sentence():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit!", include={"interjections"}, interjection_mode="suggest"
    )
    assert result.document_style == "conversational"
    assert result.sentences[0].interjection_suggestions
    assert result.sentences[0].text == "Sunt extrem de fericit!"  # unmodified


@requires_model
def test_insert_mode_preserves_original_text():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Sunt extrem de fericit!", include={"interjections"}, interjection_mode="insert"
    )
    sentence = result.sentences[0]
    assert sentence.text == "Sunt extrem de fericit!"
    assert sentence.text_with_interjections is not None
    assert sentence.text_with_interjections != sentence.text


@requires_model
def test_formal_legal_sentence_gets_zero_suggestions():
    pipeline = PreprocessPipeline()
    text = (
        "Consiliul Superior al Magistraturii propune Președintelui României "
        "numirea în funcție a judecătorilor și a procurorilor, cu excepția "
        "celor stagiari, în condițiile legii."
    )
    result = pipeline.process(text, include={"interjections"}, interjection_mode="insert")
    assert result.document_style == "formal"
    assert result.sentences[0].interjection_suggestions == []
    assert result.sentences[0].text_with_interjections is None


@requires_model
def test_existing_interjection_prevents_repeated_suggestion():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Uau, ce bine îmi pare!", include={"interjections"}, interjection_mode="suggest"
    )
    assert result.sentences[0].interjection_suggestions == []
