from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.phonemizer import espeak_available, unknown_phonemes
from expressive_tts.preprocess.pipeline import PreprocessPipeline


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_stack = pytest.mark.skipif(
    not (_model_available() and espeak_available()),
    reason="requires the Trankit Romanian model and the espeak binary",
)


@requires_stack
def test_full_pronunciation_profile_on_negated_exclamation():
    pipeline = PreprocessPipeline.from_profile("pronunciation")
    result = pipeline.process("Nu pot să cred! Am reușit.")

    first, second = result.sentences
    assert first.sentence_type == "exclamative"
    assert first.is_negated is True
    assert second.sentence_type == "declarative"
    assert second.is_negated is False

    reusit = next(t for t in second.tokens if t.text == "reușit")
    assert reusit.lemma == "reuși"
    assert reusit.upos == "VERB"
    assert reusit.syllables == ["reu", "șit"]
    assert reusit.stressed_syllable_index == 1
    assert reusit.phonemes

    assert result.phoneme_text


@requires_stack
def test_full_pronunciation_profile_on_objectives_worked_example():
    pipeline = PreprocessPipeline.from_profile("pronunciation")
    result = pipeline.process(
        "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30."
    )

    assert result.normalized_text == (
        "Doctor Popescu a trimis douăzeci și cinci de kilograme pe doisprezece "
        "iulie două mii douăzeci și șase, la ora paisprezece și treizeci de minute."
    )

    all_tokens = [token for sentence in result.sentences for token in sentence.tokens]
    word_tokens = [t for t in all_tokens if t.upos != "PUNCT"]
    assert word_tokens  # linguistic analysis ran
    assert all(t.phonemes for t in word_tokens)  # every word got a pronunciation

    unknown = {ch for t in word_tokens for ch in unknown_phonemes(t.phonemes)}
    assert unknown == set(), f"unexpected unknown phonemes: {unknown}"


@requires_stack
def test_imperative_sentence_type_end_to_end():
    pipeline = PreprocessPipeline.from_profile("pronunciation")
    result = pipeline.process("Ascultă-mă!")
    assert result.sentences[0].sentence_type == "imperative"


@requires_stack
def test_interjection_detected_end_to_end():
    pipeline = PreprocessPipeline.from_profile("pronunciation")
    result = pipeline.process("Uau, ce frumos!")
    interjections = [t for t in result.sentences[0].tokens if t.is_interjection]
    assert any(t.text.lower() == "uau" for t in interjections)
