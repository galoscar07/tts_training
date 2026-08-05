from functools import lru_cache

import pytest

from expressive_tts.preprocess.emotion import ModelNotAvailableError as EmotionModelNotAvailableError
from expressive_tts.preprocess.emotion import _get_classifier
from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.phonemizer import espeak_available
from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.schemas import PreprocessResult


@lru_cache(maxsize=None)
def _default_profile_stack_available() -> bool:
    try:
        _get_pipeline()
        _get_classifier()
    except (ModelNotAvailableError, EmotionModelNotAvailableError):
        return False
    return espeak_available()


def test_process_returns_valid_schema():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.",
        include={"normalized"},
    )

    assert isinstance(result, PreprocessResult)
    assert result.schema_version == "1.0"


def test_original_text_preserved_unchanged():
    text = "  Dr. Popescu  a   plecat.  "
    pipeline = PreprocessPipeline()
    result = pipeline.process(text, include={"normalized"})
    assert result.original_text == text


def test_normalized_text_matches_objectives_example():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.",
        include={"normalized"},
    )
    assert result.normalized_text == (
        "Doctor Popescu a trimis douăzeci și cinci de kilograme pe doisprezece "
        "iulie două mii douăzeci și șase, la ora paisprezece și treizeci de minute."
    )


def test_clean_only_does_not_run_normalizer():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Am 25 kg.", include={"clean"})

    assert result.clean_text == "Am 25 kg."
    assert result.normalized_text is None
    assert not any(entry.stage == "normalizer" for entry in result.trace)


def test_multi_sentence_text():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Nu pot să cred! Am reușit.", include={"normalized"})

    assert len(result.sentences) == 2
    assert result.sentences[0].text == "Nu pot să cred!"
    assert result.sentences[1].text == "Am reușit."


@pytest.mark.skipif(
    not _default_profile_stack_available(),
    reason="default profile now includes linguistic/phonemes/stress/emotion, "
    "which need the Stanza model, espeak, and the NRC lexicon cache",
)
def test_default_profile_matches_explicit_include():
    pipeline = PreprocessPipeline.from_profile("default")
    result = pipeline.process("Am 25 kg.")
    assert result.normalized_text == "Am douăzeci și cinci de kilograme."


def test_empty_input_does_not_crash():
    pipeline = PreprocessPipeline()
    result = pipeline.process("", include={"normalized"})
    assert result.original_text == ""
    assert result.clean_text == ""
    assert result.normalized_text == ""
    assert result.sentences == []


def test_schema_round_trips_through_json():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Bună ziua!", include={"normalized"})
    round_tripped = PreprocessResult.model_validate_json(result.model_dump_json())
    assert round_tripped == result
