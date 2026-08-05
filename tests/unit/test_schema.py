import pytest
from pydantic import ValidationError

from expressive_tts.preprocess.schemas import (
    PreprocessResult,
    Provenance,
    Sentence,
    TraceEntry,
)


def test_minimal_result_is_valid():
    result = PreprocessResult(original_text="Bună ziua.")
    assert result.schema_version == "1.0"
    assert result.original_text == "Bună ziua."
    assert result.clean_text is None
    assert result.sentences == []
    assert result.trace == []


def test_original_text_is_required():
    with pytest.raises(ValidationError):
        PreprocessResult()


def test_full_round_trip():
    result = PreprocessResult(
        id="example-001",
        original_text="Nu pot să cred! Am reușit.",
        clean_text="Nu pot să cred! Am reușit.",
        normalized_text="Nu pot să cred! Am reușit.",
        sentences=[
            Sentence(text="Nu pot să cred!", start=0, end=15, sentence_type="exclamative"),
            Sentence(text="Am reușit.", start=16, end=26),
        ],
        trace=[
            TraceEntry(
                stage="cleaner",
                operation="whitespace",
                original="  ",
                replacement=" ",
                start=0,
                end=2,
                provenance=Provenance.RULE,
                producer="cleaner_v1",
            )
        ],
    )

    round_tripped = PreprocessResult.model_validate_json(result.model_dump_json())
    assert round_tripped == result
    assert round_tripped.sentences[0].sentence_type == "exclamative"


def test_provenance_values_match_readme_contract():
    assert {p.value for p in Provenance} == {
        "user", "source", "rule", "lexicon", "predicted", "generated", "fallback",
    }
