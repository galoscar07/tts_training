"""Edge-case robustness tests (preprocess/objectives.md Phase 13's
remaining unchecked items). Each asserts the full pipeline completes
without raising and produces a schema-valid `PreprocessResult` — this is
not fuzz coverage (a finite, hand-picked set of edge cases), which is
noted honestly in objectives.md rather than claimed as exhaustive.
"""

from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.schemas import PreprocessResult

ALL_LAYERS = {
    "clean",
    "sentences",
    "normalized",
    "linguistic",
    "phonemes",
    "syllables",
    "emotion",
    "focus",
    "prosody",
    "interjections",
}


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(
    not _model_available(), reason="linguistic backend (Trankit/Stanza) not downloaded"
)


def _run_full(text: str) -> PreprocessResult:
    return PreprocessPipeline().process(text, include=ALL_LAYERS)


@requires_model
def test_malformed_unicode_lone_surrogate_replacement_char():
    text = "Bun� venit! Ce mai faci�?"
    result = _run_full(text)
    assert isinstance(result, PreprocessResult)


@requires_model
def test_malformed_unicode_combining_characters():
    text = "Ta̦re bine, mult̃umesc!"  # combining marks stacked on ordinary letters
    result = _run_full(text)
    assert isinstance(result, PreprocessResult)


@requires_model
def test_very_long_paragraph():
    sentence = "Astăzi este o zi frumoasă și senină, iar oamenii se plimbă liniștiți pe stradă. "
    text = sentence * 40
    result = _run_full(text)
    assert len(result.sentences) >= 40


@requires_model
def test_code_switching_romanian_english():
    text = "Am avut un meeting foarte productiv, apoi am mers la coffee break cu echipa."
    result = _run_full(text)
    tokens = [t for s in result.sentences for t in s.tokens]
    assert tokens  # tokenized without crashing despite embedded English words


@requires_model
def test_unsupported_symbols_emoji_and_control_chars():
    text = "Sunt fericit azi \U0001f600\U0001f389! \x0bText cu tab\x09si control char."
    result = _run_full(text)
    assert isinstance(result, PreprocessResult)


@requires_model
def test_formal_academic_text_full_pipeline():
    text = (
        "Astfel, valoarea punctului de pensie se va determina conform "
        "prevederilor legale, cu excepția celor stagiari, în condițiile legii."
    )
    result = _run_full(text)
    assert result.document_style == "formal"
    assert all(not s.interjection_suggestions for s in result.sentences)


@requires_model
@requires_model
def test_short_exclamation_yields_well_formed_emotion():
    # "Ce panică!" — a short, punctuation-heavy exclamation. The transformer
    # emotion layer must return a well-formed label without crashing on the
    # terse input (it classifies this as fear). The earlier rule-based
    # baseline abstained here on a tied NRC-lexicon weight; the transformer
    # instead makes a confident call, so we assert well-formedness, not a
    # specific abstention.
    result = PreprocessPipeline().process("Ce panică!", include={"emotion"})
    emotion = result.sentences[0].emotion
    assert emotion.label in {
        "happy", "angry", "sad", "fear", "surprise", "neutral", "unspecified"
    }
    assert emotion.provenance.value == "predicted"


@requires_model
def test_existing_interjection_end_to_end():
    result = _run_full("Uau, ce veste minunată!")
    tokens = [t for s in result.sentences for t in s.tokens]
    assert any(t.is_interjection for t in tokens)


@requires_model
def test_repeated_punctuation():
    text = "Ce???!!! Chiar nu știi???"
    result = _run_full(text)
    assert isinstance(result, PreprocessResult)
    assert len(result.sentences) >= 1


@requires_model
def test_empty_and_whitespace_only_still_validate():
    for text in ["", "   ", "\n\n"]:
        result = PreprocessPipeline().process(text, include={"normalized"})
        assert isinstance(result, PreprocessResult)
