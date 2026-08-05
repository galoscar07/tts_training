"""One explicit end-to-end test per registered pipeline layer
(preprocess/objectives.md Phase 13: "Add integration tests for every
pipeline stage"). Each requests exactly that layer via `include` and
asserts the fields it owns actually populate — closing the gap the
Phase 13 checklist flagged (only clean/sentences/normalized were
indirectly covered before this file existed).
"""

from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline

TEXT = "Uau, nu pot să cred! Am reușit."


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(not _model_available(), reason="Stanza Romanian model not downloaded")


def test_clean_layer():
    result = PreprocessPipeline().process(TEXT, include={"clean"})
    assert result.clean_text is not None


def test_sentences_layer():
    result = PreprocessPipeline().process(TEXT, include={"sentences"})
    assert len(result.sentences) == 2
    assert all(s.start < s.end for s in result.sentences)


def test_normalized_layer():
    result = PreprocessPipeline().process("Am 2 mere.", include={"normalized"})
    assert result.normalized_text is not None
    assert "2" not in result.normalized_text


@requires_model
def test_linguistic_layer():
    result = PreprocessPipeline().process(TEXT, include={"linguistic"})
    tokens = result.sentences[0].tokens
    assert tokens
    assert any(t.lemma is not None for t in tokens)
    assert any(t.upos is not None for t in tokens)


@requires_model
def test_phonemes_layer():
    result = PreprocessPipeline().process(TEXT, include={"phonemes"})
    assert result.phoneme_text is not None
    assert any(t.phonemes is not None for s in result.sentences for t in s.tokens)


@requires_model
def test_syllables_stress_layer():
    result = PreprocessPipeline().process(TEXT, include={"syllables"})
    tokens = [t for s in result.sentences for t in s.tokens]
    assert any(t.syllables for t in tokens)
    assert any(t.stressed_syllable_index is not None for t in tokens)


@requires_model
def test_emotion_layer():
    result = PreprocessPipeline().process(TEXT, include={"emotion"})
    assert result.sentences[0].emotion is not None
    assert result.sentences[0].emotion.label


@requires_model
def test_focus_layer():
    result = PreprocessPipeline().process(TEXT, include={"focus"})
    tokens = [t for s in result.sentences for t in s.tokens]
    assert any(t.focus_score is not None for t in tokens)


@requires_model
def test_prosody_layer():
    result = PreprocessPipeline().process(TEXT, include={"prosody"})
    assert all(s.prosody is not None for s in result.sentences)


@requires_model
def test_interjections_layer_default_disabled():
    result = PreprocessPipeline().process(TEXT, include={"interjections"})
    # disabled by default (Phase 10 acceptance criterion) — layer runs, produces nothing
    assert all(s.interjection_suggestions == [] for s in result.sentences)


@requires_model
def test_interjections_layer_suggest_mode():
    result = PreprocessPipeline().process(
        "Am câștigat marele premiu!", include={"interjections"}, interjection_mode="suggest"
    )
    assert result.document_style is not None
