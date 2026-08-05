from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import (
    ModelNotAvailableError,
    _get_pipeline,
    _parse_feats,
    analyze,
    infer_negation,
    infer_sentence_type,
    to_tokens,
)
from expressive_tts.preprocess.schemas import Token


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(
    not _model_available(), reason="Trankit Romanian model not downloaded"
)


def make_token(**overrides) -> Token:
    fields = dict(text="x", start=0, end=1, deprel=None, feats={}, upos=None)
    fields.update(overrides)
    return Token(**fields)


def test_parse_feats_empty():
    assert _parse_feats(None) == {}
    assert _parse_feats("") == {}


def test_parse_feats_multiple():
    assert _parse_feats("Mood=Ind|Number=Plur|Person=3") == {
        "Mood": "Ind",
        "Number": "Plur",
        "Person": "3",
    }


def test_to_tokens_marks_interjection():
    words = [
        {
            "text": "Uau",
            "start_char": 0,
            "end_char": 3,
            "lemma": "uau",
            "upos": "INTJ",
            "xpos": None,
            "feats": None,
            "head": 0,
            "deprel": "root",
        }
    ]
    tokens = to_tokens(words)
    assert tokens[0].is_interjection is True
    assert tokens[0].upos == "INTJ"


def test_infer_sentence_type_interrogative():
    tokens = [make_token(text="?", deprel="punct")]
    assert infer_sentence_type(tokens, "Ce faci?") == "interrogative"


def test_infer_sentence_type_exclamative():
    tokens = [make_token(text="!", deprel="punct")]
    assert infer_sentence_type(tokens, "Ce bine!") == "exclamative"


def test_infer_sentence_type_declarative():
    tokens = [make_token(text=".", deprel="punct")]
    assert infer_sentence_type(tokens, "Plouă.") == "declarative"


def test_infer_sentence_type_incomplete_no_terminal_punctuation():
    tokens = [make_token(text="Plouă", deprel="root")]
    assert infer_sentence_type(tokens, "Plouă") == "incomplete"


def test_infer_sentence_type_imperative_from_root_mood():
    tokens = [
        make_token(text="Ascultă", deprel="root", feats={"Mood": "Imp"}),
        make_token(text="!", deprel="punct"),
    ]
    # Imperative mood wins even though the sentence also ends in "!".
    assert infer_sentence_type(tokens, "Ascultă!") == "imperative"


def test_infer_negation_true():
    tokens = [make_token(text="Nu", feats={"Polarity": "Neg"})]
    assert infer_negation(tokens) is True


def test_infer_negation_false():
    tokens = [make_token(text="Da", feats={})]
    assert infer_negation(tokens) is False


@requires_model
def test_analyze_real_sentence():
    words = analyze("Nu pot să cred!")
    texts = [w["text"] for w in words]
    assert texts == ["Nu", "pot", "să", "cred", "!"]
    negation_word = next(w for w in words if w["text"] == "Nu")
    assert negation_word["feats"] == "Polarity=Neg"


@requires_model
def test_analyze_uses_cache(tmp_path, monkeypatch):
    import expressive_tts.preprocess.linguistic as linguistic_module

    monkeypatch.setattr(linguistic_module, "CACHE_DIR", tmp_path)
    words = analyze("Plouă.")
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1

    # Second call should hit the cache (same content).
    words_again = analyze("Plouă.")
    assert words == words_again


@requires_model
def test_pipeline_root_verb_imperative_real():
    words = analyze("Ascultă-mă!")
    tokens = to_tokens(words)
    assert infer_sentence_type(tokens, "Ascultă-mă!") == "imperative"
