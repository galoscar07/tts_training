import pytest

from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.stress import (
    StressProcessor,
    fallback_stress_index,
    stressed_group_index,
    syllabify,
)
from expressive_tts.preprocess.schemas import Provenance, Token


@pytest.mark.parametrize(
    "word, expected",
    [
        ("fericire", ["fe", "ri", "ci", "re"]),  # objectives.md Phase 5 worked example
        ("reușit", ["reu", "șit"]),
        ("copii", ["co", "pii"]),
        ("copil", ["co", "pil"]),
        ("frumoasă", ["fru", "moa", "să"]),
        ("important", ["im", "por", "tant"]),
        ("cred", ["cred"]),
    ],
)
def test_syllabify(word, expected):
    assert syllabify(word) == expected


def test_syllabify_inseparable_onset_cluster():
    # "pl" (obstruent+liquid) stays with the following syllable.
    assert syllabify("aplic") == ["a", "plic"]


def test_syllabify_digraph_not_split():
    # "ch" must never split into "c"-"h".
    assert syllabify("ochi") == ["o", "chi"]


def test_syllabify_empty_string():
    assert syllabify("") == []


def test_syllabify_no_vowels_returns_whole_word():
    assert syllabify("xyz") == ["xyz"]


@pytest.mark.parametrize(
    "ipa, expected_group",
    [
        ("reuʃˈit", 1),
        ("kˈopiɪ", 0),
        ("fˌeɾitʃˈiɾe", 2),  # matches objectives.md's stress index 2 for "fericire"
        ("krˈed", 0),
        ("kopˈil", 1),
        ("frumˈɔasə", 1),
        ("ˌimportˈant", 2),
    ],
)
def test_stressed_group_index(ipa, expected_group):
    assert stressed_group_index(ipa) == expected_group


def test_stressed_group_index_no_stress_mark():
    assert stressed_group_index("kred") is None


@pytest.mark.parametrize(
    "syllables, word, expected",
    [
        (["cred"], "cred", 0),
        (["im", "por", "tant"], "important", 2),  # ends in consonant -> last
        (["fru", "moa", "să"], "frumoasă", 1),  # ends in vowel -> penultimate
    ],
)
def test_fallback_stress_index(syllables, word, expected):
    assert fallback_stress_index(syllables, word) == expected


def _document_with_token(token: Token, config_overrides: dict | None = None) -> PipelineDocument:
    span = SentenceSpan(text="x", start=0, end=1, tokens=[token])
    return PipelineDocument(original_text="x", sentence_spans=[span])


def test_process_uses_stress_override():
    token = Token(text="reușit", start=0, end=6, upos="VERB", phonemes="reuʃˈit")
    document = _document_with_token(token)
    override = {"reușit": {"syllables": ["re", "u", "șit"], "stressed_syllable_index": 2}}
    StressProcessor().process(document, {"stress_overrides": override})
    result = document.sentence_spans[0].tokens[0]
    assert result.syllables == ["re", "u", "șit"]
    assert result.stressed_syllable_index == 2
    assert result.stress_provenance == Provenance.LEXICON


def test_process_derives_stress_from_phonemes():
    token = Token(text="fericire", start=0, end=8, upos="NOUN", phonemes="fˌeɾitʃˈiɾe")
    document = _document_with_token(token)
    StressProcessor().process(document, {"stress_overrides": {}})
    result = document.sentence_spans[0].tokens[0]
    assert result.syllables == ["fe", "ri", "ci", "re"]
    assert result.stressed_syllable_index == 2
    assert result.stress_provenance == Provenance.PREDICTED


def test_process_falls_back_when_group_index_out_of_range():
    # Phonemes deliberately inconsistent with the grapheme syllable count:
    # 3 IPA vowel-nucleus groups (a, e, i) but "cred" is 1 grapheme syllable.
    token = Token(text="cred", start=0, end=4, upos="VERB", phonemes="ˈa ˈe ˈi")
    document = _document_with_token(token)
    StressProcessor().process(document, {"stress_overrides": {}})
    result = document.sentence_spans[0].tokens[0]
    assert result.stress_provenance == Provenance.FALLBACK
    assert result.stressed_syllable_index == 0  # single-syllable "cred"
    assert any("fell back" in w for w in document.warnings)


def test_process_skips_punctuation():
    token = Token(text=".", start=0, end=1, upos="PUNCT")
    document = _document_with_token(token)
    StressProcessor().process(document, {"stress_overrides": {}})
    result = document.sentence_spans[0].tokens[0]
    assert result.syllables == []
    assert result.stressed_syllable_index is None
