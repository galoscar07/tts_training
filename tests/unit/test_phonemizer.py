import pytest

from expressive_tts.preprocess.phonemizer import (
    PhonemizerProcessor,
    default_overrides,
    espeak_available,
    espeak_ipa_for_words,
    phonetics_only,
    phoneme_inventory,
    unknown_phonemes,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import Provenance, Token

requires_espeak = pytest.mark.skipif(not espeak_available(), reason="espeak binary not on PATH")


def test_phoneme_inventory_has_expected_categories():
    inventory = phoneme_inventory()
    assert "ˈ" in inventory  # primary stress
    assert "ʲ" in inventory  # palatalization
    assert "a" in inventory
    assert "ʃ" in inventory


def test_unknown_phonemes_detects_out_of_inventory_symbol():
    assert unknown_phonemes("krˈed") == set()
    assert unknown_phonemes("kr€ed") == {"€"}


def test_default_overrides_contains_tva():
    overrides = default_overrides()
    assert overrides["tva"] == "tˈe vˈe ˈa"


@requires_espeak
def test_espeak_ipa_for_words_alignment():
    results = espeak_ipa_for_words(["Nu", "pot", "să", "cred"])
    assert len(results) == 4
    assert all(results)
    assert "ˈ" in results[0] or "ˈ" in "".join(results)


def test_espeak_ipa_for_words_empty_input():
    assert espeak_ipa_for_words([]) == []


@requires_espeak
def test_phonetics_only_preserves_punctuation_and_stress():
    result = phonetics_only("Bună ziua!")
    assert result.endswith(" !")
    assert "ˈ" in result


def test_phonetics_only_applies_pronunciation_overrides(monkeypatch):
    import expressive_tts.preprocess.phonemizer as phonemizer_module

    monkeypatch.setattr(
        phonemizer_module, "espeak_ipa_for_words", lambda words: ["fallback"] * len(words)
    )
    assert phonetics_only("TVA.", {"tva": "tˈe vˈe ˈa"}) == "tˈe vˈe ˈa ."


def _make_document_with_tokens(word_tokens: list[Token]) -> PipelineDocument:
    span = SentenceSpan(text="x", start=0, end=1, tokens=word_tokens)
    return PipelineDocument(original_text="x", sentence_spans=[span])


def test_process_uses_override_before_espeak():
    document = _make_document_with_tokens([Token(text="TVA", start=0, end=3, upos="NOUN")])
    PhonemizerProcessor().process(document, {"pronunciation_overrides": {"tva": "tˈe vˈe ˈa"}})
    token = document.sentence_spans[0].tokens[0]
    assert token.phonemes == "tˈe vˈe ˈa"
    assert token.pronunciation_provenance == Provenance.LEXICON
    assert token.pronunciation_producer == "pronunciation_overrides_v1"


def test_process_skips_punctuation_tokens():
    document = _make_document_with_tokens(
        [Token(text="!", start=0, end=1, upos="PUNCT")]
    )
    PhonemizerProcessor().process(document, {"pronunciation_overrides": {}})
    token = document.sentence_spans[0].tokens[0]
    assert token.phonemes is None
    assert token.pronunciation_provenance is None


@requires_espeak
def test_process_uses_espeak_when_no_override():
    document = _make_document_with_tokens([Token(text="cred", start=0, end=4, upos="VERB")])
    PhonemizerProcessor().process(document, {"pronunciation_overrides": {}})
    token = document.sentence_spans[0].tokens[0]
    assert token.phonemes
    assert token.pronunciation_provenance == Provenance.PREDICTED
    assert token.pronunciation_producer == "espeak_ro"


def test_process_falls_back_to_grapheme_when_espeak_unavailable(monkeypatch):
    import expressive_tts.preprocess.phonemizer as phonemizer_module

    monkeypatch.setattr(phonemizer_module, "espeak_available", lambda: False)
    document = _make_document_with_tokens([Token(text="Cred", start=0, end=4, upos="VERB")])
    PhonemizerProcessor().process(document, {"pronunciation_overrides": {}})
    token = document.sentence_spans[0].tokens[0]
    assert token.phonemes == "cred"
    assert token.pronunciation_provenance == Provenance.FALLBACK
    assert any("grapheme fallback" in w for w in document.warnings)
