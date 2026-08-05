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
def test_contrastive_construction_end_to_end():
    pipeline = PreprocessPipeline()
    result = pipeline.process("Nu e roșu, ci albastru.", include={"focus"})
    tokens = {t.text: t for t in result.sentences[0].tokens}
    assert tokens["roșu"].is_focus is False
    assert "contrastive_negation_suppressed" in tokens["roșu"].focus_rules
    assert tokens["albastru"].is_focus is True
    assert "corrective_construction" in tokens["albastru"].focus_rules


@requires_model
def test_all_caps_and_intensifier_end_to_end():
    pipeline = PreprocessPipeline()
    result = pipeline.process("E FOARTE frumos.", include={"focus"})
    tokens = {t.text: t for t in result.sentences[0].tokens}
    assert tokens["FOARTE"].is_focus is True
    assert tokens["frumos"].is_focus is True
    assert "intensifier_target" in tokens["frumos"].focus_rules


@requires_model
def test_user_focus_words_override_end_to_end():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Vreau cafea.", include={"focus"}, user_focus_words={"cafea"}
    )
    tokens = {t.text: t for t in result.sentences[0].tokens}
    assert tokens["cafea"].is_focus is True
    assert tokens["cafea"].focus_provenance.value == "user"


@requires_model
def test_focus_alone_does_not_require_phonemes_or_espeak():
    # focus.FocusProcessor.requires == {"linguistic"} only — requesting it
    # alone must not pull in phonemizer/stress/emotion (which need espeak
    # and the NRC lexicon cache).
    pipeline = PreprocessPipeline()
    result = pipeline.process("Nu pot să cred! Am reușit.", include={"focus"})
    assert result.phoneme_text is None
    for sentence in result.sentences:
        for token in sentence.tokens:
            if token.upos != "PUNCT":
                assert token.focus_producer == "focus_rules_v1"
