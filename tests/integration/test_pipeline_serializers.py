import re
from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.pipeline import PreprocessPipeline
from expressive_tts.preprocess.serializers import (
    parse_control_tokens,
    to_annotated_text,
    to_control_tokens,
    to_ssml_like,
)


@lru_cache(maxsize=None)
def _model_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return True


requires_model = pytest.mark.skipif(not _model_available(), reason="Stanza Romanian model not downloaded")


@requires_model
def test_control_tokens_real_sentence_end_to_end():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Nu pot să cred! Am reușit.", include={"emotion", "prosody", "focus", "phonemes"}
    )
    output = to_control_tokens(result)
    assert output.count("[SENT_") == 2
    assert "[BREAK_" in output or "[BREAK_UNSPECIFIED]" in output
    parsed = parse_control_tokens(output)
    assert len(parsed) == 2


@requires_model
def test_serialize_control_tokens_param_populates_tts_token_text():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Uau, ce veste minunată!", include={"emotion", "prosody"}, serialize_control_tokens=True
    )
    assert result.tts_token_text is not None
    # An emotion control token is emitted; the exact category is the
    # transformer's call (here: happy), so assert on the token shape.
    assert re.search(r"\[EMO_[A-Z]+\]", result.tts_token_text)


@requires_model
def test_serialization_deterministic_across_two_runs():
    pipeline = PreprocessPipeline()
    r1 = pipeline.process("Ce panică, nu mai suport!", include={"emotion", "prosody", "focus"})
    r2 = pipeline.process("Ce panică, nu mai suport!", include={"emotion", "prosody", "focus"})
    assert to_control_tokens(r1) == to_control_tokens(r2)
    assert to_annotated_text(r1) == to_annotated_text(r2)
    assert to_ssml_like(r1) == to_ssml_like(r2)


@requires_model
def test_all_serializers_succeed_on_formal_legal_sentence():
    pipeline = PreprocessPipeline()
    result = pipeline.process(
        "Astfel, valoarea punctului de pensie se va determina conform prevederilor legale.",
        include={"emotion", "prosody", "focus", "interjections"},
    )
    # must not raise for any serializer, even on formal/no-interjection text
    to_control_tokens(result)
    to_annotated_text(result)
    to_ssml_like(result)
