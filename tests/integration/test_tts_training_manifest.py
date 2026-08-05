"""Integration test for the manifest builder: runs the real preprocess
pipeline over a few MARA utterances. Gated on the linguistic backend +
espeak being available (same pattern as the other pipeline integration
tests); no Coqui needed."""

from functools import lru_cache

import pytest

from expressive_tts.preprocess.linguistic import ModelNotAvailableError, _get_pipeline
from expressive_tts.preprocess.phonemizer import espeak_available
from tts_training import paths
from tts_training.data.manifest import build_manifest


@lru_cache(maxsize=1)
def _stack_available() -> bool:
    try:
        _get_pipeline()
    except ModelNotAvailableError:
        return False
    return espeak_available()


def _mara_present() -> bool:
    base = paths.datasets_dir()
    return base is not None and (base / "mara" / "metadata.csv").exists()


requires_stack = pytest.mark.skipif(
    not _stack_available(),
    reason="requires the linguistic backend (Trankit/Stanza) and espeak",
)
requires_mara = pytest.mark.skipif(not _mara_present(), reason="datasets/mara not present")


@requires_stack
@requires_mara
def test_build_manifest_over_mara_subset(tmp_path):
    out = tmp_path / "mara.manifest"
    stats = build_manifest("mara", out, limit=5)

    assert stats.written == 5
    assert stats.unexpected_symbols == set()  # every phoneme within the inventory

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        audio, phonemes, speaker, emotion = line.split("|")
        assert audio.endswith(".wav")
        assert phonemes.strip()          # non-empty phoneme sequence
        assert speaker == "mara"
        assert emotion == ""             # base build carries no emotion label


@requires_stack
@requires_mara
def test_limit_caps_written_rows(tmp_path):
    stats = build_manifest("mara", tmp_path / "m.manifest", limit=3)
    assert stats.written == 3


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        build_manifest("nope", tmp_path / "x.manifest")


def test_explicit_corpus_root_overrides_resolution(tmp_path):
    # A made-up dataset key isn't in the registry, but the point here is that
    # an explicit, non-existent corpus root surfaces a clear FileNotFoundError
    # rather than silently falling back to ./datasets.
    with pytest.raises((ValueError, FileNotFoundError)):
        build_manifest("mara", tmp_path / "m.manifest", corpus_root=tmp_path / "nope")
