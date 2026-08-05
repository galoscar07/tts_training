"""Unit tests for the VITS symbol set. No Coqui / no models required."""

import pytest

from tts_training.frontend import symbols


def test_phoneme_symbols_nonempty_and_single_codepoint():
    syms = symbols.phoneme_symbols()
    assert syms  # inventory loaded
    # Every symbol must be one codepoint so Coqui's per-character tokenizer
    # maps one phoneme -> one id (affricates are sequences, not single syms).
    assert all(len(s) == 1 for s in syms), [s for s in syms if len(s) != 1]


def test_phoneme_symbols_are_unique():
    syms = symbols.phoneme_symbols()
    assert len(syms) == len(set(syms))


def test_symbol_set_includes_space_and_punctuation():
    full = symbols.symbol_set()
    assert " " in full
    for mark in ".,!?":
        assert mark in full
    assert set(symbols.phoneme_symbols()) <= full


def test_training_characters_include_space_and_manifest_fallbacks(tmp_path):
    manifest = tmp_path / "train.manifest"
    manifest.write_text(
        "wavs/a.wav|dˈa Ă ș|speaker|\n"
        "wavs/b.wav|tʃ...|speaker|\n",
        encoding="utf-8",
    )
    characters = symbols.training_characters([manifest])
    assert " " in characters
    assert "Ă" in characters
    assert "ș" in characters
    assert "ˈ" in characters
    # Punctuation has its own Coqui class and must not be duplicated here.
    assert "." not in characters


def test_characters_config_is_lazy_about_coqui():
    # Without coqui-tts installed, characters_config() must raise a clear
    # ImportError (not fail at module import). With it installed, it returns
    # a config object. Either outcome is acceptable; a silent None is not.
    try:
        import TTS  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            symbols.characters_config()
    else:  # pragma: no cover - only when coqui is installed
        assert symbols.characters_config() is not None
