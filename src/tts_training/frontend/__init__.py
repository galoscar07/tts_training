"""Frontend glue: the phoneme symbol set the VITS tokenizer uses.

The `expressive_tts.preprocess` pipeline already converts text to IPA phoneme
strings (with stress marks), so Coqui's own phonemizer is switched off
(`use_phonemes=False`) and these symbols are consumed directly.
"""

from tts_training.frontend.symbols import (
    PUNCTUATION,
    characters_config,
    phoneme_symbols,
)

__all__ = ["PUNCTUATION", "characters_config", "phoneme_symbols"]
