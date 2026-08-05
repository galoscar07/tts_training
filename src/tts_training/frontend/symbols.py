"""The VITS symbol set, taken from the project's canonical phoneme inventory.

Because `expressive_tts.preprocess` pre-phonemizes text to IPA, the "text"
Coqui sees is already a phoneme string. We build Coqui's `characters`
vocabulary directly from the inventory instead of using Coqui's grapheme
cleaners/phonemizer. Every inventory symbol is a single Unicode codepoint
(affricates are consonant *sequences*, e.g. `tʃ`/`dʒ`), so Coqui's default
per-character tokenizer maps each phoneme to exactly one id.

The inventory is read through `expressive_tts.preprocess.phonemizer`
(`phoneme_inventory()`) — a package call, not a repo-relative file read — so
this module has no dependency on the repository's directory layout and moves
cleanly with the rest of `tts_training`. The symbol order is `sorted()`, i.e.
deterministic, so token ids are stable across runs.

No Coqui dependency for `phoneme_symbols()`/`symbol_set()` (the manifest
builder validates coverage locally); only `characters_config()` imports
Coqui, lazily.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from expressive_tts.preprocess.phonemizer import phoneme_inventory

# Punctuation the preprocess phonemizer preserves verbatim in `phoneme_text`
# (PUNCT tokens keep their literal text). The VITS tokenizer treats these as
# its punctuation class.
PUNCTUATION = '.,!?;:…"«»()-–—\''

BLANK = "<BLNK>"
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"


@lru_cache(maxsize=1)
def phoneme_symbols() -> list[str]:
    """Every phoneme/suprasegmental/modifier symbol the pipeline can emit, in
    a deterministic (sorted) order so Coqui token ids are stable."""
    return sorted(phoneme_inventory())


def symbol_set() -> set[str]:
    """Phonemes + punctuation + space — every token that may legitimately
    appear in a `phoneme_text` string. Used by the manifest builder to catch
    anything unexpected before it reaches training."""
    return set(phoneme_symbols()) | set(PUNCTUATION) | {" "}


def symbols_from_manifests(manifest_paths: Iterable[str | Path]) -> set[str]:
    """Return every character observed in the phoneme column of manifests."""
    observed: set[str] = set()
    for manifest_path in manifest_paths:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            for raw in handle:
                parts = raw.rstrip("\n").split("|", 2)
                if len(parts) >= 2:
                    observed.update(parts[1])
    return observed


def training_characters(manifest_paths: Iterable[str | Path] = ()) -> str:
    """Characters class used by VITS.

    It contains the canonical IPA inventory, an explicit word-space token,
    and every non-punctuation character actually observed in the supplied
    manifests. The latter safely covers grapheme fallbacks such as Romanian
    ``Ă``/``ș`` without guessing a partial alphabet.
    """
    observed = symbols_from_manifests(manifest_paths)
    characters = set(phoneme_symbols()) | {" "} | (observed - set(PUNCTUATION))
    return "".join(sorted(characters))


def characters_config(manifest_paths: Iterable[str | Path] = ()):
    """Build Coqui's `CharactersConfig` from inventory + actual manifests.

    Imports Coqui lazily — only needed at training time on the GPU box.
    """
    try:
        from TTS.tts.configs.shared_configs import CharactersConfig
    except ImportError as exc:  # pragma: no cover - only hit without Coqui
        raise ImportError(
            "coqui-tts is not installed; install the training extra "
            "(`pip install -e '.[training]'`) on the training machine."
        ) from exc

    return CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.Graphemes",
        pad=PAD,
        eos=EOS,
        bos=BOS,
        blank=BLANK,
        characters=training_characters(manifest_paths),
        punctuations=PUNCTUATION,
        phonemes=None,  # we pre-phonemize; Coqui's phonemizer stays off
    )
