"""Romanian phonemization: a project-overrides → espeak → grapheme-fallback
cascade, per readme.md section 2.3 and preprocess/objectives.md Phase 4.

Uses the classic `espeak` binary already available on this machine (via
`subprocess`), not the `espeak-ng` fork objectives.md names — see
src/expressive_tts/preprocess/README.md for why (no system package
changes needed; `espeak -v ro --ipa` already gives IPA phonemes with
stress marks). RoLEX (the primary lexicon objectives.md ranks above
eSpeak) is not integrated: licensing/distribution is unclear;
`_lookup_rolex` is a stub so it's a one-function change to add later.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.schemas import Provenance

PRODUCER = "phonemizer_espeak_v1"
ESPEAK_BINARY = "espeak"

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    path = _CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_overrides() -> dict:
    return _load_yaml("pronunciation_overrides.yaml")


@lru_cache(maxsize=None)
def phoneme_inventory() -> set[str]:
    data = _load_yaml("phoneme_inventory.yaml")
    symbols: set[str] = set()
    for category in ("suprasegmental", "modifier", "vowels", "consonants"):
        symbols.update(data.get(category) or [])
    return symbols


def unknown_phonemes(phonemes: str) -> set[str]:
    """Characters in `phonemes` (ignoring whitespace) not in the canonical
    inventory."""
    inventory = phoneme_inventory()
    return {ch for ch in phonemes if not ch.isspace() and ch not in inventory}


def espeak_available() -> bool:
    return shutil.which(ESPEAK_BINARY) is not None


def _lookup_rolex(word: str) -> str | None:
    """RoLEX integration point. Always returns None for now: RoLEX's
    licensing/distribution terms haven't been cleared for use in this
    project (see preprocess/objectives.md Phase 4 and the project plan).
    Fill this in once that's resolved; the cascade already falls through
    to espeak."""
    return None


def _espeak_single(word: str) -> str:
    try:
        result = subprocess.run(
            [ESPEAK_BINARY, "-v", "ro", "--ipa", "-q"],
            input=word,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def espeak_ipa_for_words(words: list[str]) -> list[str]:
    """IPA phonemes for each word in `words`, in order.

    One `espeak` call for the whole batch (one word per line) when the
    output line count matches the input — falls back to one call per word
    if it doesn't, so a batching quirk never silently misaligns words to
    phonemes.
    """
    if not words:
        return []
    if not espeak_available():
        return ["" for _ in words]

    try:
        result = subprocess.run(
            [ESPEAK_BINARY, "-v", "ro", "--ipa", "-q"],
            input="\n".join(words),
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        resolved = shutil.which(ESPEAK_BINARY) or ESPEAK_BINARY
        raise RuntimeError(
            f"eSpeak timed out after 15s ({resolved}). Test it with: "
            "`printf 'Bună\\nziua\\n' | timeout 5 espeak -v ro --ipa -q`. "
            "Install a working eSpeak binary and ensure it is first on PATH."
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        resolved = shutil.which(ESPEAK_BINARY) or ESPEAK_BINARY
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"eSpeak failed ({resolved}): {stderr.strip() or exc}"
        ) from exc

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if len(lines) == len(words):
        return lines
    return [_espeak_single(word) for word in words]


class PhonemizerProcessor:
    name = "phonemizer"
    version = PRODUCER
    provides = {"phonemes"}
    requires = {"tokens"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        overrides = config.get("pronunciation_overrides", default_overrides())

        phoneme_sentences: list[str] = []
        for sentence in document.sentence_spans:
            tokens = sentence.tokens or []
            word_tokens = [t for t in tokens if t.upos != "PUNCT"]
            word_texts = [t.text for t in word_tokens]
            espeak_results = espeak_ipa_for_words(word_texts)

            for token, phonemes in zip(word_tokens, espeak_results):
                key = token.text.lower()
                override = overrides.get(key)
                rolex_entry = None if override else _lookup_rolex(key)

                if override:
                    token.phonemes = override
                    token.pronunciation_provenance = Provenance.LEXICON
                    token.pronunciation_producer = "pronunciation_overrides_v1"
                    token.pronunciation_confidence = 1.0
                elif rolex_entry:
                    token.phonemes = rolex_entry
                    token.pronunciation_provenance = Provenance.LEXICON
                    token.pronunciation_producer = "rolex"
                    token.pronunciation_confidence = 1.0
                elif phonemes:
                    token.phonemes = phonemes
                    token.pronunciation_provenance = Provenance.PREDICTED
                    token.pronunciation_producer = "espeak_ro"
                    token.pronunciation_confidence = 0.85
                    unknown = unknown_phonemes(phonemes)
                    if unknown:
                        document.warnings.append(
                            f"unknown phoneme(s) {sorted(unknown)} in {token.text!r}"
                        )
                else:
                    token.phonemes = token.text.lower()
                    token.pronunciation_provenance = Provenance.FALLBACK
                    token.pronunciation_producer = "grapheme_fallback_v1"
                    token.pronunciation_confidence = 0.1
                    document.warnings.append(f"grapheme fallback used for {token.text!r}")

            phoneme_sentences.append(
                " ".join(token.phonemes if token.phonemes is not None else token.text for token in tokens)
            )

        document.phoneme_text = " ".join(phoneme_sentences)
