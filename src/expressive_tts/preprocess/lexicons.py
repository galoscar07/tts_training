"""Shared lexical resources: the NRC EmoLex/VAD caches and the YAML
configuration lexicons (intensifiers, diminishers, interjection emotions).

Extracted from `emotion.py` when the emotion layer moved from a rule-based
lexicon baseline to a transformer classifier (preprocess/objectives.md
Phase 6). These resources are still needed by other rule-based layers that
were *not* replaced:

- `focus.py` (Phase 8) uses `load_emolex()` for emotion-bearing word
  detection and `default_intensifiers()` for intensifier-target scoring;
- `interjections.py` (Phase 10) uses `default_interjection_emotions()`.

Keeping them here (rather than in `emotion.py`) means those layers no
longer import from the emotion module, whose transformer dependency is
heavier and conceptually unrelated.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"
_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "lexicons"


class LexiconNotAvailableError(RuntimeError):
    """The NRC lexicon cache isn't present yet."""


def _require_cache_file(name: str) -> Path:
    path = _CACHE_DIR / name
    if not path.exists():
        raise LexiconNotAvailableError(
            f"{path} not found; run `./.venv/bin/python scripts/fetch_emotion_lexicon.py` first"
        )
    return path


@lru_cache(maxsize=None)
def load_emolex() -> dict[str, list[str]]:
    return json.loads(_require_cache_file("nrc_emolex_ro.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_vad() -> dict[str, dict[str, float]]:
    return json.loads(_require_cache_file("nrc_vad_ro.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    path = _CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_intensifiers() -> dict:
    return _load_yaml("intensifiers.yaml")


def default_diminishers() -> dict:
    return _load_yaml("diminishers.yaml")


def default_interjection_emotions() -> dict:
    return _load_yaml("interjection_emotions.yaml")
