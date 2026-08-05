"""Romanian syllabification and lexical stress. See readme.md section 2.3
and preprocess/objectives.md Phase 5.

Important distinction (objectives.md): *lexical stress* (the stressed
syllable within a word) is separate from sentence-level *focus*
(`focus.py`, not yet implemented) — this module only handles the former.

Syllabification is a deterministic grapheme-based approximation: any
maximal run of vowel letters is treated as one syllable nucleus (handles
the common Romanian diphthongs/triphthongs — "ea", "oa", "ia", "eau", ... —
uniformly). This is a simplification: Romanian sometimes distinguishes a
genuine diphthong from a hiatus at a morpheme boundary (e.g. "reuși" is
phonetically "re-u-șit", not a diphthong, because "re-" is a prefix), which
a purely phonetic rule can't always get right. Verified against several
real words including objectives.md's own worked example ("fericire" ->
["fe","ri","ci","re"], stress index 2) and confirmed correct for that case
plus "reușit", "copil", "frumoasă", "important" (see tests).

Stress is derived from espeak's `ˈ` mark by counting IPA vowel-nucleus
groups up to the mark and matching that *ordinal position* against our own
grapheme-syllable nuclei — robust to the phoneme string not being
letter-aligned with the spelling. Falls back to a documented default-stress
heuristic (last syllable if the word ends in a consonant, penultimate if it
ends in a vowel — a common approximation, not a rule without exceptions)
when espeak has no answer or the ordinal mismatches the syllable count.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.schemas import Provenance

PRODUCER = "stress_ro_v1"
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"

_VOWELS = "aăâîeiouAĂÂÎEIOU"
_VOWEL_RUN = re.compile(f"[{_VOWELS}]+")

# Consonant clusters that stay together as the onset of the following
# syllable (obstruent+liquid onsets, plus the che/chi/ghe/ghi digraphs,
# which must never be split between the "h" and the preceding letter).
_INSEPARABLE_ONSETS = {
    "pl", "pr", "bl", "br", "tr", "dr", "cl", "cr", "gl", "gr", "fl", "fr", "vl", "vr",
    "ch", "gh",
}

_IPA_VOWELS = set("aeiouəɨɔɪʊy")


@lru_cache(maxsize=None)
def default_stress_overrides() -> dict:
    path = _CONFIG_DIR / "stress_overrides.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _cluster_split_point(cluster: str) -> int:
    """How many leading characters of the consonant `cluster` (the run
    between two vowel nuclei) stay with the *preceding* syllable."""
    n = len(cluster)
    if n <= 1:
        return 0
    if n == 2:
        return 0 if cluster in _INSEPARABLE_ONSETS else 1
    tail = cluster[-2:]
    return (n - 2) if tail in _INSEPARABLE_ONSETS else (n - 1)


def syllabify(word: str) -> list[str]:
    """Split `word` into syllables using vowel-nucleus detection."""
    if not word:
        return []

    lower = word.lower()
    nuclei = list(_VOWEL_RUN.finditer(lower))
    if not nuclei:
        return [word]

    boundaries = []
    for i in range(1, len(nuclei)):
        cluster = lower[nuclei[i - 1].end() : nuclei[i].start()]
        boundaries.append(nuclei[i - 1].end() + _cluster_split_point(cluster))

    points = [0, *boundaries, len(word)]
    return [word[points[i] : points[i + 1]] for i in range(len(points) - 1)]


def stressed_group_index(phonemes: str) -> int | None:
    """0-based ordinal, among IPA vowel-nucleus groups in `phonemes`, of
    the group marked with primary stress (`ˈ`). None if unmarked."""
    group_index = -1
    prev_was_vowel = False
    pending_stress = False
    stressed = None

    for char in phonemes:
        if char in ("ˈ", "ˌ"):
            if char == "ˈ":
                pending_stress = True
            prev_was_vowel = False
            continue
        is_vowel = char in _IPA_VOWELS
        if is_vowel and not prev_was_vowel:
            group_index += 1
            if pending_stress:
                stressed = group_index
                pending_stress = False
        prev_was_vowel = is_vowel

    return stressed


def fallback_stress_index(syllables: list[str], word: str) -> int:
    """Default Romanian stress heuristic: last syllable if the word ends
    in a consonant, penultimate if it ends in a vowel. A common
    approximation, not exception-free — used only when espeak gives no
    usable answer."""
    if len(syllables) <= 1:
        return 0
    ends_in_vowel = bool(word) and word[-1].lower() in _VOWELS.lower()
    return len(syllables) - 2 if ends_in_vowel else len(syllables) - 1


class StressProcessor:
    name = "stress"
    version = PRODUCER
    provides = {"syllables", "lexical_stress"}
    requires = {"phonemes"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        overrides = config.get("stress_overrides", default_stress_overrides())

        for sentence in document.sentence_spans:
            for token in sentence.tokens or []:
                if token.upos == "PUNCT":
                    continue

                key = token.text.lower()
                override = overrides.get(key)
                if override:
                    token.syllables = list(override["syllables"])
                    token.stressed_syllable_index = override["stressed_syllable_index"]
                    token.stress_provenance = Provenance.LEXICON
                    token.stress_producer = "stress_overrides_v1"
                    token.stress_confidence = 1.0
                    continue

                syllables = syllabify(token.text)
                token.syllables = syllables

                group_index = stressed_group_index(token.phonemes or "")
                if group_index is not None and group_index < len(syllables):
                    token.stressed_syllable_index = group_index
                    token.stress_provenance = Provenance.PREDICTED
                    token.stress_producer = "espeak_ro"
                    token.stress_confidence = 0.8
                else:
                    token.stressed_syllable_index = fallback_stress_index(syllables, token.text)
                    token.stress_provenance = Provenance.FALLBACK
                    token.stress_producer = "default_stress_heuristic_v1"
                    token.stress_confidence = 0.3
                    document.warnings.append(
                        f"lexical stress fell back to the default heuristic for {token.text!r}"
                    )
