"""Focus and emphasis detection: which words in a sentence carry
sentence-level prominence. See readme.md section 2.4 and
preprocess/objectives.md Phase 8.

Distinct from lexical stress (Phase 5, `stress.py`) — that's *which
syllable within a word* is stressed; this is *which words in the
sentence* are emphasized. objectives.md requires the two stay separate
output fields (`Token.stressed_syllable_index` vs. `Token.focus_score`).

Each signal below is independent and additive, and every contribution
records its rule name in `Token.focus_rules` so a decision is always
explainable, per objectives.md's "Store evidence and confidence".
"""

from __future__ import annotations

from collections import Counter

from expressive_tts.preprocess.lexicons import (
    LexiconNotAvailableError,
    default_intensifiers,
    load_emolex,
)
from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.schemas import Provenance, Token

PRODUCER = "focus_rules_v1"

FOCUS_THRESHOLD = 0.5
FUNCTION_WORD_GUARD_THRESHOLD = 0.8  # function words need strong evidence to count

# UPOS tags treated as function words — excluded from most positive
# signals, and zeroed out below FUNCTION_WORD_GUARD_THRESHOLD even if some
# signal did fire (objectives.md: "Prevent function words from receiving
# focus without strong evidence").
_FUNCTION_UPOS = {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "PUNCT"}

_ALL_CAPS_SCORE = 0.9
_INTENSIFIER_TARGET_SCORE = 0.6
_REPETITION_SCORE = 0.4
_EMOTION_BEARING_SCORE = 0.3
_CORRECTIVE_SCORE = 0.7
_CONTRASTIVE_SUPPRESSION_FACTOR = 0.3
_MAIN_PREDICATE_SCORE = 0.15

_CONTRASTIVE_CONJUNCTION = "ci"  # Romanian "but rather" (corrective coordination)


def _key(token: Token) -> str:
    return (token.lemma or token.text).lower()


def _is_all_caps(token: Token) -> bool:
    return len(token.text) > 1 and token.text.isalpha() and token.text.isupper()


def _intensifier_precedes(tokens: list[Token], index: int, intensifiers: dict) -> bool:
    if index >= 2:
        two = f"{_key(tokens[index - 2])} {_key(tokens[index - 1])}"
        if two in intensifiers:
            return True
    if index >= 1:
        return _key(tokens[index - 1]) in intensifiers
    return False


def _contrastive_span(tokens: list[Token]) -> tuple[int, int] | None:
    """Index of a negation token and a later "ci" (but-rather) token, if
    both are present — the Romanian "nu X, ci Y" corrective construction."""
    negation_index = next((i for i, t in enumerate(tokens) if t.feats.get("Polarity") == "Neg"), None)
    if negation_index is None:
        return None
    ci_index = next(
        (i for i, t in enumerate(tokens) if i > negation_index and _key(t) == _CONTRASTIVE_CONJUNCTION),
        None,
    )
    if ci_index is None:
        return None
    return negation_index, ci_index


def score_focus(
    tokens: list[Token],
    *,
    emolex: dict[str, list[str]],
    intensifiers: dict,
) -> None:
    """Compute and set `focus_*` fields on every non-punctuation token in
    `tokens`, in place."""
    lemma_counts = Counter(_key(t) for t in tokens if t.upos != "PUNCT")
    contrastive = _contrastive_span(tokens)

    root_index = next(
        (i for i, t in enumerate(tokens) if t.deprel == "root" and t.upos == "VERB"), None
    )

    for i, token in enumerate(tokens):
        if token.upos == "PUNCT":
            continue

        score = 0.0
        rules: list[str] = []

        if _is_all_caps(token):
            score += _ALL_CAPS_SCORE
            rules.append("all_caps")

        if _intensifier_precedes(tokens, i, intensifiers):
            score += _INTENSIFIER_TARGET_SCORE
            rules.append("intensifier_target")

        if token.upos not in _FUNCTION_UPOS and lemma_counts[_key(token)] >= 2:
            score += _REPETITION_SCORE
            rules.append("repetition")

        if token.upos not in _FUNCTION_UPOS and _key(token) in emolex:
            score += _EMOTION_BEARING_SCORE
            rules.append("emotion_bearing")

        if contrastive is not None:
            negation_index, ci_index = contrastive
            if negation_index < i < ci_index:
                score *= _CONTRASTIVE_SUPPRESSION_FACTOR
                rules.append("contrastive_negation_suppressed")
            elif i > ci_index:
                score += _CORRECTIVE_SCORE
                rules.append("corrective_construction")

        if i == root_index:
            score += _MAIN_PREDICATE_SCORE
            rules.append("main_predicate")

        if token.upos in _FUNCTION_UPOS and score < FUNCTION_WORD_GUARD_THRESHOLD:
            score = 0.0
            rules = []

        score = min(score, 1.0)
        token.focus_score = round(score, 3)
        token.is_focus = score >= FOCUS_THRESHOLD
        token.focus_provenance = Provenance.RULE
        token.focus_producer = PRODUCER
        token.focus_rules = rules


def apply_user_focus(tokens: list[Token], user_focus_words: set[str]) -> None:
    """Explicit user-provided emphasis overrides every rule-based signal
    with full priority (objectives.md: "Explicitly marked focus has 100%
    priority over predictions")."""
    for token in tokens:
        if token.upos == "PUNCT":
            continue
        if _key(token) in user_focus_words or token.text.lower() in user_focus_words:
            token.focus_score = 1.0
            token.is_focus = True
            token.focus_provenance = Provenance.USER
            token.focus_producer = "user"
            token.focus_rules = ["user_provided"]


class FocusProcessor:
    name = "focus"
    version = PRODUCER
    provides = {"focus"}
    requires = {"linguistic"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        intensifiers = config.get("intensifiers", default_intensifiers())
        emolex = config.get("emolex")
        if emolex is None:
            try:
                emolex = load_emolex()
            except LexiconNotAvailableError:
                emolex = {}
        user_focus_words = {w.lower() for w in (config.get("user_focus_words") or [])}

        for sentence in document.sentence_spans:
            tokens = sentence.tokens or []
            score_focus(tokens, emolex=emolex, intensifiers=intensifiers)
            if user_focus_words:
                apply_user_focus(tokens, user_focus_words)
