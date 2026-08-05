"""Linguistic analysis: tokens, lemmas, UPOS, morphology, dependency
relations, negation, and sentence type. See readme.md's `tokens`/`linguistic`
layers and preprocess/objectives.md Phase 3.

Two interchangeable backends, both free and local, both emitting the *full*
linguistic stack (tokenize / POS / morphology / lemma / dependency parsing)
so the downstream fields other processors depend on (lemma for the
phonemizer, `feats` for negation/imperative detection, `deprel` for
focus/prosody) keep flowing unchanged:

- **Trankit** (preferred) — transformer-based, XLM-RoBERTa via HuggingFace
  `transformers` (nlp-uoregon/trankit). This is the "free AI/transformer"
  POS tagger.
- **Stanza** (fallback) — Stanford's neural pipeline.

The backend is picked lazily at first use: Trankit if it imports and loads,
otherwise Stanza. This matters in practice because Trankit 1.1.1's vendored
`adapter_transformers` crashes on import under Python 3.11+ (mutable dataclass
default), so on a modern interpreter the layer transparently runs on Stanza —
still a free neural model with the same output shape — instead of failing.
Both backends normalize to one word-dict shape, and the disk cache is keyed
by backend so their outputs never mix.

`requires={"normalized"}`, not just clean/sentences: we parse the
*normalized* text (numbers spelled out, abbreviations expanded, ...), per
readme.md's dependency chain example
`clean → normalized → tokens → phonemes → syllables → lexical_stress`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.schemas import Token

_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = _ROOT / ".cache" / "linguistic"
MODEL_CACHE_DIR = _ROOT / ".cache" / "models" / "trankit"
LANGUAGE = "romanian"  # Trankit language name

PRODUCER_BY_BACKEND = {"trankit": "trankit_ro_v1", "stanza": "stanza_ro_v1"}

_TERMINAL_SENTENCE_TYPE = {".": "declarative", "!": "exclamative", "?": "interrogative", "…": "declarative"}

_pipeline_instance = None  # lazy singleton — constructing either pipeline is expensive
_backend: str | None = None  # "trankit" | "stanza", chosen on first use


class ModelNotAvailableError(RuntimeError):
    """No linguistic backend (Trankit or Stanza) could be loaded."""


def _init_backend():
    """Pick and construct a backend once. Prefer Trankit (transformer); fall
    back to Stanza (neural) if Trankit can't import/load — e.g. under Python
    3.13, where trankit 1.1.1 crashes on import."""
    global _pipeline_instance, _backend
    if _pipeline_instance is not None:
        return

    errors: list[str] = []

    try:
        from trankit import Pipeline

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _pipeline_instance = Pipeline(LANGUAGE, cache_dir=str(MODEL_CACHE_DIR))
        _backend = "trankit"
        return
    except Exception as exc:  # ImportError on 3.13, or model/network errors
        errors.append(f"trankit unavailable: {exc}")

    try:
        import stanza

        _pipeline_instance = stanza.Pipeline(
            "ro", processors="tokenize,pos,lemma,depparse", verbose=False
        )
        _backend = "stanza"
        return
    except Exception as exc:
        errors.append(f"stanza unavailable: {exc}")

    raise ModelNotAvailableError(
        "no linguistic backend available:\n  "
        + "\n  ".join(errors)
        + "\nInstall one and download its model: "
        "`scripts/download_trankit_model.py` (transformer, needs Python <=3.12) "
        "or `scripts/download_stanza_model.py` (neural, works on 3.13)."
    )


def _get_pipeline():
    _init_backend()
    return _pipeline_instance


def active_backend() -> str:
    """The backend actually in use (constructs the pipeline if needed)."""
    _init_backend()
    assert _backend is not None
    return _backend


def _cache_path(text: str) -> Path:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Key by backend so Trankit and Stanza outputs never get mixed in cache.
    return CACHE_DIR / f"{active_backend()}_{digest}.json"


def _parse_feats(feats: str | None) -> dict[str, str]:
    if not feats or feats == "_":
        return {}
    result = {}
    for pair in feats.split("|"):
        key, _, value = pair.partition("=")
        result[key] = value
    return result


def _word_dict(text, start, end, lemma, upos, xpos, feats, head, deprel) -> dict:
    return {
        "text": text,
        "start_char": start,
        "end_char": end,
        "lemma": lemma,
        "upos": upos,
        "xpos": xpos,
        "feats": feats,
        "head": head,
        "deprel": deprel,
    }


def _trankit_span(item: dict) -> tuple[int | None, int | None]:
    span = item.get("dspan") or item.get("span")
    if span and len(span) == 2:
        return int(span[0]), int(span[1])
    return None, None


def _analyze_trankit(pipeline, text: str) -> list[dict]:
    doc = pipeline(text)
    words: list[dict] = []
    for sentence in doc.get("sentences", []):
        for token in sentence.get("tokens", []):
            start, end = _trankit_span(token)
            # Multi-word tokens (e.g. "într-un") expand to the syntactic words
            # that carry UPOS/feats/deprel; they inherit the surface span.
            members = token.get("expanded") or [token]
            for member in members:
                words.append(
                    _word_dict(
                        member.get("text"), start, end,
                        member.get("lemma"), member.get("upos"), member.get("xpos"),
                        member.get("feats"), member.get("head"), member.get("deprel"),
                    )
                )
    return words


def _analyze_stanza(pipeline, text: str) -> list[dict]:
    doc = pipeline(text)
    return [
        _word_dict(
            word.text, word.start_char, word.end_char,
            word.lemma, word.upos, word.xpos, word.feats, word.head, word.deprel,
        )
        for sentence in doc.sentences
        for word in sentence.words
    ]


def analyze(text: str) -> list[dict]:
    """Return raw word annotations for one sentence via the active backend,
    using the disk cache (keyed by backend + sha256 of `text`) when
    available."""
    if not text.strip():
        return []

    cache_path = _cache_path(text)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    pipeline = _get_pipeline()
    if _backend == "trankit":
        words = _analyze_trankit(pipeline, text)
    else:
        words = _analyze_stanza(pipeline, text)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    return words


def to_tokens(words: list[dict]) -> list[Token]:
    return [
        Token(
            text=word["text"],
            start=word["start_char"],
            end=word["end_char"],
            lemma=word["lemma"],
            upos=word["upos"],
            xpos=word["xpos"],
            feats=_parse_feats(word["feats"]),
            head=word["head"],
            deprel=word["deprel"],
            is_interjection=word["upos"] == "INTJ",
        )
        for word in words
    ]


def infer_sentence_type(tokens: list[Token], text: str) -> str:
    """Sentence type from the root verb's mood plus terminal punctuation,
    per objectives.md Phase 3 ("Determine sentence type from syntax and
    punctuation")."""
    root = next((t for t in tokens if t.deprel == "root"), None)
    if root is not None and root.feats.get("Mood") == "Imp":
        return "imperative"

    stripped = text.rstrip()
    if not stripped:
        return "incomplete"
    return _TERMINAL_SENTENCE_TYPE.get(stripped[-1], "incomplete")


def infer_negation(tokens: list[Token]) -> bool:
    return any(token.feats.get("Polarity") == "Neg" for token in tokens)


class LinguisticProcessor:
    name = "linguistic"
    version = "linguistic_v1"  # backend-agnostic; see active_backend()
    provides = {"tokens", "linguistic"}
    requires = {"normalized"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        for sentence in document.sentence_spans:
            text = sentence.normalized_text or sentence.text
            tokens = to_tokens(analyze(text))
            sentence.tokens = tokens
            sentence.sentence_type = infer_sentence_type(tokens, text)
            sentence.is_negated = infer_negation(tokens)
