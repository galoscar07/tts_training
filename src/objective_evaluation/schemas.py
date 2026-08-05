"""Evaluation-set schema for preprocess/objectives.md Phase 1.

Every subjective judgment call is wrapped in `Annotation`, which always
carries a `provenance` (and usually a `confidence`) so it's never ambiguous
whether a label came from a real human-annotated source dataset
(`Provenance.SOURCE`), a deterministic rule/our own pipeline
(`Provenance.RULE`), or an unreviewed judgment call
(`Provenance.PREDICTED`) — see src/expressive_tts/preprocess/README.md for
why: there is no second human annotator for this project yet, so anything
I draft myself must stay clearly distinguishable from gold data.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from expressive_tts.preprocess.schemas import (
    EMOTION_LABELS,
    INTENSITY_LABELS,
    SENTENCE_TYPES,
    Provenance,
)

REGISTERS = {"conversational", "narrative", "news", "technical", "formal"}
SPLITS = {"dev", "test", "context"}

__all__ = [
    "EMOTION_LABELS",
    "INTENSITY_LABELS",
    "SENTENCE_TYPES",
    "REGISTERS",
    "SPLITS",
    "Annotation",
    "LexicalStressAnnotation",
    "PauseLocation",
    "EvaluationExample",
    "ContextParagraph",
]


class Annotation(BaseModel):
    value: Any
    provenance: Provenance
    confidence: Optional[float] = None
    note: Optional[str] = None


class LexicalStressAnnotation(BaseModel):
    word: str
    syllables: list[str]
    stressed_syllable_index: int  # 0-based
    provenance: Provenance
    confidence: Optional[float] = None


class PauseLocation(BaseModel):
    offset: int  # character offset into `text`
    category: str  # e.g. "comma", "period", "exclamation", "question", "clause_boundary"


class EvaluationExample(BaseModel):
    id: str
    text: str
    source: str
    license: str
    text_register: str
    phenomena: list[str] = Field(default_factory=list)
    split: str

    expected_normalized_text: Optional[str] = None
    sentence_boundaries: list[tuple[int, int]] = Field(default_factory=list)

    emotion: Annotation
    secondary_emotion: Optional[Annotation] = None
    intensity: Annotation
    sentence_type: Annotation
    focus_words: Annotation
    pause_locations: Annotation
    lexical_stress: list[LexicalStressAnnotation] = Field(default_factory=list)
    interjection_appropriate: Annotation
    acceptable_interjections: Optional[Annotation] = None


class ContextParagraph(BaseModel):
    """A multi-sentence paragraph for context-aware evaluation
    (preprocess/objectives.md Phase 1: "at least 30 multi-sentence
    paragraphs")."""

    id: str
    sentences: list[str]
    source: str
    license: str
    text_register: str
    split: str = "context"
