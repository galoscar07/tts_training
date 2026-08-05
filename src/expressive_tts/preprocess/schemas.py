"""Phase 0 annotation contract for the preprocessing pipeline.

Defines the JSON schema exchanged between the frontend and downstream TTS
models, per readme.md section 1.5/9 and preprocess/objectives.md Phase 0.

Most fields stay ``None``/empty until the processor that owns them is
implemented (linguistic analysis, phonemization, emotion, prosody, ...);
the schema documents the full target shape up front so downstream code can
be written against a stable contract while the pipeline grows.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# preprocess/objectives.md Phase 0 "Initial label set".
EMOTION_LABELS = {"neutral", "happy", "angry", "sad", "fear", "surprise", "unspecified"}
INTENSITY_LABELS = {"low", "medium", "high", "unspecified"}
SENTENCE_TYPES = {"declarative", "interrogative", "exclamative", "imperative", "incomplete"}


class Provenance(str, Enum):
    """Origin of an annotation. See readme.md section 9.2."""

    USER = "user"
    SOURCE = "source"
    RULE = "rule"
    LEXICON = "lexicon"
    PREDICTED = "predicted"
    GENERATED = "generated"
    FALLBACK = "fallback"


class TraceEntry(BaseModel):
    """One recorded transformation applied while producing the output.

    ``start``/``end`` are character offsets into the text as it existed
    immediately before this stage ran (readme.md section 2.1: "Record every
    normalization operation in a trace").
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    operation: str
    original: str
    replacement: str
    start: int
    end: int
    provenance: Provenance = Provenance.RULE
    producer: str
    confidence: Optional[float] = None


class Token(BaseModel):
    """A single token and its linguistic/phonetic annotations.

    Populated in stages by different processors: `lemma`/`upos`/`feats`/
    `head`/`deprel`/`is_interjection` by `linguistic.LinguisticProcessor`;
    `phonemes`/`pronunciation_*` by `phonemizer.PhonemizerProcessor`;
    `syllables`/`stressed_syllable_index`/`stress_*` by
    `stress.StressProcessor`; `focus_*` by `focus.FocusProcessor`. Any
    field stays `None`/empty if its owning processor wasn't requested.

    `focus_score`/`is_focus` (sentence-level emphasis, preprocess/
    objectives.md Phase 8) are distinct from `stressed_syllable_index`
    (word-internal lexical stress, Phase 5) — objectives.md requires the
    two stay separate output fields.
    """

    text: str
    start: int
    end: int

    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    feats: dict[str, str] = Field(default_factory=dict)
    head: Optional[int] = None  # 1-based index of the governing token in the sentence; 0 = root
    deprel: Optional[str] = None
    is_interjection: bool = False

    phonemes: Optional[str] = None
    pronunciation_provenance: Optional[Provenance] = None
    pronunciation_producer: Optional[str] = None
    pronunciation_confidence: Optional[float] = None

    syllables: list[str] = Field(default_factory=list)
    stressed_syllable_index: Optional[int] = None  # 0-based index into `syllables`
    stress_provenance: Optional[Provenance] = None
    stress_producer: Optional[str] = None
    stress_confidence: Optional[float] = None

    focus_score: Optional[float] = None  # continuous, 0-1
    is_focus: bool = False
    focus_provenance: Optional[Provenance] = None
    focus_producer: Optional[str] = None
    focus_rules: list[str] = Field(default_factory=list)  # e.g. ["all_caps", "repetition"]

    pause_before_ms: Optional[int] = None
    pause_after_ms: Optional[int] = None
    # Local deviation from the sentence's ProsodyAnnotation baseline, set
    # only where it differs (focused tokens) — objectives.md Phase 9:
    # "Apply focus locally rather than to the entire sentence."
    local_relative_pitch: Optional[float] = None
    local_relative_energy: Optional[float] = None
    prosody_provenance: Optional[Provenance] = None
    prosody_producer: Optional[str] = None
    prosody_rules: list[str] = Field(default_factory=list)


class EmotionEvidence(BaseModel):
    """One piece of evidence that contributed to an `EmotionAnnotation`.
    See readme.md section 1.5's example provenance JSON."""

    span: str
    rule: str


class EmotionAnnotation(BaseModel):
    """Output of `emotion.EmotionProcessor` (preprocess/objectives.md
    Phase 6). `intensity` nests inside emotion, not a sibling field —
    matches readme.md section 1.5's worked example.
    """

    label: str  # one of EMOTION_LABELS
    confidence: float
    valence: Optional[float] = None
    arousal: Optional[float] = None
    intensity: str  # one of INTENSITY_LABELS
    secondary_label: Optional[str] = None
    distribution: dict[str, float] = Field(default_factory=dict)
    evidence: list[EmotionEvidence] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE
    producer: str


class ProsodyAnnotation(BaseModel):
    """Output of `prosody.ProsodyProcessor` (preprocess/objectives.md
    Phase 9). Matches readme.md section 1.5's worked example shape.
    Safe ranges (objectives.md): speaking_rate 0.80-1.20, relative_pitch
    0.85-1.20, relative_energy 0.80-1.20, pauses 0-1000ms.
    """

    speaking_rate: float
    relative_pitch: float
    relative_energy: float
    terminal_contour: str  # "rising" | "falling" | "continuation"
    pause_after_ms: int
    rules: list[str] = Field(default_factory=list)
    provenance: Provenance = Provenance.RULE
    producer: str


class ContextAdjustment(BaseModel):
    """Output of `context.ContextProcessor` (preprocess/objectives.md
    Phase 11) — the paragraph-context-smoothed emotion prediction.
    `Sentence.emotion` (the raw local prediction) is never overwritten or
    discarded; this is a separate field so both are always available
    ("Store both local and contextual results")."""

    label: str  # one of EMOTION_LABELS
    confidence: float
    valence: Optional[float] = None
    arousal: Optional[float] = None
    intensity: str  # one of INTENSITY_LABELS
    changed: bool  # whether this differs from the local Sentence.emotion prediction
    reason: Optional[str] = None  # e.g. "discourse_marker:contrast:dar", None if unchanged
    provenance: Provenance = Provenance.RULE
    producer: str


class InterjectionSuggestion(BaseModel):
    """One candidate interjection proposed by `interjections.InterjectionProcessor`
    (preprocess/objectives.md Phase 10). Never applied to `Sentence.text` —
    see `Sentence.text_with_interjections`."""

    text: str  # e.g. "Uau"
    position: int  # char offset into the sentence text (0 = sentence start)
    reason: str  # e.g. "emotion_match:happy"
    confidence: float
    matched_emotion: Optional[str] = None


class Sentence(BaseModel):
    """A single sentence and its (currently mostly unpopulated) annotations."""

    text: str
    start: int
    end: int
    sentence_type: Optional[str] = None
    is_negated: Optional[bool] = None
    emotion: Optional[EmotionAnnotation] = None
    context_emotion: Optional[ContextAdjustment] = None
    prosody: Optional[ProsodyAnnotation] = None
    tokens: list[Token] = Field(default_factory=list)

    interjection_suggestions: list[InterjectionSuggestion] = Field(default_factory=list)
    # Only populated in "insert" mode (preprocess/objectives.md Phase 10);
    # `text` above is never modified.
    text_with_interjections: Optional[str] = None


class PreprocessResult(BaseModel):
    """Top-level output of :class:`expressive_tts.preprocess.pipeline.PreprocessPipeline`."""

    schema_version: str = SCHEMA_VERSION
    id: Optional[str] = None

    original_text: str
    clean_text: Optional[str] = None
    normalized_text: Optional[str] = None
    document_style: Optional[str] = None
    phoneme_text: Optional[str] = None
    tts_token_text: Optional[str] = None

    sentences: list[Sentence] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
