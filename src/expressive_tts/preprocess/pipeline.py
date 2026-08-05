"""Pipeline entrypoint: `PreprocessPipeline.from_config()` / `.process()`.

Wires the registered processors through `registry.ProcessorRegistry` and
resolves which output layers to compute per the precedence order in
readme.md section 1.7 (command-line argument > experiment configuration >
profile configuration > application defaults). Only include/exclude layer
selection is implemented today; other configuration knobs will be added as
later phases introduce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from expressive_tts.preprocess.cleaner import CleanerProcessor
from expressive_tts.preprocess.context import ContextProcessor
from expressive_tts.preprocess.emotion import EmotionProcessor
from expressive_tts.preprocess.focus import FocusProcessor
from expressive_tts.preprocess.interjections import InterjectionProcessor
from expressive_tts.preprocess.linguistic import LinguisticProcessor
from expressive_tts.preprocess.normalizer import (
    NormalizerProcessor,
    default_abbreviations,
    default_currencies,
    default_units,
)
from expressive_tts.preprocess.phonemizer import PhonemizerProcessor, default_overrides
from expressive_tts.preprocess.prosody import ProsodyProcessor
from expressive_tts.preprocess.registry import PipelineDocument, ProcessorRegistry
from expressive_tts.preprocess.sentence_segmenter import SentenceSegmenterProcessor
from expressive_tts.preprocess.serializers import to_control_tokens
from expressive_tts.preprocess.stress import StressProcessor, default_stress_overrides
from expressive_tts.preprocess.schemas import PreprocessResult, Sentence

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"

DEFAULT_INCLUDE = {"normalized"}


def _default_registry() -> ProcessorRegistry:
    registry = ProcessorRegistry()
    registry.register(CleanerProcessor())
    registry.register(SentenceSegmenterProcessor())
    registry.register(NormalizerProcessor())
    registry.register(LinguisticProcessor())
    registry.register(PhonemizerProcessor())
    registry.register(StressProcessor())
    registry.register(EmotionProcessor())
    registry.register(FocusProcessor())
    registry.register(ProsodyProcessor())
    registry.register(InterjectionProcessor())
    registry.register(ContextProcessor())
    return registry


def _load_profile(name: str) -> dict:
    path = _CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class ResolvedConfig:
    include: set[str]
    exclude: set[str] = field(default_factory=set)
    units: dict = field(default_factory=dict)
    currencies: dict = field(default_factory=dict)
    abbreviations: dict = field(default_factory=dict)
    pronunciation_overrides: dict = field(default_factory=dict)
    stress_overrides: dict = field(default_factory=dict)

    def to_processor_config(self) -> dict:
        return {
            "units": self.units,
            "currencies": self.currencies,
            "abbreviations": self.abbreviations,
            "pronunciation_overrides": self.pronunciation_overrides,
            "stress_overrides": self.stress_overrides,
        }


def _resolved_config_from_dict(data: dict) -> ResolvedConfig:
    # Accept both the flat profile shape (readme.md section 1.3, `include: [...]`)
    # and the nested dataset-config shape (readme.md section 1.4, `processing: {include: [...]}`).
    processing = data.get("processing", data)
    include = set(processing.get("include", DEFAULT_INCLUDE))
    exclude = set(processing.get("exclude", []))
    return ResolvedConfig(
        include=include - exclude,
        exclude=exclude,
        units=default_units(),
        currencies=default_currencies(),
        abbreviations=default_abbreviations(),
        pronunciation_overrides=default_overrides(),
        stress_overrides=default_stress_overrides(),
    )


class PreprocessPipeline:
    """Standalone Romanian text preprocessing pipeline. See readme.md section 1.8."""

    def __init__(
        self,
        registry: ProcessorRegistry | None = None,
        config: ResolvedConfig | None = None,
    ) -> None:
        self._registry = registry or _default_registry()
        self._config = config or ResolvedConfig(
            include=set(DEFAULT_INCLUDE),
            units=default_units(),
            currencies=default_currencies(),
            abbreviations=default_abbreviations(),
            pronunciation_overrides=default_overrides(),
            stress_overrides=default_stress_overrides(),
        )

    @classmethod
    def from_config(cls, path: str | Path) -> "PreprocessPipeline":
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(config=_resolved_config_from_dict(data))

    @classmethod
    def from_profile(cls, name: str) -> "PreprocessPipeline":
        return cls(config=_resolved_config_from_dict(_load_profile(name)))

    @property
    def resolved_config(self) -> ResolvedConfig:
        return self._config

    def process(
        self,
        text: str,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        id: str | None = None,
        user_focus_words: set[str] | None = None,
        user_prosody_overrides: dict[int, dict] | None = None,
        interjection_mode: str = "disabled",
        serialize_control_tokens: bool = False,
    ) -> PreprocessResult:
        """Run the pipeline on one text.

        `include`/`exclude` override the pipeline's configured layers for
        this call — the highest-precedence level in readme.md section 1.7.
        `user_focus_words` (surface forms or lemmas, case-insensitive) are
        explicit caller-marked emphasis words that override
        `focus.FocusProcessor`'s rule-based scoring with full priority
        (preprocess/objectives.md Phase 8). `user_prosody_overrides` maps a
        0-based sentence index to any subset of `{speaking_rate,
        relative_pitch, relative_energy, terminal_contour, pause_after_ms}`
        to preserve instead of `prosody.ProsodyProcessor`'s rule-based
        values (Phase 9: "Preserve user-provided values"). `interjection_mode`
        is one of `{"disabled", "suggest", "insert"}` (Phase 10) — defaults
        to `"disabled"` per objectives.md's acceptance criterion that
        suggestions are off unless explicitly requested.
        `serialize_control_tokens` (Phase 12), when true, populates
        `PreprocessResult.tts_token_text` via
        `serializers.to_control_tokens` — off by default, same
        nothing-computed-unless-requested rule as everything else here.
        """
        requested = set(include) if include is not None else set(self._config.include)
        if exclude:
            requested -= set(exclude)
        if not requested:
            raise ValueError("no output layers requested")

        document = PipelineDocument(original_text=text, id=id)
        processors = self._registry.resolve(requested)
        processor_config = self._config.to_processor_config()
        if user_focus_words:
            processor_config["user_focus_words"] = user_focus_words
        if user_prosody_overrides:
            processor_config["user_prosody_overrides"] = user_prosody_overrides
        processor_config["interjection_mode"] = interjection_mode
        for processor in processors:
            processor.process(document, processor_config)

        result = self._to_result(document)
        if serialize_control_tokens:
            result.tts_token_text = to_control_tokens(result)
        return result

    @staticmethod
    def _to_result(document: PipelineDocument) -> PreprocessResult:
        sentences = [
            Sentence(
                text=span.normalized_text if span.normalized_text is not None else span.text,
                start=span.start,
                end=span.end,
                sentence_type=span.sentence_type,
                is_negated=span.is_negated,
                tokens=span.tokens or [],
                emotion=span.emotion,
                context_emotion=span.context_emotion,
                prosody=span.prosody,
                interjection_suggestions=span.interjection_suggestions,
                text_with_interjections=span.text_with_interjections,
            )
            for span in document.sentence_spans
        ]
        return PreprocessResult(
            id=document.id,
            original_text=document.original_text,
            clean_text=document.clean_text,
            normalized_text=document.normalized_text,
            document_style=document.document_style,
            phoneme_text=document.phoneme_text,
            sentences=sentences,
            trace=list(document.trace),
            warnings=list(document.warnings),
            errors=list(document.errors),
        )
