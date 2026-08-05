"""Processor registry and dependency resolution.

Implements the `Processor` interface and registry responsibilities described
in readme.md section 8: resolve requested layers, determine dependencies,
build an execution order, reject cycles, and skip processors that are not
needed for the requested output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from expressive_tts.preprocess.schemas import (
    ContextAdjustment,
    EmotionAnnotation,
    InterjectionSuggestion,
    ProsodyAnnotation,
    TraceEntry,
    Token,
)


@dataclass
class SentenceSpan:
    """Working representation of a sentence while the pipeline runs.

    Distinct from `expressive_tts.preprocess.schemas.Sentence`, which is the
    serialized output shape. `tokens` reuses the schema's `Token` model
    directly as working state — it's already all-optional-fields, and each
    of linguistic/phonemizer/stress fills in a further slice of it.
    """

    text: str
    start: int
    end: int
    normalized_text: str | None = None
    sentence_type: str | None = None
    is_negated: bool | None = None
    tokens: list[Token] | None = None
    emotion: EmotionAnnotation | None = None
    context_emotion: ContextAdjustment | None = None
    prosody: ProsodyAnnotation | None = None
    interjection_suggestions: list[InterjectionSuggestion] = field(default_factory=list)
    text_with_interjections: str | None = None


@dataclass
class PipelineDocument:
    """Mutable state threaded through processors during a single pipeline run."""

    original_text: str
    id: str | None = None
    clean_text: str | None = None
    normalized_text: str | None = None
    phoneme_text: str | None = None
    document_style: str | None = None
    sentence_spans: list[SentenceSpan] = field(default_factory=list)
    trace: list[TraceEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class Processor(Protocol):
    name: str
    version: str
    provides: set[str]
    requires: set[str]

    def process(self, document: PipelineDocument, config: dict) -> None: ...


class CycleError(RuntimeError):
    """A processor dependency graph contains a cycle."""


class UnknownLayerError(RuntimeError):
    """A requested or required layer has no registered processor."""


class ProcessorRegistry:
    def __init__(self) -> None:
        self._processors: list[Processor] = []
        self._by_layer: dict[str, Processor] = {}

    def register(self, processor: Processor) -> None:
        for layer in processor.provides:
            existing = self._by_layer.get(layer)
            if existing is not None:
                raise ValueError(
                    f"layer {layer!r} is already provided by {existing.name!r}; "
                    f"cannot also register {processor.name!r}"
                )
            self._by_layer[layer] = processor
        self._processors.append(processor)

    def resolve(self, include: set[str]) -> list[Processor]:
        """Return the processors needed to produce `include`, in execution order.

        Processors not reachable from `include` are skipped, matching the
        `{"normalized", "phonemes"}` example in readme.md section 8 where the
        emotion/context/interjection/prosody processors are not executed.
        """
        unknown = {layer for layer in include if layer not in self._by_layer}
        if unknown:
            raise UnknownLayerError(f"no processor provides layer(s): {sorted(unknown)}")

        ordered: list[Processor] = []
        placed: set[str] = set()
        visiting: set[str] = set()

        def visit(layer: str) -> None:
            processor = self._by_layer[layer]
            if processor.name in placed:
                return
            if processor.name in visiting:
                raise CycleError(f"dependency cycle detected at processor {processor.name!r}")
            visiting.add(processor.name)
            for required_layer in sorted(processor.requires):
                if required_layer not in self._by_layer:
                    raise UnknownLayerError(
                        f"processor {processor.name!r} requires unknown layer "
                        f"{required_layer!r}"
                    )
                visit(required_layer)
            visiting.discard(processor.name)
            placed.add(processor.name)
            ordered.append(processor)

        for layer in sorted(include):
            visit(layer)

        return ordered
