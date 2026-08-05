import pytest

from expressive_tts.preprocess.registry import (
    CycleError,
    PipelineDocument,
    ProcessorRegistry,
    UnknownLayerError,
)


class FakeProcessor:
    def __init__(self, name, provides, requires):
        self.name = name
        self.version = "v1"
        self.provides = provides
        self.requires = requires
        self.calls = []

    def process(self, document, config):
        self.calls.append(document)


def test_resolve_returns_processors_in_dependency_order():
    registry = ProcessorRegistry()
    cleaner = FakeProcessor("cleaner", {"clean"}, set())
    tokenizer = FakeProcessor("tokenizer", {"tokens"}, {"clean"})
    phonemizer = FakeProcessor("phonemizer", {"phonemes"}, {"tokens"})
    registry.register(cleaner)
    registry.register(tokenizer)
    registry.register(phonemizer)

    ordered = registry.resolve({"phonemes"})

    assert ordered == [cleaner, tokenizer, phonemizer]


def test_resolve_skips_unrequested_processors():
    # readme.md section 8: requesting {"normalized", "phonemes"} does not
    # execute the emotion/context/interjection/prosody processors.
    registry = ProcessorRegistry()
    cleaner = FakeProcessor("cleaner", {"clean"}, set())
    normalizer = FakeProcessor("normalizer", {"normalized"}, {"clean"})
    emotion = FakeProcessor("emotion", {"emotion"}, {"normalized"})

    registry.register(cleaner)
    registry.register(normalizer)
    registry.register(emotion)

    ordered = registry.resolve({"normalized"})

    assert emotion not in ordered
    assert ordered == [cleaner, normalizer]


def test_resolve_deduplicates_shared_dependencies():
    registry = ProcessorRegistry()
    cleaner = FakeProcessor("cleaner", {"clean"}, set())
    a = FakeProcessor("a", {"a"}, {"clean"})
    b = FakeProcessor("b", {"b"}, {"clean"})
    registry.register(cleaner)
    registry.register(a)
    registry.register(b)

    ordered = registry.resolve({"a", "b"})

    assert ordered.count(cleaner) == 1
    assert ordered.index(cleaner) < ordered.index(a)
    assert ordered.index(cleaner) < ordered.index(b)


def test_resolve_rejects_unknown_layer():
    registry = ProcessorRegistry()
    with pytest.raises(UnknownLayerError):
        registry.resolve({"nonexistent"})


def test_resolve_detects_cycle():
    registry = ProcessorRegistry()
    a = FakeProcessor("a", {"a"}, {"b"})
    b = FakeProcessor("b", {"b"}, {"a"})
    registry.register(a)
    registry.register(b)

    with pytest.raises(CycleError):
        registry.resolve({"a"})


def test_register_rejects_duplicate_layer_provider():
    registry = ProcessorRegistry()
    registry.register(FakeProcessor("a", {"x"}, set()))
    with pytest.raises(ValueError):
        registry.register(FakeProcessor("b", {"x"}, set()))


def test_pipeline_document_defaults():
    document = PipelineDocument(original_text="hello")
    assert document.clean_text is None
    assert document.sentence_spans == []
    assert document.trace == []
