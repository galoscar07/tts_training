import pytest
from pydantic import ValidationError

from objective_evaluation.schemas import (
    Annotation,
    ContextParagraph,
    EvaluationExample,
    LexicalStressAnnotation,
)
from expressive_tts.preprocess.schemas import Provenance


def _minimal_example(**overrides) -> EvaluationExample:
    fields = dict(
        id="ex-001",
        text="Ce panică!",
        source="REDv2/test#1",
        license="MIT",
        text_register="conversational",
        phenomena=["exclamation"],
        split="dev",
        expected_normalized_text="Ce panică!",
        sentence_boundaries=[(0, 10)],
        emotion=Annotation(value="fear", provenance=Provenance.SOURCE, confidence=1.0),
        intensity=Annotation(value="high", provenance=Provenance.PREDICTED, confidence=0.6),
        sentence_type=Annotation(value="exclamative", provenance=Provenance.RULE),
        focus_words=Annotation(value=["panică"], provenance=Provenance.PREDICTED, confidence=0.6),
        pause_locations=Annotation(value=[], provenance=Provenance.PREDICTED),
        interjection_appropriate=Annotation(value=False, provenance=Provenance.PREDICTED),
    )
    fields.update(overrides)
    return EvaluationExample(**fields)


def test_minimal_example_is_valid():
    example = _minimal_example()
    assert example.split == "dev"
    assert example.emotion.value == "fear"
    assert example.emotion.provenance == Provenance.SOURCE


def test_required_fields_enforced():
    with pytest.raises(ValidationError):
        EvaluationExample(id="x")


def test_lexical_stress_entries():
    example = _minimal_example(
        lexical_stress=[
            LexicalStressAnnotation(
                word="copii",
                syllables=["co", "pii"],
                stressed_syllable_index=1,
                provenance=Provenance.PREDICTED,
                confidence=0.9,
            )
        ]
    )
    assert example.lexical_stress[0].word == "copii"
    assert example.lexical_stress[0].stressed_syllable_index == 1


def test_round_trip_through_json():
    example = _minimal_example()
    round_tripped = EvaluationExample.model_validate_json(example.model_dump_json())
    assert round_tripped == example


def test_context_paragraph():
    paragraph = ContextParagraph(
        id="ctx-001",
        sentences=["Prima propoziție.", "A doua propoziție."],
        source="datasets/mara/metadata.csv",
        license="see datasets/mara",
        text_register="narrative",
    )
    assert paragraph.split == "context"
    assert len(paragraph.sentences) == 2


def test_annotation_defaults():
    annotation = Annotation(value=True, provenance=Provenance.RULE)
    assert annotation.confidence is None
    assert annotation.note is None
