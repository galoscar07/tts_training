import pytest

from expressive_tts.preprocess.interjections import (
    InterjectionProcessor,
    detect_document_style,
    suggest_for_sentence,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import EmotionAnnotation, Token

INTERJECTION_EMOTIONS = {
    "uau": {"label": "surprise", "weight": 1.0},
    "bravo": {"label": "happy", "weight": 1.0},
    "super": {"label": "happy", "weight": 0.8},
}
FORMAL_MARKERS = ("astfel", "cu excepția", "condițiile legii")


def tok(text, upos="NOUN", lemma=None, is_interjection=False, feats=None, deprel=None):
    return Token(
        text=text,
        start=0,
        end=len(text),
        lemma=lemma or text.lower(),
        upos=upos,
        is_interjection=is_interjection,
        feats=feats or {},
        deprel=deprel,
    )


def span(tokens, text="x", emotion=None, normalized_text=None):
    return SentenceSpan(
        text=text,
        start=0,
        end=len(text),
        tokens=tokens,
        emotion=emotion,
        normalized_text=normalized_text or text,
    )


def emotion(label="happy", confidence=0.9):
    return EmotionAnnotation(
        label=label, confidence=confidence, intensity="medium", producer="x"
    )


# --- document style detection -----------------------------------------------


def test_formal_marker_hit_is_authoritative():
    s = span(
        [tok("cu"), tok("excepția")],
        text="ceva cu excepția altceva",
        normalized_text="ceva cu excepția altceva",
    )
    style, score, reasons = detect_document_style([s], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "formal"
    assert any("formal_markers" in r for r in reasons)


def test_single_fragile_signal_does_not_tip_to_formal():
    # Regression: "no_first_or_second_person_markers" (fragile — Stanza can
    # mis-tag pro-drop "sunt" as 3rd person) + "no_existing_interjections"
    # (weak — true for most short sentences) must NOT combine to "formal"
    # without a 3rd weak signal or a marker hit.
    s = span(
        [tok("Sunt", upos="AUX", lemma="fi", feats={"Person": "3"}), tok("fericit", upos="ADJ")],
        text="Sunt fericit",
        normalized_text="Sunt fericit",
    )
    style, score, reasons = detect_document_style([s], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "conversational"


def test_all_three_weak_signals_agree_gives_formal():
    long_tokens = [tok(f"w{i}", upos="NOUN") for i in range(20)]
    s = span(long_tokens, text="x", normalized_text="x")
    style, score, reasons = detect_document_style([s], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "formal"
    assert len(reasons) == 3


def test_person_marker_present_leans_conversational():
    s = span(
        [tok("vreau", upos="VERB", feats={"Person": "1"})] + [tok(f"w{i}") for i in range(20)],
        text="x",
        normalized_text="x",
    )
    style, score, reasons = detect_document_style([s], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "conversational"  # only 2/3 weak signals now


def test_existing_interjection_leans_conversational():
    s = span(
        [tok("Uau", lemma="uau")] + [tok(f"w{i}") for i in range(20)],
        text="x",
        normalized_text="x",
    )
    style, score, reasons = detect_document_style([s], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "conversational"


def test_empty_document_is_conversational():
    style, score, reasons = detect_document_style([], FORMAL_MARKERS, INTERJECTION_EMOTIONS)
    assert style == "conversational"
    assert score == 0.0


# --- suggestion gating -------------------------------------------------------


def test_disabled_mode_never_suggests():
    s = span([tok("fericit", upos="ADJ")], emotion=emotion())
    suggest_for_sentence(
        s, mode="disabled", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_formal_style_never_suggests_even_in_suggest_mode():
    s = span([tok("fericit", upos="ADJ")], emotion=emotion())
    suggest_for_sentence(
        s, mode="suggest", document_style="formal", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_existing_interjection_blocks_suggestion():
    s = span([tok("Uau", lemma="uau")], emotion=emotion())
    suggest_for_sentence(
        s, mode="suggest", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_low_emotion_confidence_blocks_suggestion():
    s = span([tok("fericit", upos="ADJ")], emotion=emotion(confidence=0.3))
    suggest_for_sentence(
        s, mode="suggest", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_no_emotion_blocks_suggestion():
    s = span([tok("fericit", upos="ADJ")], emotion=None)
    suggest_for_sentence(
        s, mode="suggest", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_label_with_no_candidates_yields_no_suggestion():
    s = span([tok("x")], emotion=emotion(label="angry"))  # no "angry" entries in the fixture dict
    suggest_for_sentence(
        s, mode="suggest", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.interjection_suggestions == []


def test_suggest_mode_produces_ranked_capped_candidates():
    s = span([tok("x")], emotion=emotion(label="happy", confidence=1.0))
    suggest_for_sentence(
        s,
        mode="suggest",
        document_style="conversational",
        interjection_emotions=INTERJECTION_EMOTIONS,
        max_suggestions=1,
    )
    assert len(s.interjection_suggestions) == 1
    assert s.interjection_suggestions[0].text == "Bravo"  # weight 1.0 > super's 0.8
    assert s.interjection_suggestions[0].matched_emotion == "happy"
    assert s.interjection_suggestions[0].reason == "emotion_match:happy"


def test_suggest_mode_does_not_modify_text():
    s = span([tok("x")], text="Original.", normalized_text="Original.", emotion=emotion())
    suggest_for_sentence(
        s, mode="suggest", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.text == "Original."
    assert s.text_with_interjections is None


def test_insert_mode_produces_enriched_text_preserving_original():
    s = span([tok("x")], text="Sunt fericit.", normalized_text="Sunt fericit.", emotion=emotion())
    suggest_for_sentence(
        s, mode="insert", document_style="conversational", interjection_emotions=INTERJECTION_EMOTIONS
    )
    assert s.text == "Sunt fericit."  # original untouched
    assert s.text_with_interjections == "Bravo, sunt fericit."


# --- processor + document_style plumbing ------------------------------------


def test_processor_sets_document_style_and_respects_disabled_default():
    doc_span = span([tok("x")] + [tok(f"w{i}") for i in range(20)], emotion=emotion())
    document = PipelineDocument(original_text="x", sentence_spans=[doc_span])
    InterjectionProcessor().process(
        document,
        {"interjection_emotions": INTERJECTION_EMOTIONS, "formal_markers": FORMAL_MARKERS},
    )
    assert document.document_style == "formal"
    assert doc_span.interjection_suggestions == []


def test_processor_suggest_mode_end_to_end():
    doc_span = span([tok("x")], emotion=emotion())
    document = PipelineDocument(original_text="x", sentence_spans=[doc_span])
    InterjectionProcessor().process(
        document,
        {
            "interjection_mode": "suggest",
            "interjection_emotions": INTERJECTION_EMOTIONS,
            "formal_markers": FORMAL_MARKERS,
        },
    )
    assert document.document_style == "conversational"
    assert doc_span.interjection_suggestions
