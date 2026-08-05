from expressive_tts.preprocess.context import (
    ALPHA,
    BETA,
    HIGH_CONFIDENCE_SKIP,
    ContextProcessor,
    _leading_discourse_category,
    _paragraph_boundaries,
    compute_context_adjustment,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import EmotionAnnotation, Token

MARKERS = {"contrast": ("dar", "însă"), "consequence": ("deci", "așadar")}


def emo(label, confidence, intensity="medium", distribution=None):
    return EmotionAnnotation(
        label=label,
        confidence=confidence,
        intensity=intensity,
        distribution=distribution or {},
        producer="x",
    )


def tok(text, lemma=None, upos="NOUN"):
    return Token(text=text, start=0, end=len(text), lemma=lemma or text, upos=upos)


def span(text="x", emotion=None, tokens=None, start=0, end=1):
    return SentenceSpan(text=text, start=start, end=end, emotion=emotion, tokens=tokens or [])


# --- compute_context_adjustment ---------------------------------------------


def test_returns_none_without_local_emotion():
    assert compute_context_adjustment(span(emotion=None), None) is None


def test_no_previous_emotion_preserves_local_untouched():
    e = emo("happy", 0.5)
    result = compute_context_adjustment(span(emotion=e), None)
    assert result.label == "happy"
    assert result.confidence == 0.5
    assert result.changed is False
    assert result.reason is None


def test_high_local_confidence_skips_blending_entirely():
    current = emo("happy", HIGH_CONFIDENCE_SKIP, distribution={"happy": 1.0})
    previous = emo("sad", 0.9, distribution={"sad": 1.0})
    result = compute_context_adjustment(span(emotion=current), previous)
    assert result.label == "happy"
    assert result.changed is False


def test_low_confidence_current_blends_toward_dominant_previous():
    # a genuinely ambiguous local prediction (near-uniform distribution
    # across categories, no single one dominant) next to a strongly
    # confident previous sentence: alpha=0.7 dominates in general, but a
    # concentrated 1.0-on-one-label previous distribution is still enough
    # to win the argmax against a diffuse local one.
    diffuse = {"happy": 0.17, "sad": 0.17, "angry": 0.17, "fear": 0.17, "surprise": 0.16, "neutral": 0.16}
    current = emo("unspecified", 0.4, distribution=diffuse)
    previous = emo("happy", 0.9, distribution={"happy": 1.0})
    result = compute_context_adjustment(span(emotion=current), previous, alpha=ALPHA, beta_default=BETA)
    assert result.label == "happy"
    assert result.changed is True
    assert "context_blend:unspecified->happy" in result.reason


def test_contrast_marker_suppresses_blending():
    current = emo("angry", 0.4, distribution={"angry": 1.0})
    previous = emo("happy", 0.9, distribution={"happy": 1.0})
    result = compute_context_adjustment(
        span(emotion=current), previous, discourse_category="contrast"
    )
    assert result.label == "angry"  # not pulled toward happy
    assert result.changed is False  # label unchanged; intensity also unchanged here (equal)


def test_consequence_marker_still_blends():
    diffuse = {"happy": 0.17, "sad": 0.17, "angry": 0.17, "fear": 0.17, "surprise": 0.16, "neutral": 0.16}
    current = emo("unspecified", 0.4, distribution=diffuse)
    previous = emo("sad", 0.9, distribution={"sad": 1.0})
    result = compute_context_adjustment(
        span(emotion=current), previous, discourse_category="consequence"
    )
    assert result.label == "sad"
    assert "discourse_marker:consequence" in result.reason


def test_intensity_smoothed_only_when_confidence_low():
    current = emo("happy", 0.3, intensity="high", distribution={"happy": 1.0})
    previous = emo("happy", 0.9, intensity="low", distribution={"happy": 1.0})
    result = compute_context_adjustment(
        span(emotion=current), previous, intensity_smooth_threshold=0.5
    )
    assert result.intensity == "low"
    assert "intensity_smoothed" in result.reason


def test_intensity_not_smoothed_when_confidence_high_enough():
    current = emo("happy", 0.6, intensity="high", distribution={"happy": 1.0})
    previous = emo("happy", 0.9, intensity="low", distribution={"happy": 1.0})
    result = compute_context_adjustment(
        span(emotion=current), previous, high_confidence_skip=0.99, intensity_smooth_threshold=0.5
    )
    assert result.intensity == "high"


def test_no_distribution_falls_back_to_single_label():
    current = emo("angry", 0.3, distribution={})
    previous = emo("happy", 0.9, distribution={})
    result = compute_context_adjustment(span(emotion=current), previous)
    # both distributions default to {label: 1.0} — blend picks the higher-weighted one
    assert result.label in {"angry", "happy"}
    assert result.confidence > 0


# --- discourse markers --------------------------------------------------


def test_leading_discourse_category_detects_contrast():
    s = span(tokens=[tok("Dar"), tok("nu")])
    assert _leading_discourse_category(s, MARKERS) == "contrast"


def test_leading_discourse_category_none_for_unmarked_sentence():
    s = span(tokens=[tok("Astăzi"), tok("plouă")])
    assert _leading_discourse_category(s, MARKERS) is None


def test_leading_discourse_category_ignores_leading_punctuation():
    s = span(tokens=[tok(",", upos="PUNCT"), tok("Deci")])
    assert _leading_discourse_category(s, MARKERS) == "consequence"


def test_leading_discourse_category_empty_tokens():
    assert _leading_discourse_category(span(tokens=[]), MARKERS) is None


# --- paragraph boundaries -------------------------------------------------


def test_paragraph_boundaries_single_paragraph_only_index_zero():
    spans = [span(start=0, end=1), span(start=2, end=3)]
    assert _paragraph_boundaries("A. B.", spans) == {0}


def test_paragraph_boundaries_detects_blank_line_split():
    spans = [span(start=0, end=1), span(start=2, end=3), span(start=4, end=5)]
    text = "Prima propoziție.\n\nA doua. A treia."
    assert _paragraph_boundaries(text, spans) == {0, 1}


def test_paragraph_boundaries_empty_spans():
    assert _paragraph_boundaries("anything", []) == set()


# --- ContextProcessor end-to-end (fake spans, no Stanza) --------------------


def test_processor_preserves_local_and_sets_context_field():
    diffuse = {"happy": 0.17, "sad": 0.17, "angry": 0.17, "fear": 0.17, "surprise": 0.16, "neutral": 0.16}
    doc = PipelineDocument(original_text="A. B.")
    doc.sentence_spans = [
        span(start=0, end=2, emotion=emo("happy", 0.9, distribution={"happy": 1.0})),
        span(start=3, end=5, emotion=emo("unspecified", 0.3, distribution=diffuse)),
    ]
    ContextProcessor().process(doc, {})
    assert doc.sentence_spans[0].emotion.label == "happy"  # untouched
    assert doc.sentence_spans[1].emotion.label == "unspecified"  # local untouched
    assert doc.sentence_spans[1].context_emotion is not None
    assert doc.sentence_spans[1].context_emotion.label == "happy"


def test_processor_chains_off_local_not_smoothed_prediction():
    # sentence 1: confident happy (skips blending).
    # sentence 2: diffuse/ambiguous local distribution -> blends toward
    # sentence 1's happy, so its CONTEXT label becomes "happy" while its
    # LOCAL label stays "unspecified".
    # sentence 3: mildly-sad local distribution, engineered so blending
    # against sentence 2's LOCAL "unspecified" keeps "sad" on top, but
    # blending against sentence 2's SMOOTHED "happy" would flip it to
    # "happy" instead — this is what actually distinguishes the two
    # chaining strategies rather than just restating the design intent.
    diffuse = {"happy": 0.17, "sad": 0.17, "angry": 0.17, "fear": 0.17, "surprise": 0.16, "neutral": 0.16}
    sentence3_dist = {"sad": 0.30, "happy": 0.25, "angry": 0.15, "fear": 0.15, "surprise": 0.15}
    doc = PipelineDocument(original_text="A. B. C.")
    doc.sentence_spans = [
        span(start=0, end=2, emotion=emo("happy", 0.9, distribution={"happy": 1.0})),
        span(start=3, end=5, emotion=emo("unspecified", 0.3, distribution=diffuse)),
        span(start=6, end=8, emotion=emo("sad", 0.4, distribution=sentence3_dist)),
    ]
    ContextProcessor().process(doc, {})
    assert doc.sentence_spans[1].context_emotion.label == "happy"  # sanity: blending did happen for #2
    third = doc.sentence_spans[2].context_emotion
    assert third.label == "sad"
