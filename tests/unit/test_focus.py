from expressive_tts.preprocess.focus import (
    FocusProcessor,
    apply_user_focus,
    score_focus,
)
from expressive_tts.preprocess.registry import PipelineDocument, SentenceSpan
from expressive_tts.preprocess.schemas import Provenance, Token

EMOLEX = {"fericit": ["joy"], "trist": ["sadness"], "albastru": ["trust"]}
INTENSIFIERS = {"foarte": 1.6, "extrem de": 1.9}


def tok(text, upos="ADJ", lemma=None, deprel=None, feats=None):
    return Token(
        text=text,
        start=0,
        end=len(text),
        lemma=lemma or text.lower(),
        upos=upos,
        deprel=deprel,
        feats=feats or {},
    )


def score(tokens):
    score_focus(tokens, emolex=EMOLEX, intensifiers=INTENSIFIERS)
    return tokens


def test_all_caps_gets_focus():
    tokens = score([tok("FRUMOS")])
    assert tokens[0].is_focus is True
    assert "all_caps" in tokens[0].focus_rules


def test_lowercase_word_no_signal_gets_no_focus():
    tokens = score([tok("masă", upos="NOUN")])
    assert tokens[0].is_focus is False
    assert tokens[0].focus_score == 0.0


def test_intensifier_target_gets_focus():
    tokens = score([tok("foarte", upos="ADV"), tok("frumos")])
    assert tokens[1].is_focus is True
    assert "intensifier_target" in tokens[1].focus_rules
    # the intensifier word itself isn't the target
    assert "intensifier_target" not in tokens[0].focus_rules


def test_repetition_gets_focus_boost():
    tokens = score([tok("bine", upos="ADV"), tok("bine", upos="ADV")])
    assert "repetition" in tokens[0].focus_rules
    assert "repetition" in tokens[1].focus_rules


def test_repetition_ignored_for_function_words():
    tokens = score([tok("și", upos="CCONJ"), tok("și", upos="CCONJ")])
    assert tokens[0].focus_score == 0.0
    assert tokens[1].focus_score == 0.0


def test_emotion_bearing_word_gets_partial_score():
    tokens = score([tok("fericit")])
    assert "emotion_bearing" in tokens[0].focus_rules
    assert 0 < tokens[0].focus_score < 1.0


def test_function_word_guard_blocks_weak_evidence():
    # An AUX with only a weak signal (repetition alone, e.g.) must not surface.
    tokens = score([tok("este", upos="AUX"), tok("este", upos="AUX")])
    assert tokens[0].focus_score == 0.0
    assert tokens[0].focus_rules == []


def test_contrastive_construction_suppresses_and_boosts():
    tokens = score(
        [
            tok("Nu", upos="PART", lemma="nu", feats={"Polarity": "Neg"}),
            tok("e", upos="AUX"),
            tok("roșu", upos="ADJ", deprel="root"),
            tok(",", upos="PUNCT"),
            tok("ci", upos="CCONJ"),
            tok("albastru", upos="ADJ"),
        ]
    )
    assert "contrastive_negation_suppressed" in tokens[2].focus_rules
    assert tokens[2].focus_score == 0.0
    assert "corrective_construction" in tokens[5].focus_rules
    assert tokens[5].is_focus is True


def test_main_predicate_gets_small_boost():
    tokens = score([tok("aleargă", upos="VERB", deprel="root")])
    assert "main_predicate" in tokens[0].focus_rules


def test_main_predicate_not_awarded_to_aux_root():
    tokens = score([tok("este", upos="AUX", deprel="root")])
    assert "main_predicate" not in tokens[0].focus_rules


def test_punctuation_never_scored():
    tokens = score([tok(".", upos="PUNCT")])
    assert tokens[0].focus_score is None
    assert tokens[0].is_focus is False


def test_apply_user_focus_overrides_everything():
    tokens = [tok("masă", upos="NOUN")]
    score_focus(tokens, emolex=EMOLEX, intensifiers=INTENSIFIERS)
    assert tokens[0].is_focus is False
    apply_user_focus(tokens, {"masă"})
    assert tokens[0].is_focus is True
    assert tokens[0].focus_score == 1.0
    assert tokens[0].focus_provenance == Provenance.USER
    assert tokens[0].focus_rules == ["user_provided"]


def test_processor_end_to_end_with_injected_config():
    span = SentenceSpan(text="FRUMOS", start=0, end=6, tokens=[tok("FRUMOS")])
    document = PipelineDocument(original_text="FRUMOS", sentence_spans=[span])
    FocusProcessor().process(
        document, {"intensifiers": INTENSIFIERS, "emolex": EMOLEX}
    )
    token = document.sentence_spans[0].tokens[0]
    assert token.is_focus is True
    assert token.focus_producer == "focus_rules_v1"


def test_processor_honors_user_focus_words_from_config():
    span = SentenceSpan(text="masă", start=0, end=4, tokens=[tok("masă", upos="NOUN")])
    document = PipelineDocument(original_text="masă", sentence_spans=[span])
    FocusProcessor().process(
        document,
        {"intensifiers": INTENSIFIERS, "emolex": EMOLEX, "user_focus_words": {"masă"}},
    )
    token = document.sentence_spans[0].tokens[0]
    assert token.focus_provenance == Provenance.USER
