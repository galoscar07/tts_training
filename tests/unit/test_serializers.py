from expressive_tts.preprocess.serializers import (
    parse_control_tokens,
    to_annotated_text,
    to_canonical_json,
    to_control_tokens,
    to_ssml_like,
)
from expressive_tts.preprocess.schemas import (
    EmotionAnnotation,
    PreprocessResult,
    ProsodyAnnotation,
    Sentence,
    Token,
)


def tok(text, is_focus=False, phonemes=None):
    return Token(text=text, start=0, end=len(text), is_focus=is_focus, phonemes=phonemes)


def emotion(label="happy", intensity="high"):
    return EmotionAnnotation(label=label, confidence=0.9, intensity=intensity, producer="x")


def prosody(rate=1.1, pitch=1.15, energy=1.05, pause=250):
    return ProsodyAnnotation(
        speaking_rate=rate,
        relative_pitch=pitch,
        relative_energy=energy,
        terminal_contour="falling",
        pause_after_ms=pause,
        producer="x",
    )


def sentence(**kwargs):
    defaults = dict(text="Am reușit.", start=0, end=10, sentence_type="declarative")
    defaults.update(kwargs)
    return Sentence(**defaults)


def result(sentences):
    return PreprocessResult(original_text="x", sentences=sentences)


def test_to_canonical_json_round_trips_through_pydantic():
    r = result([sentence(emotion=emotion())])
    text = to_canonical_json(r)
    assert PreprocessResult.model_validate_json(text) == r


def test_control_tokens_unspecified_when_layers_missing():
    r = result([sentence()])  # no emotion, no prosody, no tokens
    output = to_control_tokens(r)
    assert "[EMO_UNSPECIFIED]" in output
    assert "[INT_UNSPECIFIED]" in output
    assert "[RATE_UNSPECIFIED]" in output
    assert "[PITCH_UNSPECIFIED]" in output
    assert "[ENERGY_UNSPECIFIED]" in output
    assert "[BREAK_UNSPECIFIED]" in output


def test_control_tokens_real_values():
    s = sentence(
        sentence_type="exclamative",
        emotion=emotion(label="surprise", intensity="high"),
        prosody=prosody(rate=1.12, pitch=1.13, energy=1.04, pause=250),
        tokens=[tok("Uau", is_focus=True), tok("!")],
    )
    output = to_control_tokens(result([s]))
    assert "[SENT_EXCLAMATIVE]" in output
    assert "[EMO_SURPRISE]" in output
    assert "[INT_HIGH]" in output
    assert "[BREAK_250]" in output
    assert "[FOCUS] Uau !" in output


def test_control_tokens_zero_pause_emits_no_break_token():
    s = sentence(emotion=emotion(), prosody=prosody(pause=0))
    output = to_control_tokens(result([s]))
    assert "BREAK" not in output


def test_quantization_rounds_to_nearest_step():
    # 1.123 / 0.05 = 22.46 -> round to 22 -> 22 * 0.05 = 1.10
    s = sentence(emotion=emotion(), prosody=prosody(rate=1.123, pitch=1.176, energy=0.981))
    output = to_control_tokens(result([s]))
    assert "[RATE_1.10]" in output
    assert "[PITCH_1.20]" in output  # 1.176/0.05=23.52 -> 24 -> 1.20
    assert "[ENERGY_1.00]" in output  # 0.981/0.05=19.62 -> 20 -> 1.00


def test_phonemes_preferred_over_surface_text_when_available():
    s = sentence(emotion=emotion(), tokens=[tok("Uau", phonemes="uˈau")])
    output = to_control_tokens(result([s]))
    lines = output.splitlines()
    text_line = lines[-2]  # last line is [BREAK_...], text is second-to-last
    assert "uˈau" in text_line
    assert "Uau" not in text_line


def test_control_tokens_no_tokens_falls_back_to_sentence_text():
    s = sentence(text="Nu pot să cred!", tokens=[])
    output = to_control_tokens(result([s]))
    assert "Nu pot să cred!" in output


def test_round_trip_parses_back_quantized_values():
    s = sentence(
        sentence_type="interrogative",
        emotion=emotion(label="fear", intensity="low"),
        prosody=prosody(rate=0.9, pitch=0.95, energy=0.85, pause=150),
        tokens=[tok("Chiar", is_focus=True), tok("?")],
    )
    serialized = to_control_tokens(result([s]))
    parsed = parse_control_tokens(serialized)
    assert len(parsed) == 1
    entry = parsed[0]
    assert entry["sentence_type"] == "interrogative"
    assert entry["emotion_label"] == "fear"
    assert entry["intensity"] == "low"
    assert entry["speaking_rate"] == 0.9
    assert entry["relative_pitch"] == 0.95
    assert entry["relative_energy"] == 0.85
    assert entry["pause_after_ms"] == 150
    assert entry["focus_count"] == 1


def test_round_trip_multiple_sentences():
    s1 = sentence(text="A.", sentence_type="declarative", emotion=emotion(label="neutral"))
    s2 = sentence(text="B?", sentence_type="interrogative", emotion=emotion(label="fear"))
    parsed = parse_control_tokens(to_control_tokens(result([s1, s2])))
    assert len(parsed) == 2
    assert parsed[0]["emotion_label"] == "neutral"
    assert parsed[1]["emotion_label"] == "fear"


def test_annotated_text_includes_focus_and_annotations():
    s = sentence(
        emotion=emotion(label="happy", intensity="medium"),
        prosody=prosody(pause=200),
        tokens=[tok("Am"), tok("reușit", is_focus=True), tok(".")],
    )
    output = to_annotated_text(result([s]))
    assert "[FOCUS] reușit" in output
    assert "emotion=happy" in output
    assert "pause=200ms" in output


def test_annotated_text_no_annotations_when_layers_missing():
    output = to_annotated_text(result([sentence()]))
    assert "[" not in output  # no bracket suffix at all


def test_ssml_like_wraps_focus_in_emphasis_and_includes_break():
    s = sentence(
        sentence_type="exclamative",
        prosody=prosody(rate=1.1, pitch=1.1, pause=300),
        tokens=[tok("Uau", is_focus=True), tok("!")],
    )
    output = to_ssml_like(result([s]))
    assert "<speak>" in output and "</speak>" in output
    assert "<emphasis>Uau</emphasis>" in output
    assert '<break time="300ms"/>' in output
    assert 'rate="1.10"' in output


def test_serialization_never_raises_on_empty_result():
    r = result([])
    assert to_control_tokens(r) == ""
    assert to_annotated_text(r) == ""
    assert "<speak>" in to_ssml_like(r)
    assert parse_control_tokens(to_control_tokens(r)) == []


def test_determinism_same_input_same_output():
    s = sentence(emotion=emotion(), prosody=prosody(), tokens=[tok("Am"), tok("reușit", is_focus=True)])
    r = result([s])
    assert to_control_tokens(r) == to_control_tokens(r)
    assert to_annotated_text(r) == to_annotated_text(r)
    assert to_ssml_like(r) == to_ssml_like(r)
