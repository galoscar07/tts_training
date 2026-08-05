from expressive_tts.preprocess.sentence_segmenter import segment


def texts(sentences):
    return [s.text for s in sentences]


def test_splits_on_terminal_punctuation():
    result = segment("Nu pot să cred! Am reușit.")
    assert texts(result) == ["Nu pot să cred!", "Am reușit."]


def test_abbreviation_period_does_not_split():
    result = segment("Dr. Popescu a venit. A plecat repede.")
    assert texts(result) == ["Dr. Popescu a venit.", "A plecat repede."]


def test_date_periods_do_not_split():
    result = segment("Ne-am întâlnit pe 12.07.2026. A fost o zi bună.")
    assert texts(result) == ["Ne-am întâlnit pe 12.07.2026.", "A fost o zi bună."]


def test_decimal_number_does_not_split():
    result = segment("Rezultatul este 3.14. Următorul pas urmează.")
    assert texts(result) == ["Rezultatul este 3.14.", "Următorul pas urmează."]


def test_ellipsis_is_one_boundary():
    result = segment("Nu știu... poate mâine.")
    assert texts(result) == ["Nu știu...", "poate mâine."]


def test_empty_text_returns_no_sentences():
    assert segment("") == []


def test_offsets_point_back_into_original_text():
    text = "Bună! Ce faci?"
    result = segment(text)
    for sentence in result:
        assert text[sentence.start : sentence.end] == sentence.text
