import pytest

from expressive_tts.preprocess.normalizer import normalize_text

UNITS = {
    "kg": {"singular": "kilogram", "plural": "kilograme", "gender": "neuter"},
    "h": {"singular": "oră", "plural": "ore", "gender": "feminine"},
    "min": {"singular": "minut", "plural": "minute", "gender": "neuter"},
}
CURRENCIES = {
    "lei": {"singular": "leu", "plural": "lei", "gender": "masculine"},
    "€": {"singular": "euro", "plural": "euro", "gender": "masculine"},
}
ABBREVIATIONS = {"dr": "doctor", "nr": "numărul"}


def normalize(text: str) -> str:
    normalized, _ = normalize_text(
        text, units=UNITS, currencies=CURRENCIES, abbreviations=ABBREVIATIONS
    )
    return normalized


def test_objectives_worked_example():
    # Verbatim example from preprocess/objectives.md Phase 2.
    assert normalize("Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30.") == (
        "Doctor Popescu a trimis douăzeci și cinci de kilograme pe doisprezece "
        "iulie două mii douăzeci și șase, la ora paisprezece și treizeci de minute."
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Am 25 de mere.", "Am douăzeci și cinci de mere."),
        ("Am 3 mere.", "Am trei mere."),
        ("Anul 2026.", "Anul două mii douăzeci și șase."),
    ],
)
def test_plain_numbers(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ne vedem pe 1.01.2026.", "Ne vedem pe întâi ianuarie două mii douăzeci și șase."),
        ("Ne vedem pe 12.07.2026.", "Ne vedem pe doisprezece iulie două mii douăzeci și șase."),
    ],
)
def test_dates(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Vino la ora 14:30.", "Vino la ora paisprezece și treizeci de minute."),
        ("Vino la ora 09:00.", "Vino la ora nouă fix."),
    ],
)
def test_times(text, expected):
    assert normalize(text) == expected


def test_percentages():
    assert normalize("Reducere de 25%.") == "Reducere de douăzeci și cinci la sută."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Costă 25 lei.", "Costă douăzeci și cinci de lei."),
        ("Costă 10€.", "Costă zece euro."),
    ],
)
def test_currencies(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Am mers 5 km.", "Am mers cinci km."),  # "km" not in the test UNITS fixture
        ("Am cumpărat 2 kg.", "Am cumpărat două kilograme."),
        ("A durat 1 h.", "A durat o oră."),
        ("A durat 2 h.", "A durat două ore."),
    ],
)
def test_units(text, expected):
    assert normalize(text) == expected


def test_abbreviation_preserves_capitalization():
    assert normalize("Dr. Ionescu, vezi nr. 5.") == "Doctor Ionescu, vezi numărul cinci."


def test_roman_numeral_ordinal_masculine():
    assert normalize("Ludovic al XIV-lea a domnit mult.") == (
        "Ludovic al paisprezecelea a domnit mult."
    )


def test_roman_numeral_ordinal_feminine():
    assert normalize("Elisabeta a II-a a domnit mult.") == "Elisabeta a doua a domnit mult."


def test_decimal_number():
    assert normalize("Rezultatul este 3,14.") == "Rezultatul este trei virgulă paisprezece."


def test_trace_records_every_normalization():
    _, trace = normalize_text(
        "25 kg", units=UNITS, currencies=CURRENCIES, abbreviations=ABBREVIATIONS
    )
    assert len(trace) == 1
    entry = trace[0]
    assert entry.operation == "unit"
    assert entry.original == "25 kg"
    assert entry.replacement == "douăzeci și cinci de kilograme"
    assert entry.stage == "normalizer"
    assert entry.confidence == 1.0


def test_default_config_dictionaries_load_from_yaml():
    # No explicit dictionaries passed -> falls back to configs/preprocess/*.yaml.
    normalized, _ = normalize_text("Am cumpărat 25 kg.")
    assert normalized == "Am cumpărat douăzeci și cinci de kilograme."
