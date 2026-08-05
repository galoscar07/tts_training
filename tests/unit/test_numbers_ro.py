import pytest

from expressive_tts.preprocess.numbers_ro import (
    cardinal,
    count_phrase,
    ordinal_feminine,
    ordinal_masculine,
    roman_to_int,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "zero"),
        (1, "unu"),
        (2, "doi"),
        (9, "nouă"),
        (10, "zece"),
        (11, "unsprezece"),
        (12, "doisprezece"),
        (19, "nouăsprezece"),
        (20, "douăzeci"),
        (21, "douăzeci și unu"),
        (25, "douăzeci și cinci"),
        (99, "nouăzeci și nouă"),
        (100, "o sută"),
        (101, "o sută unu"),
        (200, "două sute"),
        (256, "două sute cincizeci și șase"),
        (1000, "o mie"),
        (2026, "două mii douăzeci și șase"),
        (2000, "două mii"),
        (1_000_000, "un milion"),
        (2_000_000, "două milioane"),
        (1_000_000_000, "un miliard"),
    ],
)
def test_cardinal_masculine(n, expected):
    assert cardinal(n, "masculine") == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "una"),
        (2, "două"),
        (12, "douăsprezece"),
        (21, "douăzeci și una"),
        (22, "douăzeci și două"),
    ],
)
def test_cardinal_feminine(n, expected):
    assert cardinal(n, "feminine") == expected


def test_cardinal_negative():
    assert cardinal(-5, "masculine") == "minus cinci"


def test_cardinal_scale_multiplier_is_always_feminine():
    # "douăzeci și una de mii", not "...unu de mii" — "mie/mii" is feminine.
    assert cardinal(21_000, "masculine") == "douăzeci și una de mii"
    # "douăzeci și două de milioane" — milion/milioane is neuter, plural
    # agreement is feminine.
    assert cardinal(22_000_000, "masculine") == "douăzeci și două de milioane"


@pytest.mark.parametrize(
    "n, singular, plural, gender, expected",
    [
        (0, "kilogram", "kilograme", "neuter", "zero kilograme"),
        # Neuter: masculine article in the singular ("un kilogram"), feminine
        # numeral agreement in the plural ("două kilograme").
        (1, "kilogram", "kilograme", "neuter", "un kilogram"),
        (2, "kilogram", "kilograme", "neuter", "două kilograme"),
        (1, "leu", "lei", "masculine", "un leu"),
        (1, "oră", "ore", "feminine", "o oră"),
        (2, "oră", "ore", "feminine", "două ore"),
        (10, "euro", "euro", "masculine", "zece euro"),
        (19, "kilogram", "kilograme", "neuter", "nouăsprezece kilograme"),
        (25, "kilogram", "kilograme", "neuter", "douăzeci și cinci de kilograme"),
        (25, "leu", "lei", "masculine", "douăzeci și cinci de lei"),
    ],
)
def test_count_phrase(n, singular, plural, gender, expected):
    assert count_phrase(n, singular, plural, gender) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "primul"),
        (2, "al doilea"),
        (10, "al zecelea"),
        (14, "al paisprezecelea"),
        (20, "al douăzecilea"),
        (21, "al douăzeci și unulea"),
        (23, "al douăzeci și treilea"),
    ],
)
def test_ordinal_masculine(n, expected):
    assert ordinal_masculine(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "prima"),
        (2, "a doua"),
        (14, "a paisprezecea"),
        (20, "a douăzecea"),
        (23, "a douăzeci și treia"),
    ],
)
def test_ordinal_feminine(n, expected):
    assert ordinal_feminine(n) == expected


def test_ordinal_out_of_supported_range_raises():
    with pytest.raises(ValueError):
        ordinal_masculine(150)


@pytest.mark.parametrize(
    "numeral, expected",
    [
        ("I", 1),
        ("IV", 4),
        ("IX", 9),
        ("XIV", 14),
        ("XX", 20),
        ("XL", 40),
        ("MCMXCIX", 1999),
    ],
)
def test_roman_to_int(numeral, expected):
    assert roman_to_int(numeral) == expected


def test_roman_to_int_invalid_character():
    with pytest.raises(ValueError):
        roman_to_int("ABC")
