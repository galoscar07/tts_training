"""Romanian cardinal/ordinal number-to-words conversion.

Covers the grammatical rules exercised by preprocess/objectives.md Phase 2:

- gendered forms of "one" and "two" (`unu/una`, `doi/două`) — digits 3-9 and
  the teens are gender-invariant;
- the "de"-insertion rule for countable nouns: singular for 1, a bare plural
  for 2-19, and a plural preceded by "de" for values >= 20
  (e.g. "douăzeci și cinci de kilograme");
- the scale words "sută"/"sute", "mie"/"mii", "milion"/"milioane",
  "miliard"/"miliarde", each of which is always counted with the
  *feminine* numeral form (they are feminine or neuter-plural-as-feminine
  nouns), independent of the gender requested for the final remainder.

Ordinal support is limited to 1-99 (masculine "al X-lea" / feminine "a X-a"
forms); larger ordinals raise `ValueError` since they fall outside what this
project currently needs (chapter/ruler numbering, `al XIV-lea` style spans).
"""

from __future__ import annotations

Gender = str  # "masculine" | "feminine"

_UNITS_MASC = [
    "zero", "unu", "doi", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă",
]
_UNITS_FEM = [
    "zero", "una", "două", "trei", "patru", "cinci", "șase", "șapte", "opt", "nouă",
]
_TEENS_MASC = [
    "zece", "unsprezece", "doisprezece", "treisprezece", "paisprezece",
    "cincisprezece", "șaisprezece", "șaptesprezece", "optsprezece", "nouăsprezece",
]
_TEENS_FEM = list(_TEENS_MASC)
_TEENS_FEM[2] = "douăsprezece"
_TENS = [
    None, None, "douăzeci", "treizeci", "patruzeci", "cincizeci",
    "șaizeci", "șaptezeci", "optzeci", "nouăzeci",
]

_ONE_ARTICLE = {"masculine": "un", "feminine": "o", "neuter": "un"}

# (scale value, singular noun, plural noun, indefinite-article word for count == 1)
_SCALES = [
    (10**9, "miliard", "miliarde", "un"),
    (10**6, "milion", "milioane", "un"),
    (10**3, "mie", "mii", "o"),
]


def _units(gender: Gender) -> list[str]:
    return _UNITS_FEM if gender == "feminine" else _UNITS_MASC


def _teens(gender: Gender) -> list[str]:
    return _TEENS_FEM if gender == "feminine" else _TEENS_MASC


def _read_tens_units(n: int, gender: Gender) -> str:
    """Render 1-99."""
    if n < 10:
        return _units(gender)[n]
    if n < 20:
        return _teens(gender)[n - 10]
    tens_word = _TENS[n // 10]
    unit = n % 10
    if unit == 0:
        return tens_word
    return f"{tens_word} și {_units(gender)[unit]}"


def _read_0_999(n: int, gender: Gender) -> str:
    if n == 0:
        return ""
    hundreds, rem = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        if hundreds == 1:
            parts.append("o sută")
        else:
            parts.append(f"{_read_tens_units(hundreds, 'feminine')} sute")
    if rem:
        parts.append(_read_tens_units(rem, gender))
    return " ".join(parts)


def cardinal(n: int, gender: Gender = "masculine") -> str:
    """Convert an integer to Romanian words.

    `gender` controls the trailing 1/2 digit of the final (rightmost, < 1000)
    group only; higher-order groups (thousand/million/billion) always use
    the feminine form to agree with their (feminine or neuter) scale noun.
    """
    if n == 0:
        return "zero"
    if n < 0:
        return f"minus {cardinal(-n, gender)}"

    parts: list[str] = []
    remainder = n
    for scale_value, singular, plural, one_word in _SCALES:
        count, remainder = divmod(remainder, scale_value)
        if not count:
            continue
        if count == 1:
            parts.append(f"{one_word} {singular}")
        else:
            group = _read_0_999(count, "feminine")
            connector = "de " if count >= 20 else ""
            parts.append(f"{group} {connector}{plural}")
    if remainder:
        parts.append(_read_0_999(remainder, gender))

    return " ".join(parts)


def count_phrase(n: int, singular: str, plural: str, gender: Gender = "masculine") -> str:
    """Render `n` followed by the correctly inflected noun, per the
    Romanian "de"-insertion rule: 1 -> singular with the indefinite article,
    2-19 -> bare plural, >= 20 -> plural preceded by "de".

    `gender` is the noun's grammatical gender: "masculine" (un leu, doi lei),
    "feminine" (o oră, două ore), or "neuter" (un kilogram, două kilograme —
    Romanian neuter nouns take the masculine article in the singular but
    feminine numeral agreement in the plural).
    """
    if n == 0:
        return f"zero {plural}"
    if n == 1:
        return f"{_ONE_ARTICLE[gender]} {singular}"
    plural_gender: Gender = "feminine" if gender == "neuter" else gender
    connector = "de " if n >= 20 else ""
    return f"{cardinal(n, plural_gender)} {connector}{plural}"


_ORDINAL_MASC: dict[int, str] = {
    1: "primul", 2: "al doilea", 3: "al treilea", 4: "al patrulea",
    5: "al cincilea", 6: "al șaselea", 7: "al șaptelea", 8: "al optulea",
    9: "al nouălea", 10: "al zecelea", 11: "al unsprezecelea",
    12: "al doisprezecelea", 13: "al treisprezecelea", 14: "al paisprezecelea",
    15: "al cincisprezecelea", 16: "al șaisprezecelea", 17: "al șaptesprezecelea",
    18: "al optsprezecelea", 19: "al nouăsprezecelea", 20: "al douăzecilea",
}
_ORDINAL_FEM: dict[int, str] = {
    1: "prima", 2: "a doua", 3: "a treia", 4: "a patra", 5: "a cincea",
    6: "a șasea", 7: "a șaptea", 8: "a opta", 9: "a noua", 10: "a zecea",
    11: "a unsprezecea", 12: "a douăsprezecea", 13: "a treisprezecea",
    14: "a paisprezecea", 15: "a cincisprezecea", 16: "a șaisprezecea",
    17: "a șaptesprezecea", 18: "a optsprezecea", 19: "a nouăsprezecea",
    20: "a douăzecea",
}
_TENS_ORDINAL_MASC = {
    20: "douăzecilea", 30: "treizecilea", 40: "patruzecilea", 50: "cincizecilea",
    60: "șaizecilea", 70: "șaptezecilea", 80: "optzecilea", 90: "nouăzecilea",
}
_TENS_ORDINAL_FEM = {
    20: "douăzecea", 30: "treizecea", 40: "patruzecea", 50: "cincizecea",
    60: "șaizecea", 70: "șaptezecea", 80: "optzecea", 90: "nouăzecea",
}

# Suffix used for the units digit (1-9) inside a compound ordinal like
# "douăzeci și unulea" (21st) — distinct from the irregular standalone
# forms in _ORDINAL_MASC/_ORDINAL_FEM ("primul"/"prima" for 1st).
_ORDINAL_UNIT_SUFFIX_MASC = {
    1: "unulea", 2: "doilea", 3: "treilea", 4: "patrulea", 5: "cincilea",
    6: "șaselea", 7: "șaptelea", 8: "optulea", 9: "nouălea",
}
_ORDINAL_UNIT_SUFFIX_FEM = {
    1: "una", 2: "doua", 3: "treia", 4: "patra", 5: "cincea",
    6: "șasea", 7: "șaptea", 8: "opta", 9: "noua",
}


def _ordinal(
    n: int,
    table: dict[int, str],
    tens_table: dict[int, str],
    unit_suffix_table: dict[int, str],
    prefix: str,
) -> str:
    if n in table:
        return table[n]
    if 21 <= n < 100:
        tens, unit = divmod(n, 10)
        tens_value = tens * 10
        if unit == 0:
            return f"{prefix} {tens_table[tens_value]}"
        return f"{prefix} {_TENS[tens]} și {unit_suffix_table[unit]}"
    raise ValueError(f"ordinal conversion not supported for n={n}")


def ordinal_masculine(n: int) -> str:
    return _ordinal(n, _ORDINAL_MASC, _TENS_ORDINAL_MASC, _ORDINAL_UNIT_SUFFIX_MASC, "al")


def ordinal_feminine(n: int) -> str:
    return _ordinal(n, _ORDINAL_FEM, _TENS_ORDINAL_FEM, _ORDINAL_UNIT_SUFFIX_FEM, "a")


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(numeral: str) -> int:
    """Convert an upper-case Roman numeral to an integer. Does not validate
    that the numeral is in canonical form (e.g. accepts "IIII")."""
    total = 0
    previous = 0
    for char in reversed(numeral.upper()):
        if char not in _ROMAN_VALUES:
            raise ValueError(f"not a Roman numeral character: {char!r}")
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total
