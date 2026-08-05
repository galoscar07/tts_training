"""Romanian text normalization: dates, times, percentages, currencies,
measurement units, cardinal numbers, abbreviations, and Roman-numeral
ordinals. See readme.md section 2.1 and preprocess/objectives.md Phase 2.

The worked example from objectives.md Phase 2:

    "Dr. Popescu a trimis 25 kg pe 12.07.2026, la ora 14:30."
    -> "Doctor Popescu a trimis douăzeci și cinci de kilograme pe
        doisprezece iulie două mii douăzeci și șase, la ora paisprezece
        și treizeci de minute."
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from expressive_tts.preprocess.numbers_ro import cardinal, count_phrase, ordinal_feminine, ordinal_masculine, roman_to_int
from expressive_tts.preprocess.registry import PipelineDocument
from expressive_tts.preprocess.trace_utils import apply_substitution
from expressive_tts.preprocess.schemas import TraceEntry

PRODUCER = "normalizer_v1"

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "preprocess"

_MONTHS = [
    "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie",
    "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie",
]


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    path = _CONFIG_DIR / name
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_units() -> dict:
    return _load_yaml("units.yaml")


def default_currencies() -> dict:
    return _load_yaml("currencies.yaml")


def default_abbreviations() -> dict:
    return _load_yaml("abbreviations.yaml")


# --- Roman-numeral ordinals: "al XIV-lea" / "a XIV-a" -----------------------

_ROMAN_MASC_PATTERN = re.compile(r"\bal\s+([IVXLCDM]+)-lea\b")
_ROMAN_FEM_PATTERN = re.compile(r"\ba\s+([IVXLCDM]+)-a\b")


def _roman_masc_replacement(match: re.Match) -> str:
    try:
        return ordinal_masculine(roman_to_int(match.group(1)))
    except ValueError:
        return match.group(0)


def _roman_fem_replacement(match: re.Match) -> str:
    try:
        return ordinal_feminine(roman_to_int(match.group(1)))
    except ValueError:
        return match.group(0)


# --- Dates: DD.MM.YYYY --------------------------------------------------

_DATE_PATTERN = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")


def _date_replacement(match: re.Match) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return match.group(0)
    day_word = "întâi" if day == 1 else cardinal(day, "masculine")
    month_word = _MONTHS[month - 1]
    year_word = cardinal(year, "masculine")
    return f"{day_word} {month_word} {year_word}"


# --- Times: HH:MM ---------------------------------------------------------

_TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _time_replacement(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return match.group(0)
    hour_word = cardinal(hour, "masculine")
    if minute == 0:
        return f"{hour_word} fix"
    minute_word = count_phrase(minute, "minut", "minute", "neuter")
    return f"{hour_word} și {minute_word}"


# --- Percentages ------------------------------------------------------------

_PERCENT_PATTERN = re.compile(r"\b(\d+)\s*%")


def _percent_replacement(match: re.Match) -> str:
    return f"{cardinal(int(match.group(1)), 'masculine')} la sută"


# --- Currencies and measurement units (dictionary-driven) -------------------


def _token_piece(token: str) -> str:
    escaped = re.escape(token)
    return rf"\b{escaped}\b" if token.isalpha() else escaped


def _build_token_pattern(tokens: dict) -> re.Pattern:
    pieces = sorted((_token_piece(token) for token in tokens), key=len, reverse=True)
    return re.compile(r"(\d+)\s*(" + "|".join(pieces) + r")", re.IGNORECASE)


def _make_count_replacement(entries: dict):
    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        entry = entries.get(match.group(2)) or entries.get(match.group(2).lower())
        if entry is None:
            return match.group(0)
        return count_phrase(n, entry["singular"], entry["plural"], entry.get("gender", "masculine"))

    return replace


# --- Decimal and plain cardinal numbers --------------------------------------

_DECIMAL_PATTERN = re.compile(r"\b(\d+)[.,](\d+)\b")


def _decimal_replacement(match: re.Match) -> str:
    integer_word = cardinal(int(match.group(1)), "masculine")
    fractional_word = cardinal(int(match.group(2)), "masculine")
    return f"{integer_word} virgulă {fractional_word}"


_INTEGER_PATTERN = re.compile(r"\b\d+\b")


def _integer_replacement(match: re.Match) -> str:
    return cardinal(int(match.group(0)), "masculine")


# --- Abbreviations (dictionary-driven) ---------------------------------------


def _build_abbreviation_pattern(abbreviations: dict) -> re.Pattern:
    pieces = sorted((re.escape(key) for key in abbreviations), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(pieces) + r")\.", re.IGNORECASE)


def _make_abbreviation_replacement(abbreviations: dict):
    def replace(match: re.Match) -> str:
        key = match.group(1).lower()
        expansion = abbreviations.get(key)
        if expansion is None:
            return match.group(0)
        if match.group(1)[0].isupper():
            expansion = expansion[0].upper() + expansion[1:]
        return expansion

    return replace


def normalize_text(
    text: str,
    *,
    units: dict | None = None,
    currencies: dict | None = None,
    abbreviations: dict | None = None,
    stage: str = "normalizer",
    producer: str = PRODUCER,
) -> tuple[str, list[TraceEntry]]:
    """Apply the full normalization cascade to one sentence of text.

    Pass explicit `units`/`currencies`/`abbreviations` dicts to override the
    project defaults loaded from configs/preprocess/*.yaml (useful for unit
    tests that want isolated, file-independent fixtures).
    """
    units = units if units is not None else default_units()
    currencies = currencies if currencies is not None else default_currencies()
    abbreviations = abbreviations if abbreviations is not None else default_abbreviations()

    trace: list[TraceEntry] = []

    def step(current_text: str, pattern: re.Pattern, replace_fn, operation: str) -> str:
        new_text, entries = apply_substitution(
            current_text,
            pattern,
            replace_fn,
            stage=stage,
            operation=operation,
            producer=producer,
        )
        trace.extend(entries)
        return new_text

    text = step(text, _ROMAN_MASC_PATTERN, _roman_masc_replacement, "roman_ordinal")
    text = step(text, _ROMAN_FEM_PATTERN, _roman_fem_replacement, "roman_ordinal")
    text = step(text, _DATE_PATTERN, _date_replacement, "date")
    text = step(text, _TIME_PATTERN, _time_replacement, "time")
    text = step(text, _PERCENT_PATTERN, _percent_replacement, "percentage")
    if currencies:
        text = step(text, _build_token_pattern(currencies), _make_count_replacement(currencies), "currency")
    if units:
        text = step(text, _build_token_pattern(units), _make_count_replacement(units), "unit")
    text = step(text, _DECIMAL_PATTERN, _decimal_replacement, "decimal_number")
    text = step(text, _INTEGER_PATTERN, _integer_replacement, "cardinal_number")
    if abbreviations:
        text = step(
            text,
            _build_abbreviation_pattern(abbreviations),
            _make_abbreviation_replacement(abbreviations),
            "abbreviation",
        )

    return text, trace


class NormalizerProcessor:
    name = "normalizer"
    version = PRODUCER
    provides = {"normalized"}
    requires = {"clean", "sentences"}

    def process(self, document: PipelineDocument, config: dict) -> None:
        units = config.get("units", default_units())
        currencies = config.get("currencies", default_currencies())
        abbreviations = config.get("abbreviations", default_abbreviations())

        normalized_sentences = []
        for sentence in document.sentence_spans:
            normalized, entries = normalize_text(
                sentence.text,
                units=units,
                currencies=currencies,
                abbreviations=abbreviations,
            )
            sentence.normalized_text = normalized
            document.trace.extend(entries)
            normalized_sentences.append(normalized)

        document.normalized_text = " ".join(normalized_sentences)
