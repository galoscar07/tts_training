"""Serialize a finished `PreprocessResult` into TTS-facing formats.

preprocess/objectives.md Phase 12. Pure functions over the already-built
result — unlike every other stage in this package, these don't fit the
`Processor`/`SentenceSpan` contract (readme.md section 2's stage 14,
"serialization", runs *after* the intermediate representation is fully
resolved, not as another annotation layer on it).

`to_control_tokens` prefers `Sentence.context_emotion` (Phase 11, may not
exist on older results) over `Sentence.emotion` when present, since it
represents the paragraph-smoothed final prediction; falls back to the
local prediction otherwise. Both are looked up via `getattr(..., None)`
so this module works whether or not the context layer ran.
"""

from __future__ import annotations

import re

from expressive_tts.preprocess.schemas import PreprocessResult, Sentence

RATE_STEP = 0.05
PITCH_STEP = 0.05
ENERGY_STEP = 0.05

_SENTENCE_TYPE_TOKENS = {
    "declarative": "DECLARATIVE",
    "interrogative": "INTERROGATIVE",
    "exclamative": "EXCLAMATIVE",
    "imperative": "IMPERATIVE",
    "incomplete": "INCOMPLETE",
}


def _quantize(value: float, step: float) -> float:
    """Round to the nearest `step` — objectives.md Phase 12: "continuous
    values should initially be quantized into bins"."""
    return round(round(value / step) * step, 2)


def _emotion_for(sentence: Sentence):
    context = getattr(sentence, "context_emotion", None)
    return context if context is not None else sentence.emotion


def to_canonical_json(result: PreprocessResult, *, indent: int = 2) -> str:
    """The `PreprocessResult` itself, as JSON. Explicit named entry point
    for objectives.md Phase 12's "Canonical JSON" serializer."""
    return result.model_dump_json(indent=indent)


# --- human-readable annotated text -----------------------------------------


def _annotated_text_line(sentence: Sentence) -> str:
    if not sentence.tokens:
        text = sentence.text
    else:
        parts = []
        for token in sentence.tokens:
            surface = f"[FOCUS] {token.text}" if token.is_focus else token.text
            parts.append(surface)
        text = " ".join(parts)

    annotations = []
    emotion = _emotion_for(sentence)
    if emotion is not None:
        annotations.append(f"emotion={emotion.label}, intensity={emotion.intensity}")
    if sentence.prosody is not None and sentence.prosody.pause_after_ms:
        annotations.append(f"pause={sentence.prosody.pause_after_ms}ms")

    suffix = f" [{'; '.join(annotations)}]" if annotations else ""
    return f"{text}{suffix}"


def to_annotated_text(result: PreprocessResult) -> str:
    """Human-readable text with inline emotion/intensity/pause annotations
    and `[FOCUS]` markers. Reconstructed from tokens when available, so
    spacing/punctuation may not be byte-identical to the original text."""
    return "\n".join(_annotated_text_line(sentence) for sentence in result.sentences)


# --- discrete control tokens -------------------------------------------------


def _sentence_type_token(sentence: Sentence) -> str:
    name = _SENTENCE_TYPE_TOKENS.get(sentence.sentence_type or "")
    return f"[SENT_{name}]" if name else "[SENT_UNSPECIFIED]"


def _emotion_tokens(sentence: Sentence) -> list[str]:
    emotion = _emotion_for(sentence)
    if emotion is None:
        return ["[EMO_UNSPECIFIED]", "[INT_UNSPECIFIED]"]
    label = (emotion.label or "unspecified").upper()
    intensity = (emotion.intensity or "unspecified").upper()
    return [f"[EMO_{label}]", f"[INT_{intensity}]"]


def _prosody_tokens(sentence: Sentence) -> list[str]:
    prosody = sentence.prosody
    if prosody is None:
        return ["[RATE_UNSPECIFIED]", "[PITCH_UNSPECIFIED]", "[ENERGY_UNSPECIFIED]"]
    rate = _quantize(prosody.speaking_rate, RATE_STEP)
    pitch = _quantize(prosody.relative_pitch, PITCH_STEP)
    energy = _quantize(prosody.relative_energy, ENERGY_STEP)
    return [f"[RATE_{rate:.2f}]", f"[PITCH_{pitch:.2f}]", f"[ENERGY_{energy:.2f}]"]


def _break_token(sentence: Sentence) -> str | None:
    prosody = sentence.prosody
    if prosody is None:
        return "[BREAK_UNSPECIFIED]"
    if prosody.pause_after_ms:  # a real, computed zero means "no pause" — emit nothing
        return f"[BREAK_{prosody.pause_after_ms}]"
    return None


def _control_text_line(sentence: Sentence) -> str:
    if not sentence.tokens:
        return sentence.text
    parts = []
    for token in sentence.tokens:
        if token.is_focus:
            parts.append("[FOCUS]")
        parts.append(token.phonemes if token.phonemes else token.text)
    return " ".join(parts)


def to_control_tokens(result: PreprocessResult) -> str:
    """Discrete control-token stream matching readme.md section 12's
    worked example. Every value that can't be determined maps to an
    explicit `..._UNSPECIFIED` token rather than being silently
    dropped — objectives.md Phase 12: "unknown values must map to
    explicit UNSPECIFIED tokens", which is also what keeps the
    unknown-control-token rate at 0% by construction, not by tuning.
    """
    lines: list[str] = []
    for sentence in result.sentences:
        lines.append(_sentence_type_token(sentence))
        lines.extend(_emotion_tokens(sentence))
        lines.extend(_prosody_tokens(sentence))
        lines.append(_control_text_line(sentence))
        pause_token = _break_token(sentence)
        if pause_token:
            lines.append(pause_token)
    return "\n".join(lines) + ("\n" if lines else "")


_HEADER_TOKEN_RE = re.compile(r"^\[([A-Z]+)_(.+)\]$")


def parse_control_tokens(serialized: str) -> list[dict]:
    """Parse `to_control_tokens` output back into per-sentence dicts.

    Round-trips to the *quantized* representation `to_control_tokens`
    produced, not to the original unquantized floats — objectives.md
    Phase 12's "reversible to the intermediate representation where
    possible", with that limitation documented rather than glossed over.
    """
    sentences: list[dict] = []
    current: dict | None = None

    def _flush() -> None:
        if current is not None:
            sentences.append(current)

    for raw_line in serialized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADER_TOKEN_RE.match(line)
        if match and match.group(1) == "SENT":
            _flush()
            value = match.group(2)
            current = {"sentence_type": None if value == "UNSPECIFIED" else value.lower()}
            continue
        if current is None:
            continue  # malformed input before any [SENT_*] header — skip
        if match:
            prefix, value = match.group(1), match.group(2)
            is_unspecified = value == "UNSPECIFIED"
            if prefix == "EMO":
                current["emotion_label"] = None if is_unspecified else value.lower()
            elif prefix == "INT":
                current["intensity"] = None if is_unspecified else value.lower()
            elif prefix == "RATE":
                current["speaking_rate"] = None if is_unspecified else float(value)
            elif prefix == "PITCH":
                current["relative_pitch"] = None if is_unspecified else float(value)
            elif prefix == "ENERGY":
                current["relative_energy"] = None if is_unspecified else float(value)
            elif prefix == "BREAK":
                current["pause_after_ms"] = None if is_unspecified else int(value)
        else:
            current["text"] = line
            current["focus_count"] = line.count("[FOCUS]")
    _flush()
    return sentences


# --- minimal, illustrative SSML-like output ----------------------------------


def _ssml_text_line(sentence: Sentence) -> str:
    if not sentence.tokens:
        return sentence.text
    parts = []
    for token in sentence.tokens:
        surface = f"<emphasis>{token.text}</emphasis>" if token.is_focus else token.text
        parts.append(surface)
    return " ".join(parts)


def to_ssml_like(result: PreprocessResult) -> str:
    """A minimal, illustrative SSML-*like* format — not validated against
    the SSML schema, not intended for direct consumption by an SSML
    engine. objectives.md Phase 12 lists this as explicitly optional."""
    lines = ["<speak>"]
    for sentence in result.sentences:
        sentence_type = sentence.sentence_type or "unspecified"
        prosody = sentence.prosody
        prosody_attrs = ""
        if prosody is not None:
            prosody_attrs = f' rate="{prosody.speaking_rate:.2f}" pitch="{prosody.relative_pitch:.2f}"'
        body = _ssml_text_line(sentence)
        break_tag = ""
        if prosody is not None and prosody.pause_after_ms:
            break_tag = f'<break time="{prosody.pause_after_ms}ms"/>'
        lines.append(f'  <s type="{sentence_type}"><prosody{prosody_attrs}>{body}</prosody></s>{break_tag}')
    lines.append("</speak>")
    return "\n".join(lines)
