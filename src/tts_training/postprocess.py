"""Audio post-processing for synthesized speech — the ``--postprocess`` step.

VITS output tends to sound a little sterile: dead-silent gaps, a synthetic
noise-free floor, no room. This chain applies light, well-known mastering /
realism filters to make it sound more like a real recording, in order:

    DC/rumble removal (high-pass) → trim silence → light room reverb →
    faint room-tone noise floor → loudness normalization → peak limiting

Everything is toggled from ``PostProcessConfig``; ``--postprocess`` on the
synthesizer applies the default "realism" preset. Pure ``numpy``/``scipy``
(+ optional ``librosa``/``pyloudnorm`` if present) — no Coqui needed, so it
can also be run standalone over existing wavs.

Design notes:
 - Reverb and room-tone use a *seeded* RNG so a given input yields identical
   output (reproducible synthesis).
 - Each stage is a small pure function; the orchestrator just sequences them,
   so you can import and reuse any single filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


@dataclass
class PostProcessConfig:
    """The realism preset. Set a field to its disabling value to skip a stage."""

    high_pass_hz: float = 70.0        # 0 disables — removes DC offset / sub-bass rumble
    trim_silence: bool = True         # trim dead air at the ends
    trim_top_db: float = 30.0
    reverb_wet: float = 0.10          # 0 disables — small-room convolution reverb
    reverb_decay_s: float = 0.25
    room_tone_dbfs: float | None = -55.0  # None disables — faint recorded-room noise floor
    target_lufs: float = -23.0        # None disables loudness normalization (EBU R128)
    peak_ceiling_dbfs: float = -1.0   # final true-peak ceiling
    seed: int = 1234                  # makes reverb / room-tone deterministic


# --- individual filters ----------------------------------------------------


def high_pass(wav: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        return wav
    sos = butter(order, cutoff_hz / (sr / 2.0), btype="highpass", output="sos")
    return sosfilt(sos, wav).astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float) -> np.ndarray:
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        return trimmed if trimmed.size else wav
    except Exception:
        # Energy-threshold fallback if librosa isn't importable.
        eps = 10 ** (-top_db / 20) * (np.max(np.abs(wav)) + 1e-9)
        loud = np.where(np.abs(wav) > eps)[0]
        return wav[loud[0]: loud[-1] + 1] if loud.size else wav


def light_reverb(wav: np.ndarray, sr: int, wet: float, decay_s: float, rng: np.random.Generator) -> np.ndarray:
    """Convolution reverb with a synthetic exponentially-decaying-noise impulse
    response — cheap way to add a small room without an IR file."""
    if wet <= 0:
        return wav
    n = max(1, int(sr * decay_s))
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0.0, 6.0, n))
    ir /= np.abs(ir).sum() + 1e-9
    wet_sig = fftconvolve(wav, ir, mode="full")[: len(wav)]
    return ((1.0 - wet) * wav + wet * wet_sig).astype(np.float32)


def add_room_tone(wav: np.ndarray, level_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """A faint noise floor — real recordings are never digitally silent, and a
    tiny amount of room tone reads as 'recorded' rather than 'synthetic'."""
    level = 10 ** (level_dbfs / 20.0)
    return (wav + rng.standard_normal(len(wav)).astype(np.float32) * level).astype(np.float32)


def loudness_normalize(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isfinite(loudness):
            return pyln.normalize.loudness(wav, loudness, target_lufs).astype(np.float32)
    except Exception:
        pass
    # RMS fallback: map current RMS to the target treated as dBFS.
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-9
    return (wav * (10 ** (target_lufs / 20.0) / rms)).astype(np.float32)


def peak_limit(wav: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    ceiling = 10 ** (ceiling_dbfs / 20.0)
    peak = np.max(np.abs(wav)) + 1e-9
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    return np.clip(wav, -ceiling, ceiling).astype(np.float32)


# --- orchestrator ----------------------------------------------------------


def postprocess(wav: np.ndarray, sr: int, config: PostProcessConfig | None = None) -> np.ndarray:
    """Apply the realism chain and return a float32 wav in [-1, 1]."""
    config = config or PostProcessConfig()
    rng = np.random.default_rng(config.seed)

    wav = np.asarray(wav, dtype=np.float32)
    wav = high_pass(wav, sr, config.high_pass_hz)
    if config.trim_silence:
        wav = trim_silence(wav, sr, config.trim_top_db)
    wav = light_reverb(wav, sr, config.reverb_wet, config.reverb_decay_s, rng)
    if config.room_tone_dbfs is not None:
        wav = add_room_tone(wav, config.room_tone_dbfs, rng)
    if config.target_lufs is not None:
        wav = loudness_normalize(wav, sr, config.target_lufs)
    wav = peak_limit(wav, config.peak_ceiling_dbfs)
    return wav
