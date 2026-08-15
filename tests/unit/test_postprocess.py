"""Unit tests for the audio post-processing chain. Pure numpy/scipy — no
Coqui, no model, no audio files."""

import numpy as np

from tts_training.postprocess import (
    PostProcessConfig,
    high_pass,
    peak_limit,
    postprocess,
    trim_silence,
)

SR = 22050


def _tone_with_silence(dc: float = 0.0) -> np.ndarray:
    t = np.linspace(0, 0.5, SR // 2, endpoint=False)
    tone = (0.9 * np.sin(2 * np.pi * 220 * t)).astype(np.float32) + dc
    return np.concatenate([np.zeros(4000, np.float32), tone, np.zeros(4000, np.float32)])


def test_high_pass_removes_dc_offset():
    wav = _tone_with_silence(dc=0.3)
    out = high_pass(wav, SR, cutoff_hz=70.0)
    assert abs(float(out.mean())) < 0.02  # DC largely gone


def test_high_pass_disabled_when_cutoff_zero():
    wav = _tone_with_silence()
    assert np.array_equal(high_pass(wav, SR, cutoff_hz=0.0), wav)


def test_peak_limit_enforces_ceiling():
    wav = np.array([1.5, -1.5, 0.5], np.float32)  # over full-scale
    out = peak_limit(wav, ceiling_dbfs=-1.0)
    assert np.max(np.abs(out)) <= 10 ** (-1.0 / 20) + 1e-4


def test_trim_silence_shortens():
    wav = _tone_with_silence()
    assert len(trim_silence(wav, SR, top_db=30.0)) < len(wav)


def test_postprocess_full_chain_is_bounded_and_deterministic():
    wav = _tone_with_silence(dc=0.2)
    out1 = postprocess(wav, SR)
    out2 = postprocess(wav, SR)
    assert np.allclose(out1, out2)                        # seeded → reproducible
    assert np.max(np.abs(out1)) <= 10 ** (-1.0 / 20) + 1e-3  # within ceiling
    assert abs(float(out1.mean())) < 0.02                # DC removed
    assert len(out1) < len(wav)                          # silence trimmed
    assert out1.dtype == np.float32


def test_disabling_stages_via_config():
    wav = _tone_with_silence()
    cfg = PostProcessConfig(
        high_pass_hz=0.0, trim_silence=False, reverb_wet=0.0,
        room_tone_dbfs=None, target_lufs=None,
    )
    out = postprocess(wav, SR, cfg)
    # Only peak-limiting remains, so length is unchanged.
    assert len(out) == len(wav)
