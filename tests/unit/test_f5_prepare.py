"""Unit test for the F5 dataset adapter. Pure stdlib — no F5, no models."""

import wave

from tts_training.f5.prepare import to_f5_dataset


def _write_wav(path, seconds, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def _make_corpus(root, name, rows):
    """rows: list of (rel_wav, phonemes). Creates the wav files + a manifest."""
    (root / "wavs").mkdir(parents=True, exist_ok=True)
    manifest = root / f"{name}.manifest"
    with manifest.open("w", encoding="utf-8") as m:
        for rel_wav, phon in rows:
            (root / rel_wav).parent.mkdir(parents=True, exist_ok=True)
            (root / rel_wav).write_bytes(b"RIFF")  # stand-in wav
            m.write(f"{rel_wav}|{phon}|spk|\n")
    return manifest


def test_to_f5_dataset_symlinks_and_metadata(tmp_path):
    mara = tmp_path / "MARA"
    swara = tmp_path / "SWARA"
    m1 = _make_corpus(mara, "mara", [("wavs/a.wav", "ˈa mˈaɾa"), ("wavs/b.wav", "bˈun")])
    m2 = _make_corpus(swara, "swara", [("wavs/a.wav", "sˈej")])  # same basename a.wav!

    out = tmp_path / "f5ds"
    stats = to_f5_dataset([(m1, mara), (m2, swara)], out)

    assert stats.written == 3
    assert stats.skipped_missing_wav == 0

    all_lines = (out / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    assert all_lines[0] == "audio_file|text"   # F5-required header
    lines = all_lines[1:]
    assert len(lines) == 3
    # per-manifest prefix keeps the two a.wav files distinct
    names = {line.split("|")[0] for line in lines}
    assert "wavs/mara__a.wav" in names
    assert "wavs/swara__a.wav" in names
    # text column is our IPA phonemes, unchanged
    assert any(line.split("|")[1] == "ˈa mˈaɾa" for line in lines)
    # symlinks exist and resolve to the real files
    for name in names:
        link = out / name
        assert link.is_symlink() and link.resolve().exists()


def test_abs_paths_writes_absolute_audio_paths(tmp_path):
    root = tmp_path / "MARA"
    m = _make_corpus(root, "mara", [("wavs/a.wav", "ˈa")])
    out = tmp_path / "f5ds"
    to_f5_dataset([(m, root)], out, abs_paths=True)

    rows = [l for l in (out / "metadata.csv").read_text(encoding="utf-8").splitlines()[1:]]
    audio = rows[0].split("|")[0]
    assert audio.startswith("/")               # absolute
    assert audio.endswith("wavs/mara__a.wav")


def test_max_duration_drops_long_utterances(tmp_path):
    root = tmp_path / "C"
    _write_wav(root / "wavs" / "short.wav", seconds=1.0)
    _write_wav(root / "wavs" / "long.wav", seconds=10.0)
    manifest = root / "c.manifest"
    manifest.write_text(
        "wavs/short.wav|ˈa|spk|\nwavs/long.wav|bˈar|spk|\n", encoding="utf-8"
    )
    out = tmp_path / "f5ds"
    stats = to_f5_dataset([(manifest, root)], out, max_duration=5.0)

    assert stats.written == 1
    assert stats.skipped_too_long == 1
    rows = (out / "metadata.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert len(rows) == 1 and "short.wav" in rows[0]


def test_skips_missing_wav_and_empty_text(tmp_path):
    root = tmp_path / "C"
    (root / "wavs").mkdir(parents=True)
    (root / "wavs" / "present.wav").write_bytes(b"RIFF")
    manifest = root / "c.manifest"
    manifest.write_text(
        "wavs/present.wav|fˈoo|spk|\n"
        "wavs/gone.wav|bˈar|spk|\n"       # wav missing
        "wavs/present.wav||spk|\n",        # empty phonemes
        encoding="utf-8",
    )
    stats = to_f5_dataset([(manifest, root)], tmp_path / "out")
    assert stats.written == 1
    assert stats.skipped_missing_wav == 1
    assert stats.skipped_empty == 1
