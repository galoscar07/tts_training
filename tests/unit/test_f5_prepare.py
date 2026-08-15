"""Unit test for the F5 dataset adapter. Pure stdlib — no F5, no models."""

from tts_training.f5.prepare import to_f5_dataset


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

    lines = (out / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
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
