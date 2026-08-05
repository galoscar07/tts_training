"""Unit tests for the corpus readers. Pure — no pipeline, no Coqui, no
models — using tiny on-disk fixtures."""

from tts_training.data.readers import (
    common_voice_reader,
    ljspeech_reader,
    swara_reader,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# --- LJSpeech --------------------------------------------------------------


def test_ljspeech_reader_pairs_text_and_wav(tmp_path):
    (tmp_path / "metadata.csv").write_text("u1|Salut lume\nu2|A doua replică\n", encoding="utf-8")
    _touch(tmp_path / "wavs" / "u1.wav")
    _touch(tmp_path / "wavs" / "u2.wav")

    utts = list(ljspeech_reader("metadata.csv", "wavs", "spk")(tmp_path))
    assert [u.rel_wav for u in utts] == ["wavs/u1.wav", "wavs/u2.wav"]
    assert [u.text for u in utts] == ["Salut lume", "A doua replică"]
    assert {u.speaker for u in utts} == {"spk"}


def test_ljspeech_reader_resolves_alternate_suffix(tmp_path):
    # HRIA-style: only the alternate (.WAV) suffix exists. On a case-sensitive
    # filesystem this proves the fallback; on a case-insensitive one (macOS)
    # it resolves via the first suffix — either way a real file is found.
    (tmp_path / "metadata.csv").write_text("u1|text\n", encoding="utf-8")
    _touch(tmp_path / "d" / "u1.WAV")
    utt = next(iter(ljspeech_reader("metadata.csv", "d", "spk", suffixes=(".WAV", ".wav"))(tmp_path)))
    assert (tmp_path / utt.rel_wav).exists()


def test_ljspeech_reader_missing_wav_uses_primary_suffix(tmp_path):
    (tmp_path / "metadata.csv").write_text("u9|fără audio\n", encoding="utf-8")
    utts = list(ljspeech_reader("metadata.csv", "wavs", "spk")(tmp_path))
    # No file exists → falls back to the first suffix; builder counts it missing.
    assert utts[0].rel_wav == "wavs/u9.wav"


# --- SWARA -----------------------------------------------------------------


def test_swara_reader_sidecar_convention(tmp_path):
    spk = tmp_path / "FDG"
    _touch(spk / "fdg_0001.wav")
    (spk / "fdg_0001.txt").write_text("Text de probă", encoding="utf-8")

    utts = list(swara_reader(tmp_path))
    assert len(utts) == 1
    assert utts[0].speaker == "FDG"
    assert utts[0].rel_wav == "FDG/fdg_0001.wav"
    assert utts[0].text == "Text de probă"


def test_swara_reader_central_transcript_convention(tmp_path):
    spk = tmp_path / "SAM"
    _touch(spk / "sam_0001.wav")
    _touch(spk / "sam_0002.wav")
    (spk / "transcripts.txt").write_text(
        "sam_0001|Prima\nsam_0002|A doua\n", encoding="utf-8"
    )
    utts = {u.rel_wav: u.text for u in swara_reader(tmp_path)}
    assert utts["SAM/sam_0001.wav"] == "Prima"
    assert utts["SAM/sam_0002.wav"] == "A doua"


# --- Common Voice ----------------------------------------------------------


def test_common_voice_reader_reads_tsv(tmp_path):
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\n"
        "abcdef0123456789\tclip_a.mp3\tBună ziua\n"
        "abcdef0123456789\tclip_b.mp3\t\n"          # empty sentence -> skipped
        "999888777666\tclip_c.mp3\tA treia frază\n",
        encoding="utf-8",
    )
    utts = list(common_voice_reader()(tmp_path))
    assert [u.rel_wav for u in utts] == ["clips/clip_a.mp3", "clips/clip_c.mp3"]
    assert utts[0].text == "Bună ziua"
    assert utts[0].speaker == "cv_abcdef012345"   # short, stable id from client_id
