"""Per-corpus readers: each turns a corpus directory into a stream of
`Utterance(rel_wav, text, speaker)`, where `rel_wav` is relative to the corpus
root. Readers are pure w.r.t. the transcript format (they don't phonemize and
don't gate on audio existence — the manifest builder does both), which keeps
them small and unit-testable with tiny fixtures.

Supported corpora:
  * MARA, HRIA   — single-speaker, LJSpeech-style `id|text`.
  * SWARA        — multi-speaker; speaker = top-level subdirectory. Two
                   transcript conventions are auto-detected (see swara_reader).
  * Common Voice — Mozilla `*.tsv` + `clips/*.mp3`; speaker = client_id.

New corpora: add a reader here and register it in `manifest.DATASETS`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

Utterance = None  # replaced below (keeps type-checkers happy about the name)


@dataclass(frozen=True)
class Utterance:  # noqa: F811
    rel_wav: str
    text: str
    speaker: str


DatasetReader = Callable[[Path], Iterator[Utterance]]


# --- shared helpers --------------------------------------------------------


def _resolve_suffix(corpus_root: Path, rel_stem: str, suffixes: tuple[str, ...]) -> str:
    """Return `rel_stem + <suffix>` for the first suffix whose file exists;
    fall back to the first suffix (the builder then records it as missing)."""
    for suffix in suffixes:
        if (corpus_root / f"{rel_stem}{suffix}").exists():
            return f"{rel_stem}{suffix}"
    return f"{rel_stem}{suffixes[0]}"


def _read_pipe_metadata(path: Path, delimiter: str) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            utt_id, _, text = line.partition(delimiter)
            yield utt_id.strip(), text.strip()


# --- LJSpeech-style (MARA, HRIA) -------------------------------------------


def ljspeech_reader(
    metadata_rel: str,
    wav_dir_rel: str,
    speaker: str,
    suffixes: tuple[str, ...] = (".wav", ".WAV"),
    delimiter: str = "|",
) -> DatasetReader:
    def read(corpus_root: Path) -> Iterator[Utterance]:
        metadata_path = corpus_root / metadata_rel
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata not found: {metadata_path}")
        for utt_id, text in _read_pipe_metadata(metadata_path, delimiter):
            if not utt_id:
                continue
            rel_wav = _resolve_suffix(corpus_root, f"{wav_dir_rel}/{utt_id}", suffixes)
            yield Utterance(rel_wav=rel_wav, text=text, speaker=speaker)

    return read


# --- SWARA (multi-speaker) -------------------------------------------------


def _swara_transcript_map(speaker_dir: Path) -> dict[str, str]:
    """Best-effort per-speaker transcript map for the 'central transcript
    file' convention: any `*.txt`/`*.csv` directly in the speaker dir whose
    lines are `id|text`, `id<TAB>text`, or `id text`."""
    mapping: dict[str, str] = {}
    for meta in sorted([*speaker_dir.glob("*.txt"), *speaker_dir.glob("*.csv")]):
        for raw in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            for sep in ("|", "\t"):
                if sep in line:
                    stem, _, text = line.partition(sep)
                    mapping.setdefault(stem.strip(), text.strip())
                    break
            else:
                stem, _, text = line.partition(" ")
                if text:
                    mapping.setdefault(stem.strip(), text.strip())
    return mapping


def swara_reader(corpus_root: Path) -> Iterator[Utterance]:
    """SWARA: each immediate subdirectory of the corpus root is one speaker.

    Two transcript conventions are auto-detected per speaker:
      (a) per-utterance sidecar — `<utt>.wav` next to `<utt>.txt`/`<utt>.lab`;
      (b) central file — a `*.txt`/`*.csv` in the speaker dir mapping
          `id -> text` (`id|text`, `id<TAB>text`, or `id text`).
    Sidecars take precedence. `speaker` is the subdirectory name.

    SWARA isn't on disk here; this follows its documented layout. Adjust the
    conventions above if your copy differs.
    """
    # The distributed SWARA_ALL layout is flat: one pipe-delimited metadata
    # file at the corpus root and every recording under ``wavs/``.
    flat_metadata = corpus_root / "SWARA_ALL.csv"
    if flat_metadata.exists():
        yield from swara_metadata_reader("SWARA_ALL.csv")(corpus_root)
        return

    for speaker_dir in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
        speaker = speaker_dir.name
        central = _swara_transcript_map(speaker_dir)
        for wav in sorted(speaker_dir.rglob("*.wav")):
            stem = wav.stem
            sidecar = next(
                (s for s in (wav.with_suffix(".txt"), wav.with_suffix(".lab")) if s.exists()),
                None,
            )
            if sidecar is not None:
                text = sidecar.read_text(encoding="utf-8", errors="replace").strip()
            else:
                text = central.get(stem, "")
            yield Utterance(
                rel_wav=str(wav.relative_to(corpus_root)), text=text, speaker=speaker
            )


def swara_metadata_reader(metadata_rel: str) -> DatasetReader:
    """Read SWARA's ``filename.wav|transcript`` metadata distribution.

    Files live in ``wavs/`` and the prefix before the first underscore is the
    stable speaker code (for example ``bas_rnd1_001.wav`` -> ``bas``).
    """
    def read(corpus_root: Path) -> Iterator[Utterance]:
        metadata_path = corpus_root / metadata_rel
        if not metadata_path.exists():
            raise FileNotFoundError(f"SWARA metadata not found: {metadata_path}")
        for filename, text in _read_pipe_metadata(metadata_path, "|"):
            if not filename:
                continue
            basename = Path(filename).name
            speaker = Path(basename).stem.partition("_")[0].lower() or "swara"
            yield Utterance(
                rel_wav=f"wavs/{basename}",
                text=text,
                speaker=speaker,
            )

    return read


# --- CATALINA (single speaker, emotional) ----------------------------------

# Romanian emotion word (metadata col 2) -> the English token used in the wav
# filenames (`catalina_<token>_NNNN.wav`). Extend if your emotions differ; the
# manifest builder's "skipped (missing wav)" count flags any unmatched bucket.
CATALINA_EMOTION_MAP = {
    "furios": "angry",
    "trist": "sad",
    "fericit": "happy",
    "bucuros": "happy",
    "neutru": "neutral",
    "neutral": "neutral",
    "speriat": "fear",
    "frica": "fear",
    "surprins": "surprise",
    "surprindere": "surprise",
    "dezgustat": "disgust",
    "dezgust": "disgust",
    "calm": "calm",
}


def catalina_reader(corpus_root: Path) -> Iterator[Utterance]:
    """CATALINA: `metadata.csv` rows are ``path|emotion|text`` (emotion in
    Romanian). The metadata ``path`` column is stale, so each row is paired to
    an actual file in ``wavs/catalina_<english-emotion>_NNNN.wav`` by
    **(emotion bucket, order)**: the i-th metadata row of a given emotion maps
    to the i-th sorted wav of the matching English emotion token
    (`furios`->`angry`, ...). Single speaker ``catalina``.
    """
    import re
    from collections import defaultdict

    meta = corpus_root / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"CATALINA metadata not found: {meta}")

    wavs_by_emotion: dict[str, list[str]] = defaultdict(list)
    for wav in sorted((corpus_root / "wavs").glob("*.wav")):
        m = re.match(r"catalina_([a-z]+)_\d+", wav.stem)
        if m:
            wavs_by_emotion[m.group(1)].append(wav.name)

    index: dict[str, int] = defaultdict(int)
    with meta.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            _stale_path, emotion, text = parts
            token = CATALINA_EMOTION_MAP.get(emotion.strip().lower(), emotion.strip().lower())
            bucket = wavs_by_emotion.get(token, [])
            i = index[token]
            index[token] += 1
            rel_wav = f"wavs/{bucket[i]}" if i < len(bucket) else f"wavs/catalina_{token}_MISSING_{i:04d}.wav"
            yield Utterance(rel_wav=rel_wav, text=text.strip(), speaker="catalina")


# --- Common Voice (Mozilla) ------------------------------------------------


def common_voice_reader(tsv_rel: str = "validated.tsv", clips_rel: str = "clips") -> DatasetReader:
    """Mozilla Common Voice: a `*.tsv` (default `validated.tsv`) with columns
    including `client_id`, `path`, `sentence`; audio in `clips/` as mp3.

    `speaker` is a short, stable id derived from `client_id`. Clips are mp3 at
    48 kHz — Coqui resamples on load, but converting to 22.05 kHz wav first is
    faster; if you convert, point `clips_rel` at the wav folder and adjust the
    extension.
    """

    def read(corpus_root: Path) -> Iterator[Utterance]:
        tsv_path = corpus_root / tsv_rel
        if not tsv_path.exists():
            raise FileNotFoundError(f"Common Voice tsv not found: {tsv_path}")
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sentence = (row.get("sentence") or "").strip()
                clip = (row.get("path") or "").strip()
                if not sentence or not clip:
                    continue
                client = (row.get("client_id") or "unknown").strip()
                speaker = f"cv_{client[:12]}"
                yield Utterance(
                    rel_wav=f"{clips_rel}/{clip}", text=sentence, speaker=speaker
                )

    return read
