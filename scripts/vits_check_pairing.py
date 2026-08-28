#!/usr/bin/env python
"""Sanity-check a manifest's audio<->text pairing before spending GPU hours.

`catalina_reader` pairs the renamed 266-file subset by **emotion + order**, not
by filename (the originals were removed). If that ordering assumption is wrong
for any emotion, those rows carry the wrong transcript — training on them
degrades everything, and no amount of extra fine-tuning fixes it.

Speech has a fairly tight phonemes-per-second rate, so a mispaired row shows up
as an outlier in `phoneme_count / duration`. This reports the per-speaker rate
distribution and the worst offenders. Stdlib only (wave + statistics), so it
runs anywhere the corpus is mounted.

Usage:
    python scripts/vits_check_pairing.py \
        --manifest out/catalina_emotions.manifest \
        --corpus-root datasets/CATALINA
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import wave
from collections import defaultdict
from pathlib import Path


def wav_duration(path: Path) -> float | None:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--corpus-root", required=True, type=Path)
    p.add_argument("--zmax", type=float, default=3.0, help="flag rows beyond this many MADs from the median rate")
    p.add_argument("--show", type=int, default=15, help="how many outliers to list")
    args = p.parse_args(argv)

    rows = []
    missing = 0
    for raw in args.manifest.read_text(encoding="utf-8").splitlines():
        parts = raw.rstrip("\n").split("|")
        if len(parts) < 3:
            continue
        rel, phonemes, speaker = parts[0], parts[1], parts[2]
        dur = wav_duration(args.corpus_root / rel)
        if dur is None or dur <= 0:
            missing += 1
            continue
        n = len(phonemes.replace(" ", ""))
        rows.append((rel, speaker, n, dur, n / dur))

    if not rows:
        raise SystemExit("no readable rows — check --corpus-root")

    print(f"{len(rows)} rows readable" + (f", {missing} unreadable/missing" if missing else ""))
    print()
    print(f"{'speaker':<22}{'n':>6}{'median rate':>14}{'MAD':>8}{'dur med':>10}")
    by_speaker: dict[str, list] = defaultdict(list)
    for row in rows:
        by_speaker[row[1]].append(row)

    flagged = []
    for speaker, group in sorted(by_speaker.items()):
        rates = [r[4] for r in group]
        med = statistics.median(rates)
        mad = statistics.median([abs(x - med) for x in rates]) or 1e-9
        durs = statistics.median([r[3] for r in group])
        print(f"{speaker:<22}{len(group):>6}{med:>14.2f}{mad:>8.2f}{durs:>10.2f}")
        for rel, spk, n, dur, rate in group:
            z = abs(rate - med) / mad
            if z > args.zmax:
                flagged.append((z, rel, spk, n, dur, rate))

    print()
    print(f"{len(flagged)} row(s) beyond {args.zmax} MADs "
          f"({100 * len(flagged) / len(rows):.1f}% of the corpus)")
    if flagged:
        print("\nworst offenders (likely mispaired audio/text):")
        print(f"{'z':>6}  {'phon':>5} {'dur':>6} {'rate':>6}  file")
        for z, rel, spk, n, dur, rate in sorted(flagged, reverse=True)[: args.show]:
            print(f"{z:>6.1f}  {n:>5} {dur:>6.2f} {rate:>6.1f}  {rel}")
        print("\nA cluster of these inside one emotion means the emotion+order "
              "pairing for its renamed subset is off — fix the reader before retraining.")


if __name__ == "__main__":
    main()
