"""python -m midiviz score.mid -o outdir

Every token string is returned by MidiTok. This program does not rename them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import svg
from .official import tokenize


def render(path: str, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    schemes = tokenize(path)
    counts = []
    for name, item in schemes.items():
        tokens = item["tokens"]
        counts.append((name, len(tokens)))
        if item["fields"]:
            svg.table(
                name,
                item["fields"],
                [[str(cell) for cell in row] for row in tokens],
                out / f"{name.lower()}.svg",
            )
        else:
            svg.token_strip(
                name,
                [str(tok) for tok in tokens],
                out / f"{name.lower()}.svg",
            )
    svg.lengths(counts, out / "lengths.svg")
    _html(out, Path(path).name, counts)
    print(f"wrote {out}")
    for name, n in counts:
        print(f"  {name}: {n}")


def _html(out: Path, source: str, counts: list[tuple[str, int]]) -> None:
    parts = [
        "<!DOCTYPE html><meta charset='utf-8'><title>MidiTok output</title>",
        "<body style='font-family:Times New Roman,serif;max-width:1100px;margin:24px'>",
        f"<h1>MidiTok tokens of {source}</h1>",
        "<p>Token text is MidiTok's. Config: use_chords, use_tempos, use_programs, and use_time_signatures are true. Octuple also sets max_bar_embedding to 8, which that class requires.</p>",
    ]
    for name in ("midilike", "remi", "tsd", "cpword", "octuple", "lengths"):
        parts.append(f"<h2>{name}</h2><img src='{name}.svg' style='max-width:100%'>")
    parts.append("<ul>" + "".join(f"<li>{n}: {v}</li>" for n, v in counts) + "</ul></body>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MidiTok's own tokens for a MIDI file.")
    parser.add_argument("midi")
    parser.add_argument("-o", "--outdir", default="out")
    args = parser.parse_args()
    render(args.midi, args.outdir)


if __name__ == "__main__":
    main()
