"""python -m midiviz score.mid -o outdir"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import svg
from .tokenize import abc_body, compound, load_midi, midi_like, octuple, remi, tsd


def render(path: str, outdir: str) -> None:
    score = load_midi(path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    schemes = {
        "midi_like": midi_like(score),
        "remi": remi(score),
        "tsd": tsd(score),
    }
    svg.piano_roll(score, out / "pianoroll.svg")
    for name, tokens in schemes.items():
        svg.token_strip(name, tokens, out / f"{name}.svg")
    svg.table(
        "Compound word: one row is one model step",
        ["family", "bar·position", "notes"],
        compound(score),
        out / "compound.svg",
    )
    svg.table(
        "Octuple: one row is one note",
        ["meter", "tempo", "bar", "position", "program", "pitch", "duration", "velocity"],
        octuple(score),
        out / "octuple.svg",
    )
    (out / "abc.txt").write_text(
        f"M:4/4\nL:1/8\nQ:1/4={int(round(score.bpm))}\nK:C\n{abc_body(score)}\n",
        encoding="utf-8",
    )
    counts = [
        ("MIDI-like", len(schemes["midi_like"])),
        ("REMI", len(schemes["remi"])),
        ("TSD", len(schemes["tsd"])),
        ("Compound word", len(compound(score))),
        ("Octuple", len(octuple(score))),
    ]
    svg.lengths(counts, out / "lengths.svg")
    _html(out, Path(path).name, counts)
    print(f"wrote {out}")
    for name, n in counts:
        print(f"  {name}: {n}")


def _html(out: Path, source: str, counts: list[tuple[str, int]]) -> None:
    parts = [
        "<!DOCTYPE html><meta charset='utf-8'><title>MIDI tokenization</title>",
        "<body style='font-family:Times New Roman,serif;max-width:1000px;margin:24px'>",
        f"<h1>Tokenization of {source}</h1>",
        "<p>Class 1 of the survey: the same MIDI file written as MIDI-like, REMI, TSD, compound word, and Octuple. ABC is a lossy notation string, not an event vocabulary.</p>",
        "<img src='pianoroll.svg' style='max-width:100%'>",
    ]
    for name in ("midi_like", "remi", "tsd", "compound", "octuple", "lengths"):
        parts.append(f"<h2>{name}</h2><img src='{name}.svg' style='max-width:100%'>")
    abc = (out / "abc.txt").read_text(encoding="utf-8")
    parts.append("<h2>ABC</h2><pre>" + abc.replace("<", "&lt;") + "</pre>")
    parts.append("<ul>" + "".join(f"<li>{n}: {v}</li>" for n, v in counts) + "</ul></body>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MIDI as several symbolic tokenizations.")
    parser.add_argument("midi")
    parser.add_argument("-o", "--outdir", default="out")
    args = parser.parse_args()
    render(args.midi, args.outdir)


if __name__ == "__main__":
    main()
