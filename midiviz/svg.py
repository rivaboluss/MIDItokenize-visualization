"""Small SVG writer for token strips, tables, and a piano roll."""

from __future__ import annotations

from pathlib import Path

from .tokenize import Note, Score, pitch_name

INK = "#1a1a1a"
MUTED = "#4a4a4a"
HAIR = "#c8c8c8"
PAPER = "#ffffff"
NOTE = "#1f4e79"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.parts: list[str] = []

    def rect(self, x, y, w, h, fill, stroke="none", sw=1):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, text, size=11, fill=INK, anchor="start", weight=400):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family="Times New Roman, Times, Microsoft YaHei, serif">'
            f"{_esc(text)}</text>"
        )

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1):
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def save(self, path: Path) -> None:
        body = "\n".join(self.parts)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">\n<rect width="100%" height="100%" fill="{PAPER}"/>\n'
            f"{body}\n</svg>\n",
            encoding="utf-8",
        )


def token_strip(title: str, tokens: list[str], path: Path) -> None:
    y = 36
    rows = 1
    x = 8
    placed = []
    for token in tokens:
        w = 6.4 * len(token) + 12
        if x + w > 980:
            x = 8
            y += 24
            rows += 1
        placed.append((x, y, w, token))
        x += w + 4
    svg = Svg(1000, y + 28)
    svg.text(8, 20, title, 14, weight=600)
    for x, y, w, token in placed:
        svg.rect(x, y, w, 20, "#f7f7f7", HAIR, 0.8)
        svg.text(x + 6, y + 14, token, 11)
    svg.save(path)


def table(title: str, headers: list[str], rows: list[list[str]], path: Path) -> None:
    col_w = max(90, min(140, 1100 // max(1, len(headers))))
    svg = Svg(max(640, 16 + col_w * len(headers)), 48 + 22 * (len(rows) + 1))
    svg.text(8, 20, title, 14, weight=600)
    y = 32
    for i, header in enumerate(headers):
        svg.text(8 + i * col_w, y + 14, header, 11, MUTED, weight=600)
    y += 20
    for row in rows:
        for i, cell in enumerate(row):
            svg.text(8 + i * col_w, y + 14, cell, 11)
        y += 20
    svg.h = y + 8
    svg.save(path)


def lengths(pairs: list[tuple[str, int]], path: Path) -> None:
    svg = Svg(720, 40 + 28 * len(pairs))
    svg.text(8, 20, "Sequence length", 14, weight=600)
    max_v = max(v for _, v in pairs) or 1
    for i, (name, value) in enumerate(pairs):
        y = 36 + i * 26
        svg.text(8, y + 14, name, 12, MUTED)
        svg.rect(140, y, int(480 * value / max_v), 16, NOTE)
        svg.text(148 + int(480 * value / max_v), y + 13, str(value), 11)
    svg.save(path)


def piano_roll(score: Score, path: Path) -> None:
    if not score.notes:
        Svg(400, 40).save(path)
        return
    pitches = list(range(max(n.pitch for n in score.notes), min(n.pitch for n in score.notes) - 1, -1))
    cell = score.ticks_per_16th
    cols = max(_end(n, cell) for n in score.notes) or 1
    col_w = 16 if cols < 48 else 10
    row_h = 12
    label_w = 36
    svg = Svg(label_w + cols * col_w + 16, 28 + len(pitches) * row_h + 24)
    svg.text(8, 16, "Piano roll, sixteenth-note columns", 12, MUTED)
    top = 24
    for i, pitch in enumerate(pitches):
        svg.text(2, top + i * row_h + 10, pitch_name(pitch), 9, MUTED)
    for note in score.notes:
        row = pitches.index(note.pitch)
        start = note.start_tick / cell
        dur = max(note.end_tick - note.start_tick, cell) / cell
        svg.rect(label_w + start * col_w, top + row * row_h + 1, max(2, dur * col_w - 1), row_h - 2, NOTE)
    svg.save(path)


def _end(note: Note, cell: int) -> int:
    return int(round(note.end_tick / cell))
