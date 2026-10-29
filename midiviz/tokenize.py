"""Tokenizers aligned with the survey's Class-1 definitions.

MIDI-like follows note-on / note-off / time-shift in milliseconds.
REMI uses 16 positions per 4/4 bar and durations in thirty-second notes,
shown here as note values (a sixteenth is 1/4 of a beat, written Dur·1/4
when the sounding length is one quarter). TSD keeps a time shift and a
duration. Compound word packs a metric step and the notes at that position.
Octuple puts eight fields of one note into one step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mido

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(pitch: int) -> str:
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def _fraction(sixteenths: int) -> str:
    mapping = {1: "1/16", 2: "1/8", 3: "3/16", 4: "1/4", 6: "3/8", 8: "1/2", 12: "3/4", 16: "1"}
    return mapping.get(sixteenths, f"{sixteenths}/16")


@dataclass
class Note:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int
    channel: int
    program: int


@dataclass
class Score:
    notes: list[Note]
    ticks_per_beat: int
    bpm: float
    numerator: int
    denominator: int

    @property
    def ticks_per_16th(self) -> int:
        # Quarter note = one beat when the denominator is 4.
        beat = self.ticks_per_beat * 4 / self.denominator
        return max(1, int(round(beat / 4)))


def load_midi(path: str) -> Score:
    mid = mido.MidiFile(path)
    tempo = 500_000
    num, den = 4, 4
    programs = {ch: 0 for ch in range(16)}
    programs[9] = 128
    pending: dict[tuple[int, int], tuple[int, int, int]] = {}
    notes: list[Note] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "time_signature":
                num, den = msg.numerator, msg.denominator
            elif msg.type == "program_change":
                programs[msg.channel] = 128 if msg.channel == 9 else msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (tick, msg.velocity, programs.get(msg.channel, 0))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key not in pending:
                    continue
                start, vel, program = pending.pop(key)
                notes.append(Note(msg.note, start, tick, vel, msg.channel, program))
    notes.sort(key=lambda n: (n.start_tick, n.pitch, n.channel))
    return Score(notes, mid.ticks_per_beat, float(mido.tempo2bpm(tempo)), num, den)


def _q(score: Score, tick: int) -> int:
    return int(round(tick / score.ticks_per_16th))


def midi_like(score: Score) -> list[str]:
    events: list[tuple[int, int, str]] = []
    for i, note in enumerate(score.notes):
        events.append((note.start_tick, 1, i, f"on:{note.pitch}:{note.velocity}"))
        events.append((note.end_tick, 0, i, f"off:{note.pitch}"))
    events.sort()
    tokens: list[str] = []
    last = 0
    last_vel = None
    for tick, _kind_order, _i, kind in events:
        if tick > last:
            ms = int(round(mido.tick2second(tick - last, score.ticks_per_beat, mido.bpm2tempo(score.bpm)) * 1000))
            if ms > 0:
                tokens.append(f"Shift·{ms}ms")
            last = tick
        if kind.startswith("on:"):
            _, pitch_s, vel_s = kind.split(":")
            vel = int(vel_s)
            if vel != last_vel:
                tokens.append(f"Vel·{vel}")
                last_vel = vel
            tokens.append(f"On·{pitch_name(int(pitch_s))}")
        else:
            tokens.append(f"Off·{pitch_name(int(kind.split(':')[1]))}")
    return tokens


def remi(score: Score) -> list[str]:
    grouped: dict[tuple[int, int], list[Note]] = {}
    for note in score.notes:
        start = _q(score, note.start_tick)
        per_bar = score.numerator * 4 * 4 // score.denominator
        bar = start // per_bar
        pos = start % per_bar
        grouped.setdefault((bar, pos), []).append(note)
    tokens = ["Bar", f"Tempo·{int(round(score.bpm))}"]
    last_bar = 0
    for (bar, pos), group in sorted(grouped.items()):
        while last_bar < bar:
            last_bar += 1
            tokens.append("Bar")
        tokens.append(f"Pos·{pos}")
        for note in sorted(group, key=lambda n: n.pitch):
            dur = max(1, _q(score, note.end_tick) - _q(score, note.start_tick))
            tokens.append(f"Vel·{note.velocity}")
            tokens.append(f"Pitch·{pitch_name(note.pitch)}")
            tokens.append(f"Dur·{_fraction(dur)}")
    return tokens


def tsd(score: Score) -> list[str]:
    grouped: dict[int, list[Note]] = {}
    for note in score.notes:
        grouped.setdefault(_q(score, note.start_tick), []).append(note)
    tokens: list[str] = []
    last = None
    for start in sorted(grouped):
        if last is not None and start > last:
            tokens.append(f"Shift·{_fraction(start - last)}")
        for note in sorted(grouped[start], key=lambda n: n.pitch):
            dur = max(1, _q(score, note.end_tick) - start)
            tokens.extend(
                [
                    f"Pitch·{pitch_name(note.pitch)}",
                    f"Vel·{note.velocity}",
                    f"Dur·{_fraction(dur)}",
                ]
            )
        last = start
    return tokens


def compound(score: Score) -> list[list[str]]:
    """One model step per row: family, bar·position, pitches."""
    grouped: dict[tuple[int, int], list[Note]] = {}
    per_bar = score.numerator * 4 * 4 // score.denominator
    for note in score.notes:
        start = _q(score, note.start_tick)
        grouped.setdefault((start // per_bar, start % per_bar), []).append(note)
    rows = []
    for (bar, pos), group in sorted(grouped.items()):
        names = "+".join(pitch_name(n.pitch) for n in sorted(group, key=lambda n: n.pitch))
        rows.append(["metric", f"{bar + 1}·{pos}", names])
        for note in sorted(group, key=lambda n: n.pitch):
            dur = max(1, _q(score, note.end_tick) - _q(score, note.start_tick))
            rows.append(["note", "—", f"{pitch_name(note.pitch)}  {_fraction(dur)}  vel {note.velocity}"])
    return rows


def octuple(score: Score) -> list[list[str]]:
    per_bar = score.numerator * 4 * 4 // score.denominator
    rows = []
    for note in score.notes:
        start = _q(score, note.start_tick)
        dur = max(1, _q(score, note.end_tick) - start)
        bar, pos = divmod(start, per_bar)
        rows.append(
            [
                f"{score.numerator}/{score.denominator}",
                str(int(round(score.bpm))),
                str(bar + 1),
                str(pos),
                str(note.program),
                pitch_name(note.pitch),
                _fraction(dur),
                str(note.velocity),
            ]
        )
    return rows


def abc_body(score: Score) -> str:
    """One line of ABC. The unit length is an eighth note."""
    grouped: dict[int, list[Note]] = {}
    for note in score.notes:
        grouped.setdefault(_q(score, note.start_tick), []).append(note)
    if not grouped:
        return ""
    per_bar = score.numerator * 4 * 4 // score.denominator
    units = sorted(grouped)
    per_bar = score.numerator * 4 * 4 // score.denominator
    end = ((max(units) // per_bar) + 1) * per_bar
    chunks: list[str] = []
    cursor = 0

    def emit_rest(sixteenths: int) -> None:
        eighths = max(1, sixteenths // 2)
        chunks.append("z" if eighths == 1 else f"z{eighths}")

    for start in units:
        if start > cursor:
            emit_rest(start - cursor)
        if start > 0 and start % per_bar == 0:
            chunks.append("|")
        group = sorted(grouped[start], key=lambda n: n.pitch)
        dur = max(1, min(_q(score, n.end_tick) for n in group) - start)
        mult = max(1, dur // 2)
        if len(group) == 1:
            letter = _pitch_only(group[0])
            chunks.append(letter if mult == 1 else f"{letter}{mult}")
        else:
            body = "".join(_pitch_only(n) for n in group)
            chunks.append(f"[{body}]" if mult == 1 else f"[{body}]{mult}")
        cursor = start + dur
        if cursor % per_bar == 0:
            chunks.append("|")
    if cursor < end:
        emit_rest(end - cursor)
        if end % per_bar == 0:
            chunks.append("|")
    return _abc_join(chunks)


def _pitch_only(note: Note) -> str:
    letters = {0: "C", 1: "^C", 2: "D", 3: "^D", 4: "E", 5: "F", 6: "^F", 7: "G", 8: "^G", 9: "A", 10: "^A", 11: "B"}
    name = letters[note.pitch % 12]
    octave = note.pitch // 12 - 1
    if octave >= 5:
        return name.lower() + "'" * (octave - 5)
    if octave <= 3:
        return name + "," * (4 - octave)
    return name


def _abc_join(chunks: list[str]) -> str:
    text = ""
    for chunk in chunks:
        if chunk == "|":
            text = text.rstrip() + " | "
            continue
        if len(chunk) == 1 and chunk.isalpha() and text and text[-1].isalpha():
            text += chunk
            continue
        if text and not text.endswith(" "):
            text += " "
        text += chunk
    return text.strip()
