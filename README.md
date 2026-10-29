# MIDI tokenization visualization

Companion code and paper for a survey of symbolic-music tokenization. The survey classifies a representation by the object a model actually accepts:

1. **Note-level event sequences.** One step is a note or one attribute of a note (MIDI-like, REMI, TSD, compound word, Octuple, byte-pair encoding, ABC). After expansion, the symbols still name pitch, duration, or position.
2. **A time-aligned two-dimensional table.** A piano roll (time by pitch) or a track-by-time table such as GETScore. Cells may be continuous values or discrete pitch and duration symbols.
3. **Coarse learned codes.** Codebook indices with no pitch name. A decoder reconstructs the music, and reconstruction is lossy.

The same six-note excerpt is drawn under each writing in the paper. This repository reads a MIDI file and draws the Class-1 writings: piano roll, MIDI-like, REMI, TSD, compound word, Octuple, and a lossy ABC line.

The survey figures are not produced by printing MidiTok tokens. They are drawn from these event strings. `python -m midiviz` is the program that turns an input MIDI file into those pictures.

## Paper

- Chinese: [paper/survey_zh.md](paper/survey_zh.md)
- English working translation: [paper/survey_en.md](paper/survey_en.md)
- Class-1 figure, including MIDI-like, REMI, compound word, and Octuple: [paper/figures/fig3_class1.svg](paper/figures/fig3_class1.svg)

## Use

```bash
pip install -r requirements.txt
python -m midiviz examples/excerpt.mid -o examples/excerpt_out
```

Open `examples/excerpt_out/index.html`. The example file is the survey excerpt: two bars, 4/4, 120 BPM, C4+E4, G4, A4, B4, C5.

On that file the lengths match the survey figure: MIDI-like 20, REMI 26, TSD 22, compound word 11, Octuple 6.

## What is quantized

- MIDI-like keeps velocity and millisecond shifts. Simultaneous note-ons have no shift between them. Note-offs at a tick are written before note-ons at the same tick.
- REMI uses 16 positions per bar in 4/4. Duration is the sounding length on that grid, written as a note value. A sixteenth is two thirty-second notes, so a quarter note is `Dur·1/4`.
- TSD uses the same pitch, velocity, and duration, with a time shift instead of bar and position.
- Compound word writes one metric step, then one note step for each note at that position.
- Octuple writes one row per note: meter, tempo, bar, position, program, pitch, duration, velocity.
- ABC uses an eighth-note unit. It does not keep velocity. Chord texture is not in the string.
