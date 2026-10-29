# MIDI tokenization visualization

The pictures are MidiTok's own token strings. This program does not rename `Pitch_60` into `C4`, and it does not drop `Ignore_None`.

Config passed to MidiTok: `use_chords=True`, `use_tempos=True`, `use_programs=True`, `use_time_signatures=True`. Octuple also sets `max_bar_embedding=8`, which that class requires. Everything else is MidiTok's default, including `beat_res={(0, 4): 8, (4, 12): 4}` and `num_velocities=32`.

Compound word (CPWord) is one fixed-length list per step. A note step fills pitch, velocity, duration, and program. A metric step fills bar, position, tempo, or time signature. The other positions in that same list are `Ignore_None`. That is why every row has the same number of cells.

## Survey

The survey classifies symbolic music by the object a model accepts: note-level events, a time-aligned two-dimensional table, and coarse learned codes.

- Chinese: [paper/survey_zh.md](paper/survey_zh.md)
- English working translation: [paper/survey_en.md](paper/survey_en.md)

## Run

```bash
pip install -r requirements.txt
python -m midiviz examples/excerpt.mid -o examples/excerpt_out
```

Open `examples/excerpt_out/index.html`.
