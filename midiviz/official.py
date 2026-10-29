"""Official MidiTok token strings. Nothing here renames or rewrites tokens."""

from __future__ import annotations

from pathlib import Path

from miditok import CPWord, MIDILike, Octuple, REMI, TSD, TokenizerConfig

# Flags turn on the fields the papers define. The token text itself is MidiTok's.
CONFIG = TokenizerConfig(
    use_chords=True,
    use_tempos=True,
    use_programs=True,
    use_time_signatures=True,
)


def _seq(tokenizer, path: Path):
    out = tokenizer(path)
    return out[0] if isinstance(out, list) else out


def field_names(tokenizer) -> list[str]:
    idx = tokenizer.vocab_types_idx
    names = [""] * (max(idx.values()) + 1)
    for name, i in idx.items():
        if not names[i]:
            names[i] = name
        elif name not in names[i].split("/"):
            names[i] = f"{names[i]}/{name}"
    return names


def tokenize(path: str | Path) -> dict:
    path = Path(path)
    cfg = CONFIG
    oct_cfg = TokenizerConfig(
        use_chords=True,
        use_tempos=True,
        use_programs=True,
        use_time_signatures=True,
        additional_params={"max_bar_embedding": 8},
    )
    tokenizers = {
        "MIDILike": MIDILike(cfg),
        "REMI": REMI(cfg),
        "TSD": TSD(cfg),
        "CPWord": CPWord(cfg),
        "Octuple": Octuple(oct_cfg),
    }
    result = {}
    for name, tok in tokenizers.items():
        seq = _seq(tok, path)
        result[name] = {
            "tokens": seq.tokens,
            "fields": field_names(tok) if seq.tokens and isinstance(seq.tokens[0], list) else None,
        }
    return result
