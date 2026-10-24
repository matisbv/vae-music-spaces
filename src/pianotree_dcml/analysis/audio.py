"""Listen to bars, and save them as MIDI or WAV files.

Bars are played one after the other. Audio comes from a small piano-like synthesizer
written with NumPy, so it works anywhere without installing a sound bank; for a real
piano sound, open the MIDI files in any music software.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Sequence

import numpy as np
import pretty_midi

from pianotree_dcml.representation import Note

DEFAULT_TEMPO = 90  # quarter notes per minute
SAMPLE_RATE = 22050
_HARMONICS = [1.0, 0.5, 0.25, 0.12, 0.06]  # relative loudness of the overtones
_ATTACK, _RELEASE = 0.005, 0.08  # seconds


Bars = Sequence[Sequence[Note]]


def to_midi(
    bars: Bars | Sequence[Note],
    path: Path | str | None = None,
    bar_lens: Sequence[int | None] | None = None,
    tempo: float = DEFAULT_TEMPO,
    gap_steps: int = 0,
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """A piano MIDI file of the bars played in sequence, saved to ``path`` if given.

    ``bar_lens`` gives each bar's length in 16th notes (by default, its last note's end,
    rounded up to a beat); ``gap_steps`` adds silence between bars.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    step = _step_seconds(tempo)
    for start, notes in _layout(bars, bar_lens, gap_steps)[0]:
        for onset, pitch, duration in notes:
            piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                start=(start + onset) * step,
                                                end=(start + onset + duration) * step))
    midi.instruments.append(piano)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(path))
    return midi


def synthesize(
    bars: Bars | Sequence[Note],
    bar_lens: Sequence[int | None] | None = None,
    tempo: float = DEFAULT_TEMPO,
    gap_steps: int = 0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mono audio (floats between -1 and 1) of the bars played in sequence."""
    step = _step_seconds(tempo)
    layout, total_steps = _layout(bars, bar_lens, gap_steps)
    audio = np.zeros(int((total_steps * step + _RELEASE) * sample_rate) + 1)
    for start, notes in layout:
        for onset, pitch, duration in notes:
            tone = _piano_tone(pitch, duration * step, sample_rate)
            first = int((start + onset) * step * sample_rate)
            audio[first:first + len(tone)] += tone[:len(audio) - first]
    peak = np.abs(audio).max()
    return (audio * (0.9 / peak) if peak > 0 else audio).astype(np.float32)


def save_wav(audio: np.ndarray, path: Path | str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save mono audio (floats between -1 and 1) as a 16-bit WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    samples = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())


def play(
    bars: Bars | Sequence[Note],
    bar_lens: Sequence[int | None] | None = None,
    tempo: float = DEFAULT_TEMPO,
    gap_steps: int = 0,
):
    """An audio player for notebooks."""
    from IPython.display import Audio

    audio = synthesize(bars, bar_lens, tempo, gap_steps)
    return Audio(audio, rate=SAMPLE_RATE, normalize=False)


def _step_seconds(tempo: float) -> float:
    return 60 / tempo / 4


def _as_bars(bars: Bars | Sequence[Note]) -> list[list[Note]]:
    if len(bars) and isinstance(bars[0], Note):
        return [list(bars)]
    return [list(notes) for notes in bars]


def _bar_length(notes: Sequence[Note], bar_len: int | None) -> int:
    if bar_len is not None:
        return int(bar_len)
    end = max((onset + duration for onset, _, duration in notes), default=4)
    return max(4, math.ceil(end / 4) * 4)


def _layout(bars, bar_lens, gap_steps) -> tuple[list[tuple[int, list[Note]]], int]:
    """Start step of each bar when they are played in sequence, and the total length
    (covering any note that rings past the end of its bar)."""
    bars = _as_bars(bars)
    bar_lens = list(bar_lens) if bar_lens is not None else [None] * len(bars)
    if len(bar_lens) != len(bars):
        raise ValueError(f"{len(bars)} bars but {len(bar_lens)} bar lengths")
    layout, start, total = [], 0, 0
    for notes, bar_len in zip(bars, bar_lens):
        layout.append((start, notes))
        end = start + _bar_length(notes, bar_len)
        total = max([total, end] + [start + onset + duration for onset, _, duration in notes])
        start = end + gap_steps
    return layout, total


def _piano_tone(pitch: int, seconds: float, sample_rate: int) -> np.ndarray:
    """A plucked, piano-like tone: a few harmonics with a fast attack, a decay that is
    quicker for high notes, and a short release after the note ends."""
    frequency = 440.0 * 2 ** ((pitch - 69) / 12)
    t = np.arange(int((seconds + _RELEASE) * sample_rate)) / sample_rate
    tone = sum(amplitude * np.sin(2 * np.pi * k * frequency * t)
               for k, amplitude in enumerate(_HARMONICS, start=1)
               if k * frequency < sample_rate / 2)
    envelope = np.exp(-(1.5 + frequency / 400) * t) * np.minimum(t / _ATTACK, 1.0)
    envelope *= np.clip((seconds + _RELEASE - t) / _RELEASE, 0.0, 1.0)
    return 0.3 * tone * envelope
