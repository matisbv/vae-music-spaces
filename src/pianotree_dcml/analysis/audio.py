"""Listen to bars, and save them as MIDI, MP3 or WAV files.

Bars are played one after the other. Audio comes from a small piano synthesizer written
with NumPy, so it works anywhere without installing a sound bank. For a recorded piano
sound, render the MIDI files with a SoundFont in any music software.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Sequence

import lameenc
import numpy as np
import pretty_midi

from pianotree_dcml.representation import Note

DEFAULT_TEMPO = 90  # quarter notes per minute
SAMPLE_RATE = 32000
_RELEASE = 0.15  # seconds for a note to fade out after it ends (the damper)
_REVERB_SECONDS, _REVERB_MIX = 1.2, 0.2


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
    for start, duration, pitch in _events(bars, bar_lens, tempo, gap_steps)[0]:
        piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch,
                                            start=start, end=start + duration))
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
    events, seconds = _events(bars, bar_lens, tempo, gap_steps)
    return _render(events, seconds, sample_rate)


def synthesize_midi(midi: pretty_midi.PrettyMIDI | Path | str,
                    sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono audio of a MIDI file (e.g. one written by ``to_midi``), with the same piano."""
    if not isinstance(midi, pretty_midi.PrettyMIDI):
        midi = pretty_midi.PrettyMIDI(str(midi))
    events = [(n.start, n.end - n.start, n.pitch)
              for instrument in midi.instruments for n in instrument.notes]
    return _render(events, midi.get_end_time(), sample_rate)


def save_mp3(audio: np.ndarray, path: Path | str, sample_rate: int = SAMPLE_RATE,
             bitrate: int = 128) -> None:
    """Save mono audio (floats between -1 and 1) as an MP3 file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality, 7 = fast
    data = encoder.encode(_to_int16(audio).tobytes()) + encoder.flush()
    Path(path).write_bytes(data)


def save_wav(audio: np.ndarray, path: Path | str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save mono audio (floats between -1 and 1) as a 16-bit WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(_to_int16(audio).tobytes())


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


def _to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2")


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


def _events(bars, bar_lens, tempo, gap_steps) -> tuple[list[tuple[float, float, int]], float]:
    """The notes of the bars played in sequence, as (start, duration, pitch) in seconds,
    and the total length (covering any note that rings past the end of its bar)."""
    bars = _as_bars(bars)
    bar_lens = list(bar_lens) if bar_lens is not None else [None] * len(bars)
    if len(bar_lens) != len(bars):
        raise ValueError(f"{len(bars)} bars but {len(bar_lens)} bar lengths")
    step = _step_seconds(tempo)
    events, start, total = [], 0, 0
    for notes, bar_len in zip(bars, bar_lens):
        events += [((start + onset) * step, duration * step, pitch)
                   for onset, pitch, duration in notes]
        end = start + _bar_length(notes, bar_len)
        total = max([total, end] + [start + onset + duration for onset, _, duration in notes])
        start = end + gap_steps
    return events, total * step


def _render(events: Sequence[tuple[float, float, int]], seconds: float,
            sample_rate: int) -> np.ndarray:
    """Play (start, duration, pitch) events with the piano, add a little reverb and
    normalise the result."""
    audio = np.zeros(int((seconds + _RELEASE) * sample_rate) + 1)
    for start, duration, pitch in events:
        tone = _piano_tone(pitch, duration, sample_rate)
        first = int(start * sample_rate)
        audio[first:first + len(tone)] += tone[:len(audio) - first]
    if not audio.any():
        return audio.astype(np.float32)
    audio = _reverb(audio, sample_rate)
    return (audio * (0.9 / np.abs(audio).max())).astype(np.float32)


def _piano_tone(pitch: int, seconds: float, sample_rate: int) -> np.ndarray:
    """A synthetic piano note, built from what makes a piano sound like a piano:

    - many overtones, slightly sharper than exact multiples of the pitch, as on real
      (stiff) strings, with the higher ones fading faster;
    - two strings per note, a hair out of tune with each other, which makes the sound
      shimmer;
    - a short hammer knock at the start;
    - a decay that is faster for high notes, and a damper that stops the note quickly
      when it ends.
    """
    frequency = 440.0 * 2 ** ((pitch - 69) / 12)
    t = np.arange(int((seconds + _RELEASE) * sample_rate)) / sample_rate
    stiffness = 0.0004 * 2 ** ((pitch - 60) / 24)
    decay = 0.7 + frequency / 600
    tone = np.zeros_like(t)
    for k in range(1, 16):
        partial = k * frequency * np.sqrt(1 + stiffness * k * k)
        if partial > 0.45 * sample_rate:
            break
        amplitude = k ** -1.3 * (1.5 if k == 2 else 1.0)  # a strong octave overtone
        fading = np.exp(-decay * (1 + 0.35 * (k - 1)) * t)
        for detune in (0.9993, 1.0007):  # two strings, about one cent apart
            tone += 0.5 * amplitude * fading * np.sin(2 * np.pi * partial * detune * t)
    knock = np.random.default_rng(pitch).normal(size=len(t)) * np.exp(-t / 0.008)
    knock = np.convolve(knock, np.ones(8) / 8, mode="same")  # muffled: a felt hammer
    attack = np.minimum(t / 0.002, 1.0)
    damper = np.clip((seconds + _RELEASE - t) / _RELEASE, 0.0, 1.0) ** 2
    return 0.25 * (tone + 0.15 * knock) * attack * damper


def _reverb(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Add the sound of a room: the dry sound mixed with a decaying, muffled echo."""
    t = np.arange(int(_REVERB_SECONDS * sample_rate)) / sample_rate
    response = np.random.default_rng(0).normal(size=len(t)) * np.exp(-6.9 * t / _REVERB_SECONDS)
    response = np.convolve(response, np.ones(12) / 12, mode="same")  # walls absorb highs
    response /= np.sqrt(np.sum(response**2))
    size = len(audio) + len(response)
    wet = np.fft.irfft(np.fft.rfft(audio, size) * np.fft.rfft(response, size), size)
    wet = wet[:len(audio) + int(0.5 * sample_rate)]  # keep half a second of tail
    dry = np.pad(audio, (0, len(wet) - len(audio)))
    return (1 - _REVERB_MIX) * dry + _REVERB_MIX * wet * np.abs(dry).max() / np.abs(wet).max()
