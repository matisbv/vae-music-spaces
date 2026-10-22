"""PianoTree-VAE: a VAE for polyphonic music with tree-structured encoder and decoder.

Wang, Z., Zhang, Y., Zhang, Y., Jiang, J., Yang, R., Zhao, J. and Xia, G. (2020).
"PianoTree VAE: Structured Representation Learning for Polyphonic Music." ISMIR.

Adapted from the authors' implementation, https://github.com/ZZWaang/PianoTree-VAE,
Copyright (c) 2020 ZZWaang, released under the MIT License (see licenses/PianoTree-VAE.txt).

A bar is a piano-grid (see :mod:`pianotree_dcml.representation`): a sequence of time
steps, each holding the sequence of notes that start there. The model mirrors that tree.

Encoder
    1. Each note (one-hot pitch token + duration bits) is embedded by a linear layer.
    2. A bidirectional GRU reads the notes of each time step: one vector per step.
    3. A bidirectional GRU reads those vectors over time and outputs q(z | bar).
Decoder
    1. A GRU over time steps, conditioned on z, outputs one vector per time step. Its input
       at each step is a summary of the notes of the previous step.
    2. From that vector, a GRU generates the notes of the step one at a time, until <eos>.
    3. Each note's pitch comes from a linear layer and its duration bits from a small GRU.

During training, the two decoder GRUs can be fed the true notes of the previous position
instead of their own predictions (teacher forcing).

Changes from the original implementation: the decoder summarizes its own predicted notes
including <eos>, as it does for the true notes; the duration GRU reads the previous bit as
a 2-way one-hot; and the special-token durations of predicted notes are set to padding, as
in the data.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence

from pianotree_dcml.representation import (
    DUR_PAD,
    N_PITCHES,
    PITCH_EOS,
    PITCH_PAD,
    PITCH_SOS,
    dur_width_for,
)

N_PITCH_TOKENS = N_PITCHES + 2  # the 128 MIDI pitches, <sos> and <eos>; <pad> is never predicted


class PianoTreeVAE(nn.Module):
    def __init__(
        self,
        seq_len: int,
        max_simu_note: int = 16,
        note_emb_size: int = 128,
        enc_notes_hid_size: int = 256,
        enc_time_hid_size: int = 512,
        z_size: int = 512,
        dec_emb_hid_size: int = 128,
        dec_time_hid_size: int = 1024,
        dec_notes_hid_size: int = 512,
        dec_z_in_size: int = 256,
        dec_dur_hid_size: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.max_simu_note = max_simu_note
        self.dur_width = dur_width_for(seq_len)
        self.z_size = z_size

        self.note_embedding = nn.Linear(N_PITCH_TOKENS + self.dur_width, note_emb_size)

        # Encoder
        self.enc_notes_gru = nn.GRU(note_emb_size, enc_notes_hid_size,
                                    batch_first=True, bidirectional=True)
        self.enc_time_gru = nn.GRU(2 * enc_notes_hid_size, enc_time_hid_size,
                                   batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(2 * enc_time_hid_size, z_size)
        self.linear_std = nn.Linear(2 * enc_time_hid_size, z_size)

        # Decoder: time level
        self.z2dec_hid = nn.Linear(z_size, dec_time_hid_size)
        self.z2dec_in = nn.Linear(z_size, dec_z_in_size)
        self.dec_init_input = nn.Parameter(torch.rand(2 * dec_emb_hid_size))
        self.dec_notes_emb_gru = nn.GRU(note_emb_size, dec_emb_hid_size,
                                        batch_first=True, bidirectional=True)
        self.dec_time_gru = nn.GRU(dec_z_in_size + 2 * dec_emb_hid_size, dec_time_hid_size,
                                   batch_first=True)
        # Decoder: note level
        self.dec_time_to_notes_hid = nn.Linear(dec_time_hid_size, dec_notes_hid_size)
        self.dec_notes_gru = nn.GRU(dec_time_hid_size + note_emb_size, dec_notes_hid_size,
                                    batch_first=True)
        self.pitch_out = nn.Linear(dec_notes_hid_size, N_PITCH_TOKENS)
        # Decoder: duration bits
        self.dur_sos = nn.Parameter(torch.rand(2))
        self.dur_hid_linear = nn.Linear(N_PITCH_TOKENS + dec_notes_hid_size, dec_dur_hid_size)
        self.dec_dur_gru = nn.GRU(2, dec_dur_hid_size, batch_first=True)
        self.dur_out = nn.Linear(dec_dur_hid_size, 2)

    def forward(
        self,
        grids: torch.Tensor,
        teacher_forcing_time: float = 0.0,
        teacher_forcing_notes: float = 0.0,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Normal]:
        """Encode a batch of piano-grids and decode them back.

        ``grids`` has shape ``(batch, seq_len, max_simu_note, 1 + dur_width)``. The latent
        code is sampled from q(z | bar), or is its mean if ``sample`` is False. Returns the
        pitch logits ``(batch, seq_len, max_simu_note - 1, N_PITCH_TOKENS)``, the duration
        logits ``(batch, seq_len, max_simu_note - 1, dur_width, 2)`` and q(z | bar).
        """
        notes, lengths = self.embed_grids(grids)
        posterior = self.encode_embedded(notes, lengths)
        z = posterior.rsample() if sample else posterior.mean
        teacher = (notes, lengths) if max(teacher_forcing_time, teacher_forcing_notes) > 0 else None
        pitch_logits, dur_logits = self.decode(
            z, teacher, teacher_forcing_time, teacher_forcing_notes,
            min_lengths=lengths.max(dim=0).values,
        )
        return pitch_logits, dur_logits, posterior

    def encode(self, grids: torch.Tensor) -> Normal:
        """q(z | bar) for a batch of piano-grids."""
        return self.encode_embedded(*self.embed_grids(grids))

    def embed_grids(self, grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed every note token. Also returns the number of tokens of each time step,
        counting <sos> and <eos>."""
        pitch = grids[..., 0]
        lengths = (pitch != PITCH_PAD).sum(-1)
        # <pad> tokens get an all-zero pitch one-hot; they are never read by the GRUs anyway.
        one_hot = F.one_hot(pitch, N_PITCH_TOKENS + 1)[..., :N_PITCH_TOKENS]
        features = torch.cat([one_hot.float(), grids[..., 1:].float()], dim=-1)
        return self.note_embedding(features), lengths

    def encode_embedded(self, notes: torch.Tensor, lengths: torch.Tensor) -> Normal:
        batch, seq_len, max_simu_note, _ = notes.shape
        steps = _summarize(self.enc_notes_gru, notes.flatten(0, 1), lengths.flatten())
        _, hidden = self.enc_time_gru(steps.view(batch, seq_len, -1))
        hidden = hidden.transpose(0, 1).reshape(batch, -1)
        return Normal(self.linear_mu(hidden), self.linear_std(hidden).exp())

    def decode(
        self,
        z: torch.Tensor,
        teacher: tuple[torch.Tensor, torch.Tensor] | None = None,
        teacher_forcing_time: float = 0.0,
        teacher_forcing_notes: float = 0.0,
        min_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent codes into pitch and duration logits.

        ``teacher`` holds the embedded true notes and their lengths (from ``embed_grids``)
        when teacher forcing is used.

        To save time, the notes of a time step are only generated until every bar of the
        batch has predicted <eos>; the logits of later positions are left at zero. When
        computing a loss, pass the longest true number of tokens of each time step as
        ``min_lengths`` (shape ``(seq_len,)``) so that every position with a target is
        generated.
        """
        batch = z.size(0)
        hidden = self.z2dec_hid(z).unsqueeze(0)
        z_in = self.z2dec_in(z).unsqueeze(1)
        if teacher is not None:
            true_notes, true_lengths = teacher
            true_steps = _summarize(
                self.dec_notes_emb_gru, true_notes.flatten(0, 1), true_lengths.flatten()
            ).view(batch, self.seq_len, -1)

        token = self.dec_init_input.expand(batch, 1, -1)
        pitch_logits, dur_logits = [], []
        for t in range(self.seq_len):
            step_summary, hidden = self.dec_time_gru(torch.cat([token, z_in], dim=-1), hidden)
            pitch, dur, predicted, lengths = self._decode_notes(
                step_summary,
                true_notes[:, t] if teacher is not None else None,
                teacher_forcing_notes,
                int(min_lengths[t]) if min_lengths is not None else 0,
            )
            pitch_logits.append(pitch)
            dur_logits.append(dur)
            if t == self.seq_len - 1:
                break
            if teacher is not None and random.random() < teacher_forcing_time:
                token = true_steps[:, t].unsqueeze(1)
            else:
                token = _summarize(self.dec_notes_emb_gru, predicted, lengths).unsqueeze(1)
        return torch.stack(pitch_logits, dim=1), torch.stack(dur_logits, dim=1)

    def _decode_notes(
        self,
        step_summary: torch.Tensor,
        true_notes: torch.Tensor | None,
        teacher_forcing: float,
        min_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the notes of one time step from its summary ``(batch, 1, hidden)``.

        Returns the pitch and duration logits of note positions 1..max_simu_note-1, the
        embeddings of the predicted tokens (starting with <sos>) and the number of
        predicted tokens up to and including the first <eos>. Generation stops once
        every bar has predicted <eos> and at least ``min_length`` tokens exist.
        """
        batch = step_summary.size(0)
        hidden = self.dec_time_to_notes_hid(step_summary).transpose(0, 1)
        sos = self._embed_tokens(
            torch.full((batch,), PITCH_SOS, device=step_summary.device),
            torch.zeros(batch, self.dur_width, dtype=torch.long, device=step_summary.device),
        )
        token = sos.unsqueeze(1)
        predicted = [sos]
        lengths = torch.full((batch,), self.max_simu_note, device=step_summary.device)
        pitch_logits, dur_logits = [], []
        for i in range(1, self.max_simu_note):
            output, hidden = self.dec_notes_gru(torch.cat([step_summary, token], dim=-1), hidden)
            pitch, dur = self._decode_note(output.squeeze(1))
            pitch_logits.append(pitch)
            dur_logits.append(dur)

            pitch_token = pitch.argmax(-1)
            note = self._embed_tokens(pitch_token, dur.argmax(-1))
            predicted.append(note)
            first_eos = (pitch_token == PITCH_EOS) & (lengths == self.max_simu_note)
            lengths[first_eos] = i + 1
            if i + 1 >= min_length and (lengths < self.max_simu_note).all():
                break

            if true_notes is not None and random.random() < teacher_forcing:
                token = true_notes[:, i].unsqueeze(1)
            else:
                token = note.unsqueeze(1)

        pitch_logits = torch.stack(pitch_logits, dim=1)
        dur_logits = torch.stack(dur_logits, dim=1)
        missing = self.max_simu_note - 1 - pitch_logits.size(1)
        pitch_logits = F.pad(pitch_logits, (0, 0, 0, missing))
        dur_logits = F.pad(dur_logits, (0, 0, 0, 0, 0, missing))
        return pitch_logits, dur_logits, torch.stack(predicted, dim=1), lengths

    def _decode_note(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pitch logits ``(batch, N_PITCH_TOKENS)`` and duration-bit logits
        ``(batch, dur_width, 2)`` of one note, from the note GRU output."""
        pitch = self.pitch_out(output)
        hidden = self.dur_hid_linear(torch.cat([output, pitch], dim=-1)).unsqueeze(0)
        token = self.dur_sos.expand(output.size(0), 1, 2)
        bits = []
        for _ in range(self.dur_width):
            dur_output, hidden = self.dec_dur_gru(token, hidden)
            logits = self.dur_out(dur_output.squeeze(1))
            bits.append(logits)
            token = F.one_hot(logits.argmax(-1), 2).float().unsqueeze(1)
        return pitch, torch.stack(bits, dim=1)

    def _embed_tokens(self, pitch: torch.Tensor, dur_bits: torch.Tensor) -> torch.Tensor:
        """Embed note tokens given as pitch tokens ``(batch,)`` and bits ``(batch, dur_width)``.

        Special tokens get padded durations, as in the data.
        """
        is_special = (pitch >= N_PITCHES).unsqueeze(-1)
        durations = torch.where(is_special, DUR_PAD, dur_bits).float()
        features = torch.cat([F.one_hot(pitch, N_PITCH_TOKENS).float(), durations], dim=-1)
        return self.note_embedding(features)


def _summarize(gru: nn.GRU, notes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Final hidden state of a bidirectional ``gru`` over each sequence of notes.

    ``notes`` is ``(n, max_simu_note, emb)``; returns ``(n, 2 * hidden)``.
    """
    packed = pack_padded_sequence(notes, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, hidden = gru(packed)
    return hidden.transpose(0, 1).reshape(notes.size(0), -1)
