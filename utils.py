import numpy as np
import pandas as pd
import json, sys, os
from torch import nn
import torch
from torch.distributions import kl_divergence, Normal
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import math

def load_dataset_path(fn='model_config.json'):
    with open(fn) as f:
        paths = json.load(f)['dataset_path']

    train_val_path = paths['hpc_data_path']
    return train_val_path

def load_params_dict(key, fn='model_config.json'):
    with open(fn) as f:
        dict = json.load(f)[key]
    return dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def standard_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(recon_pitch, pitch, recon_dur, dur,
                  dist, pitch_criterion, dur_criterion, dur_width, normal,
                  weights=(1, 1, .1)):
    # bs = dist.mean.size(0)
    pitch_loss = pitch_criterion(recon_pitch, pitch)

    recon_dur = recon_dur.view(-1, dur_width, 2)
    dur = dur.view(-1, dur_width)

    dur_losses = []
    for dur_width_idx in range(dur_width):
        dur_losses.append(dur_criterion(recon_dur[:, dur_width_idx, :], dur[:, dur_width_idx]))

    # dur0 = dur_criterion(recon_dur[:, 0, :], dur[:, 0])
    # dur1 = dur_criterion(recon_dur[:, 1, :], dur[:, 1])
    # dur2 = dur_criterion(recon_dur[:, 2, :], dur[:, 2])
    # dur3 = dur_criterion(recon_dur[:, 3, :], dur[:, 3])
    # dur4 = dur_criterion(recon_dur[:, 4, :], dur[:, 4])

    if dur_width == 3:
        w = torch.tensor([1, 0.6, 0.4],
                    dtype=float,
                    device=recon_dur.device)
    elif dur_width == 4:
        w = torch.tensor([1, 0.6, 0.4, 0.3],
                    dtype=float,
                    device=recon_dur.device)
    elif dur_width == 5:
        w = torch.tensor([1, 0.6, 0.4, 0.3, 0.3],
            dtype=float,
            device=recon_dur.device)
    elif dur_width == 6:
        w = torch.tensor([1, 0.6, 0.4, 0.3, 0.3, 0.3],
            dtype=float,
            device=recon_dur.device)
    elif dur_width == 7:
        w = torch.tensor([1, 0.6, 0.4, 0.3, 0.3, 0.3, 0.3],
            dtype=float,
            device=recon_dur.device)
    else:
        return "Error: dur width not supported"

    # dur_loss = w[0] * dur0 + w[1] * dur1 + w[2] * dur2 + w[3] * dur3 + \
    #            w[4] * dur4

    dur_loss = 0
    for dur_width_idx in range(dur_width):
        dur_loss += w[dur_width_idx] * dur_losses[dur_width_idx]

    kl_div = kl_divergence(dist, normal).mean()
    loss = weights[0] * pitch_loss + weights[1] * dur_loss + \
        weights[2] * kl_div
    return loss, pitch_loss, dur_loss, kl_div


# Useful function for how long epochs take
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def scheduled_sampling(i, high=0.7, low=0.05):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y

# def scheduled_sampling_k(i, k):
#     return k / (k + np.exp(i / k))

def number_to_note(number: int) -> str:
    NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)

    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES
    assert 0 <= number <= 127
    note = NOTES[number % NOTES_IN_OCTAVE]

    return str(note) + str(octave-1)

def note_to_number(note: str) -> int:
    octave = int(note[-1])
    note = note[:-1]
    NOTES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)

    assert note in NOTES
    assert octave in OCTAVES

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * (octave+1))

    assert 0 <= note <= 127

    return note

def piano_roll_to_target(pr):
    #  pr: (32, 128, 3), dtype=bool

    # Assume that "not (first_layer or second layer) = third_layer"
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    # To int dtype can make addition work
    pr = pr.astype(int)
    # Initialize a matrix to store the duration of a note on the (32, 128) grid
    pr_matrix = np.zeros((32, 128))

    for i in range(31, -1, -1):
        # At each iteration
        # 1. Assure that the second layer accumulates the note duration
        # 2. collect the onset notes in time step i, and mark it on the matrix.

        # collect
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        # Accumulate
        # pr[i - 1, :, 1] += pr[i, :, 1]
        # pr[i - 1, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i, onset_idx, 1] = 0  # the onset note should be set 0.
        pr[i - 1, :, 1] += pr[i, :, 1]

    return pr_matrix


def target_to_3dtarget(pr_mat, max_note_count=11, max_pitch=107, min_pitch=22,
                       pitch_pad_ind=88, dur_pad_ind=2,
                       pitch_sos_ind=86, pitch_eos_ind=87,
                       seq_len=32, dur_width=5,
                       chop_factor=0):
    """
    :param pr_mat: (32, 128) matrix. pr_mat[t, p] indicates a note of pitch p,
    started at time step t, has a duration of pr_mat[t, p] time steps.
    :param max_note_count: the maximum number of notes in a time step,
    including <sos> and <eos> tokens.
    :param max_pitch: the highest pitch in the dataset.
    :param min_pitch: the lowest pitch in the dataset.
    :param pitch_pad_ind: see return value.
    :param dur_pad_ind: see return value.
    :param pitch_sos_ind: sos token.
    :param pitch_eos_ind: eos token.
    :return: pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    padded with <sos> and <eos> tokens in the pitch column, but with pad token
    for dur columns.
    """
    pitch_range = max_pitch - min_pitch + 1  # including pad
    pr_mat3d = np.ones((seq_len, max_note_count, 1 + dur_width), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(seq_len, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        duration_capped = np.minimum(int(pr_mat[t, p]) - 1, seq_len - 1) # hardfix for durations longer than seq_len
        if chop_factor != 0 and duration_capped > 1:
            duration_chopped = math.floor(duration_capped / chop_factor) # odd durations will be rounded down
            binary = np.binary_repr(duration_chopped, width=dur_width)
            n_notes = duration_capped // duration_chopped
            for i in range(n_notes):
                try:
                    pr_mat3d[t+i*duration_chopped, cur_idx[t+i*duration_chopped], 1:(dur_width+1)] = \
                        np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
                except:
                    print("t")
                    print(t)
                    print("duration capped")
                    print(duration_capped)
                    print("duration chopped")
                    print(duration_chopped)
                    print("n_notes")
                    print(n_notes)

                cur_idx[t+i*duration_chopped] += 1
        else: 
            binary = np.binary_repr(duration_capped, width=dur_width)
            pr_mat3d[t, cur_idx[t], 1:(dur_width+1)] = \
                np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
            cur_idx[t] += 1

    cur_idx[np.where(cur_idx > (max_note_count - 1))] = (max_note_count - 1) # hardfix for number of notes at same onset > max_note_count 
    pr_mat3d[np.arange(0, seq_len), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d


def get_low_high_dur_count(pr_mat):
    # pr_mat (32, 128)
    # return the maximum duration
    # return the pitch range
    # return the number of notes at each column

    pitch_range = np.where(pr_mat != 0)[1]
    low_pitch = pitch_range.min()
    high_pitch = pitch_range.max()
    pitch_dur = pr_mat.max()
    num_notes = np.count_nonzero(pr_mat, axis=-1)
    return low_pitch, high_pitch, pitch_dur, num_notes


def model_outputs_to_piano_grids(pitch_outs, dur_outs):
    minibatch_size = pitch_outs.shape[0]
    seq_len = pitch_outs.shape[1]
    simunote_max_n = pitch_outs.shape[2] + 1
    dur_width = dur_outs.shape[3]
    seq = torch.zeros((minibatch_size, seq_len, simunote_max_n, 1 + dur_width))

    for batch_idx in range(minibatch_size):
        pitch_outs_idx = torch.argmax(pitch_outs[batch_idx], dim=2).unsqueeze(2)
        dur_outs_idx = torch.argmax(dur_outs[batch_idx], dim=3)
        outs_idx = torch.cat((pitch_outs_idx, dur_outs_idx), dim=2)
        sos_tokens = torch.cat((torch.tensor([128]), torch.tensor([2]).repeat(dur_width))).repeat(seq_len).reshape(seq_len, 1, 1 + dur_width)
        seq[batch_idx] = torch.cat((sos_tokens, outs_idx), dim = 1)
    return seq.to(torch.int)


def piano_grids_to_seqs(piano_grids):
    minibatch_size = piano_grids.shape[0]
    seq_len = piano_grids.shape[1]
    dur_width = math.ceil(math.log2(seq_len))
    simunote_max_n = piano_grids.shape[2]
    seqs = []

    for minibatch_idx in range(minibatch_size):
        piano_grid = piano_grids[minibatch_idx]
        seq_idx = pd.DataFrame(columns = ['simu_note', 'pitch', 'note', 'duration_16'])
        for simu_note in range(seq_len):
            piano_grid_simu_note = piano_grid[simu_note]
            indicators_note_predicted = torch.logical_and(torch.logical_and(piano_grid_simu_note[:, 0] != 128, piano_grid_simu_note[:, 0] != 129), piano_grid_simu_note[:, 0] != 130)
            if torch.any(indicators_note_predicted):
                notes_predicted = piano_grid_simu_note[indicators_note_predicted]
                for i, note in enumerate(notes_predicted.numpy()):
                    pitch = note[0]
                    note_name = number_to_note(pitch)
                    dur_binary = note[1:]
                    binary_multiplier = (np.array([2]).repeat(dur_width)) ** (np.arange(dur_width)[::-1])
                    dur = np.sum(np.multiply(binary_multiplier, dur_binary))+1
                    seq_idx.loc[len(seq_idx)] = {'simu_note': simu_note, 'pitch': pitch, 'note': note_name, 'duration_16': dur}
        seqs.append(seq_idx)
    return seqs


def metrics_report_batch_counts(gt_piano_grids, rec_piano_grids, dataset = None, idx = None, mask_pad = True):

    assert (gt_piano_grids.shape == rec_piano_grids.shape), "Error: Shapes of ground truth and reconstructed piano grids are different"

    n_obs_in_batch = gt_piano_grids.shape[0]

    gt_notes_n = 0
    gt_notes_rec_tot = 0
    gt_notes_rec_pitch = 0
    gt_notes_rec_dur = 0

    rec_notes_n = 0
    rec_notes_gt_tot = 0
    rec_notes_gt_pitch = 0
    rec_notes_gt_dur = 0

    for i in range(n_obs_in_batch):
        gt_piano_grid = gt_piano_grids[i]
        rec_piano_grid = rec_piano_grids[i]
        id = idx[i]

        # If we mask pads, cut the sequences to their actual lengths (ignore padding)
        if mask_pad:
            seq_len = dataset.get_seq_len(int(id))
            gt_piano_grid = gt_piano_grid[0:seq_len, :, :]
            rec_piano_grid = rec_piano_grid[0:seq_len, :, :]

        # Get recall metrics by comparing ground truth to reconstructed
        for t, note_index in zip(*torch.where((gt_piano_grid[:, :, 0] != 128) & (gt_piano_grid[:, :, 0] != 129) & (gt_piano_grid[:, :, 0] != 130))):
            gt_notes_n += 1

            gt_note_pitch = gt_piano_grid[t, note_index, 0]
            gt_note_dur = gt_piano_grid[t, note_index, 1:]
            rec_pitches_t = rec_piano_grid[t, :, 0]
            rec_durations_t = rec_piano_grid[t, :, 1:]

            if torch.any(rec_pitches_t == gt_note_pitch):
                gt_notes_rec_pitch += 1
                if torch.all(rec_durations_t[torch.where(rec_pitches_t == gt_note_pitch)] == gt_note_dur):
                    gt_notes_rec_dur += 1
                    gt_notes_rec_tot += 1

        # Get precision metrics by comparing reconstructed to ground truth
        for t, note_index in zip(*torch.where((rec_piano_grid[:, :, 0] != 128) & (rec_piano_grid[:, :, 0] != 129) & (rec_piano_grid[:, :, 0] != 130))):
            rec_notes_n += 1

            rec_note_pitch = rec_piano_grid[t, note_index, 0]
            rec_note_dur = rec_piano_grid[t, note_index, 1:]
            gt_pitches_t = gt_piano_grid[t, :, 0]
            gt_durations_t = gt_piano_grid[t, :, 1:]

            if torch.any(gt_pitches_t == rec_note_pitch):
                rec_notes_gt_pitch += 1
                if torch.all(gt_durations_t[torch.where(gt_pitches_t == rec_note_pitch)] == rec_note_dur):
                    rec_notes_gt_dur += 1
                    rec_notes_gt_tot += 1

    return torch.tensor([gt_notes_n, gt_notes_rec_tot, gt_notes_rec_pitch, gt_notes_rec_dur, \
                         rec_notes_n, rec_notes_gt_tot, rec_notes_gt_pitch, rec_notes_gt_dur])
