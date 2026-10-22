import torch
from torch import nn
import os
import sys
import json
import numpy as np
import utils
from torch import optim
from utils import MinExponentialLR, loss_function, scheduled_sampling, model_outputs_to_piano_grids, metrics_report_batch_counts
from model import VAE
from tensorboardX import SummaryWriter
from dataset import PolyphonicDataset
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import argparse


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project-name", type=str, default="None",
                    help="Name of the project (config, train options, logs and models directories, etc)")
parser.add_argument("-m", "--mask-pad", default=False, action='store_true')
args = parser.parse_args()


###############################################################################
# Load config
###############################################################################
PROJECT_NAME = args.project_name
MASK_PAD = args.mask_pad
config_fn = './model_config_' + PROJECT_NAME + '.json'
train_hyperparams = utils.load_params_dict("train_hyperparams", config_fn)
model_params = utils.load_params_dict("model_params", config_fn)
data_repr_params = utils.load_params_dict("data_repr", config_fn)
# project_params = utils.load_params_dict("project", config_fn)
data_path = utils.load_dataset_path(config_fn)

BATCH_SIZE = train_hyperparams['batch_size']
LEARNING_RATE = train_hyperparams['learning_rate']
DECAY = train_hyperparams['decay']
PARALLEL = train_hyperparams['parallel']
if sys.platform in ['win32', 'darwin']:
    PARALLEL = False
N_EPOCH = train_hyperparams['n_epoch']
REPORT_METRICS_FREQ = train_hyperparams['report_metrics_freq']
CLIP = train_hyperparams['clip']
UP_AUG = train_hyperparams['up_aug']
DOWN_AUG = train_hyperparams['down_aug']
RANDOM_AUG = train_hyperparams['random_aug']
INIT_WEIGHT = train_hyperparams['init_weight']
WEIGHTS = tuple(train_hyperparams['weights'])
TFR1 = tuple(train_hyperparams['teacher_forcing1'])
TFR2 = tuple(train_hyperparams['teacher_forcing2'])
SCHEDULED_SAMPLING_K = train_hyperparams['scheduled_sampling_k']

###############################################################################
# Initialize project
###############################################################################
# PROJECT_NAME = project_params['project_name']
# MODEL_PATH = project_params['model_path']
# LOG_PATH = project_params['log_path']
MODEL_PATH = 'models_' + PROJECT_NAME + '/'
LOG_PATH = 'log_' + PROJECT_NAME + '/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
else:
    ans = input('Model path is not empty. '
                'Are you sure you want to continue?[y/n]')
    if ans == 'y':
        pass
    else:
        quit()

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
else:
    ans = input('Log path is not empty. '
                'Are you sure you want to continue?[y/n]')
    if ans == 'y':
        pass
    else:
        quit()

loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'loss'))
kl_writer = SummaryWriter(os.path.join(LOG_PATH, 'kl_div'))
pitch_writer = SummaryWriter(os.path.join(LOG_PATH, 'pitch_loss'))
dur_writer = SummaryWriter(os.path.join(LOG_PATH, 'dur_loss'))
f1_tot_writer = SummaryWriter(os.path.join(LOG_PATH, 'f1_tot'))
f1_pitch_writer = SummaryWriter(os.path.join(LOG_PATH, 'f1_pitch'))
f1_dur_writer = SummaryWriter(os.path.join(LOG_PATH, 'f1_dur'))
print('Project initialized.', flush=True)

###############################################################################
# load data
###############################################################################
# train_path = os.path.join(data_path, 'pop909+mlpv_t32_train_fix1.npy')
# val_path = os.path.join(data_path, 'pop909+mlpv_t32_val_fix1.npy')
if PROJECT_NAME[-13:] == "_higherKL_aug" or PROJECT_NAME[-13:] == "_aug_higherKL":
    train_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-13] + '_train.npy')
    val_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-13] + '_eval.npy')
elif PROJECT_NAME[-4:] == "_aug":
    train_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-4] + '_train.npy')
    val_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-4] + '_eval.npy')
elif PROJECT_NAME[-9:] == "_higherKL":
    train_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-9] + '_train.npy')
    val_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME[:-9] + '_eval.npy')
else:
    train_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME + '_train.npy')
    val_path = os.path.join(data_path, 'dataset_' + PROJECT_NAME + '_eval.npy')
dataset_path = os.path.join(data_path, 'dataset.csv')
metadata_path = os.path.join(data_path, 'metadata.csv')
train_dataset = PolyphonicDataset(filepath_data_preprocessed = train_path, 
                                  index_table = True,
                                  filepath_dataset = dataset_path,
                                  filepath_metadata = metadata_path,
                                  shift_low = DOWN_AUG, 
                                  shift_high = UP_AUG, 
                                  shift_random = RANDOM_AUG,
                                  max_simu_note = data_repr_params['max_simu_note'])
val_dataset = PolyphonicDataset(filepath_data_preprocessed = val_path, 
                                index_table = True,
                                filepath_dataset = dataset_path,
                                filepath_metadata = metadata_path,
                                shift_low = 0, 
                                shift_high = 0, 
                                shift_random = 0,
                                max_simu_note = data_repr_params['max_simu_note'])
seq_len = train_dataset.seq_len
dur_width=math.ceil(math.log2(seq_len))
print(f"Size of train dataset: {len(train_dataset)}, eval dataset: {len(val_dataset)}")
print(f"Sequence lengths in this dataset: {seq_len}")

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, True)

print('Dataset successfully loaded!', flush=True)

###############################################################################
# model parameter
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(max_simu_note=data_repr_params['max_simu_note'],
            max_pitch=data_repr_params['max_pitch'],
            min_pitch=data_repr_params['min_pitch'],
            pitch_sos=data_repr_params['pitch_sos'],
            pitch_eos=data_repr_params['pitch_eos'],
            pitch_pad=data_repr_params['pitch_pad'],
            dur_pad=data_repr_params['dur_pad'],
            dur_width=dur_width,
            num_step=seq_len,
            note_emb_size=model_params['note_emb_size'],
            enc_notes_hid_size=model_params['enc_notes_hid_size'],
            enc_time_hid_size=model_params['enc_time_hid_size'],
            z_size=model_params['z_size'],
            dec_emb_hid_size=model_params['dec_emb_hid_size'],
            dec_time_hid_size=model_params['dec_time_hid_size'],
            dec_notes_hid_size=model_params['dec_notes_hid_size'],
            dec_z_in_size=model_params['dec_z_in_size'],
            dec_dur_hid_size=model_params['dec_dur_hid_size'],
            device=device
            )

if INIT_WEIGHT:
    model.apply(utils.init_weights)
    print('The parameters in the model are initialized!')
print(f'The model has {utils.count_parameters(model):,} trainable parameters')

if PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    model = model.module
else:
    model = model.to(device)
print('Model loaded!')

###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pitch_loss_func = nn.CrossEntropyLoss(ignore_index=model.pitch_pad)
dur_loss_func = nn.CrossEntropyLoss(ignore_index=model.dur_pad)
normal = utils.standard_normal(model.z_size)
if DECAY:
    scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=5e-5)


###############################################################################
# Main
###############################################################################
def train(model, train_loader,
          optimizer, pitch_criterion, dur_criterion, normal, weights,
          decay, clip, epoch):
    model.train()
    epoch_loss = 0.
    epoch_pitch_loss = epoch_dur_loss = epoch_kl_loss = 0.
    num_batch = len(train_loader)

    # for i, batch in enumerate(train_loader):
    for batch_idx, (batch, idx) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            batch = batch.cuda()
        optimizer.zero_grad()
        tfr1 = scheduled_sampling(((epoch + batch_idx / num_batch) / N_EPOCH),
                                  TFR1[0], TFR1[1])
        tfr2 = scheduled_sampling(((epoch + batch_idx / num_batch) / N_EPOCH),
                                  TFR2[0], TFR2[1])
        # tfr1 = scheduled_sampling_k(epoch + i / num_batch, SCHEDULED_SAMPLING_K)
        # tfr2 = scheduled_sampling_k(epoch + i / num_batch, SCHEDULED_SAMPLING_K)
        recon_pitch, recon_dur, dist = model(x = batch, 
                                             inference = False, 
                                             sample = True, 
                                             teacher_forcing_ratio1 = tfr1, 
                                             teacher_forcing_ratio2 = tfr2)
        recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
        recon_dur = recon_dur.view(-1, recon_dur.size(-1))
        loss, pitch_loss, dur_loss, kl_loss = \
            loss_function(recon_pitch,
                          batch[:, :, 1:, 0].contiguous().view(-1),
                          recon_dur,
                          batch[:, :, 1:, 1:].contiguous().view(-1),
                          dist, pitch_criterion, dur_criterion, dur_width, normal,
                          weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_pitch_loss += pitch_loss.item()
        epoch_dur_loss += dur_loss.item()
        epoch_kl_loss += kl_loss.item()

        loss_writer.add_scalar('batch_train_loss %d' % num_batch, loss.item(),
                               epoch * num_batch + batch_idx)
        kl_writer.add_scalar('batch_train_loss %d' % num_batch, kl_loss.item(),
                             epoch * num_batch + batch_idx)
        pitch_writer.add_scalar('batch_train_loss %d' % num_batch,
                                pitch_loss.item(),
                                epoch * num_batch + batch_idx)
        dur_writer.add_scalar('batch_train_loss %d' % num_batch,
                              dur_loss.item(),
                              epoch * num_batch + batch_idx)
        if decay:
            scheduler.step()
    return (epoch_loss / num_batch, epoch_pitch_loss / num_batch,
            epoch_dur_loss / num_batch, epoch_kl_loss / num_batch)


def evaluate(model, val_loader,
             pitch_criterion, dur_criterion, normal, weights, epoch):
    model.eval()
    epoch_loss = 0.
    epoch_pitch_loss = epoch_dur_loss = epoch_kl_loss = 0.
    num_batch = len(val_loader)
    with torch.no_grad():
        # for i, batch in enumerate(val_loader):
        for batch_idx, (batch, idx) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                batch = batch.cuda()
            # tfr1 = scheduled_sampling(((epoch + batch_idx / num_batch) / N_EPOCH),
            #                           TFR1[0], TFR1[1])
            # tfr2 = scheduled_sampling(((epoch + batch_idx / num_batch) / N_EPOCH),
            #                           TFR2[0], TFR2[1])
            # tfr1 = scheduled_sampling_k(epoch + batch_idx / num_batch, SCHEDULED_SAMPLING_K)
            # tfr2 = scheduled_sampling_k(epoch + batch_idx / num_batch, SCHEDULED_SAMPLING_K)
            recon_pitch, recon_dur, dist = model(x = batch,
                                                 inference = True,
                                                 sample = True,
                                                 teacher_forcing_ratio1 = 0,
                                                 teacher_forcing_ratio2 = 0)
            recon_pitch = recon_pitch.view(-1, recon_pitch.size(-1))
            recon_dur = recon_dur.view(-1, recon_dur.size(-1))
            loss, pitch_loss, dur_loss, kl_loss = \
                loss_function(recon_pitch,
                              batch[:, :, 1:, 0].contiguous().view(-1),
                              recon_dur,
                              batch[:, :, 1:, 1:].contiguous().view(-1),
                              dist, pitch_criterion, dur_criterion, dur_width, normal,
                              weights)
            epoch_loss += loss.item()
            epoch_pitch_loss += pitch_loss.item()
            epoch_dur_loss += dur_loss.item()
            epoch_kl_loss += kl_loss.item()
            loss_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                   loss.item(),
                                   epoch * num_batch + batch_idx)
            kl_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                 kl_loss.item(),
                                 epoch * num_batch + batch_idx)
            pitch_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                    pitch_loss.item(),
                                    epoch * num_batch + batch_idx)
            dur_writer.add_scalar('batch_eval_loss %d' % num_batch,
                                  dur_loss.item(),
                                  epoch * num_batch + batch_idx)
    return (epoch_loss / num_batch, epoch_pitch_loss / num_batch,
            epoch_dur_loss / num_batch, epoch_kl_loss / num_batch)


def metrics_report(model, data_loader, dataset, mask_pad):
    model.eval()

    n_batches = len(data_loader)

    # Columns: gt_notes_n, gt_notes_rec_tot, gt_notes_rec_pitch, gt_notes_rec_dur, rec_notes_n, rec_notes_gt_tot, rec_notes_gt_pitch, rec_notes_gt_dur
    counts_for_metrics = torch.zeros((n_batches, 8))

    with torch.no_grad():
        for batch_idx, (batch, idx) in enumerate(tqdm(data_loader)):
            if torch.cuda.is_available():
                batch = batch.cuda()
            pitch_outs, dur_outs, dist = model(x = batch, 
                                               inference = True,
                                               sample = True,
                                               teacher_forcing_ratio1 = 0,
                                               teacher_forcing_ratio2 = 0)

            rec_piano_grids = model_outputs_to_piano_grids(pitch_outs.to("cpu"), dur_outs.to("cpu"))
            gt_piano_grids = batch.to("cpu")

            counts_for_metrics[batch_idx] = metrics_report_batch_counts(gt_piano_grids, rec_piano_grids, dataset, idx, mask_pad)

    # Compute total counts
    gt_notes_n, gt_notes_rec_tot, gt_notes_rec_pitch, gt_notes_rec_dur, \
          rec_notes_n, rec_notes_gt_tot, rec_notes_gt_pitch, rec_notes_gt_dur = torch.sum(counts_for_metrics, dim=0)

    # Compute metrics
    precision_total = 0 if rec_notes_n == 0 else rec_notes_gt_tot / rec_notes_n
    recall_total = gt_notes_rec_tot / gt_notes_n
    f1_total = 0 if (precision_total + recall_total == 0) else 2 * (precision_total * recall_total) / (precision_total + recall_total)
    precision_pitch = 0 if rec_notes_n == 0 else rec_notes_gt_pitch / rec_notes_n
    recall_pitch = gt_notes_rec_pitch / gt_notes_n
    f1_pitch = 0 if (precision_pitch + recall_pitch == 0) else 2 * (precision_pitch * recall_pitch) / (precision_pitch + recall_pitch)
    precision_dur = 0 if rec_notes_gt_pitch == 0 else rec_notes_gt_dur / rec_notes_gt_pitch
    recall_dur = 0 if gt_notes_rec_pitch == 0 else gt_notes_rec_dur / gt_notes_rec_pitch
    f1_dur = 0 if (precision_dur + recall_dur == 0) else 2 * (precision_dur * recall_dur) / (precision_dur + recall_dur)

    # Return dictionary of metrics
    return {"precision_total": precision_total,
            "recall_total": recall_total,
            "f1_total": f1_total,
            "precision_pitch": precision_pitch,
            "recall_pitch": recall_pitch,
            "f1_pitch": f1_pitch,
            "precision_dur": precision_dur,
            "recall_dur": recall_dur,
            "f1_dur": f1_dur}


best_valid_loss = float('inf')
for epoch in range(N_EPOCH):
    print(f'Start Epoch: {epoch + 1:02}', flush=True)

    start_time = time.time()
    train_loss, train_pl, train_dl, train_kl = \
        train(model, train_loader,
              optimizer, pitch_loss_func, dur_loss_func, normal,
              WEIGHTS, DECAY, CLIP, epoch)

    valid_loss, valid_pl, valid_dl, valid_kl = \
        evaluate(model, val_loader,
                 pitch_loss_func, dur_loss_func, normal,
                 WEIGHTS, epoch)
    end_time = time.time()

    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    torch.save(model.state_dict(), os.path.join(MODEL_PATH,
                                                'pgrid-epoch-model.pt'))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(),
                   os.path.join(MODEL_PATH, 'pgrid-best-valid-model.pt'))

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s',
          flush=True)
    print(
        f'\tTrain Loss: {train_loss:.3f}', flush=True)
    print(
        f'\t Val. Loss: {valid_loss:.3f}', flush=True)
    loss_writer.add_scalar('epoch_train_loss', train_loss, epoch+1)
    loss_writer.add_scalar('epoch_eval_loss', valid_loss, epoch+1)

    kl_writer.add_scalar('epoch_train_loss', train_kl, epoch+1)
    pitch_writer.add_scalar('epoch_train_loss', train_pl, epoch+1)
    dur_writer.add_scalar('epoch_train_loss', train_dl, epoch+1)

    kl_writer.add_scalar('epoch_eval_loss', valid_kl, epoch+1)
    pitch_writer.add_scalar('epoch_eval_loss', valid_pl, epoch+1)
    dur_writer.add_scalar('epoch_eval_loss', valid_dl, epoch+1)

    if (epoch+1) % REPORT_METRICS_FREQ == 0:
        print("Computing objective performance metrics...")

        print("Computing performance metrics on train dataset...")
        metrics_report_train = metrics_report(model, train_loader, train_dataset, MASK_PAD)
        f1_tot_writer.add_scalar('epoch_train_f1', metrics_report_train["f1_total"], epoch+1)
        print(f"Train F1: {metrics_report_train['f1_total']}")
        f1_pitch_writer.add_scalar('epoch_train_f1', metrics_report_train["f1_pitch"], epoch+1)
        f1_dur_writer.add_scalar('epoch_train_f1', metrics_report_train["f1_dur"], epoch+1)

        print("Computing performance metrics on eval dataset...")
        metrics_report_eval = metrics_report(model, val_loader, val_dataset, MASK_PAD)
        f1_tot_writer.add_scalar('epoch_eval_f1', metrics_report_eval["f1_total"], epoch+1)
        print(f"Val. F1: {metrics_report_eval['f1_total']}")
        f1_pitch_writer.add_scalar('epoch_eval_f1', metrics_report_eval["f1_pitch"], epoch+1)
        f1_dur_writer.add_scalar('epoch_eval_f1', metrics_report_eval["f1_dur"], epoch+1)


torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'pgrid-last-model.pt'))
print('Model Saved!', flush=True)


if __name__ == '__main__':
    pass
