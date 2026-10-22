import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils
import math
import random

# def clean_dataset(dataset_path):
#     dataset = np.load(dataset_path)
#     for data in dataset:
#         onset_data = data[:, :]


# def calculate_low_high(dataset_path):
#     """14, 115"""
#     dataset = np.load(dataset_path)
#     print(dataset[:, :, 79])
#     l = 127
#     h = 0
#     num_notes = 0
#     for data in dataset:

#         onset_data = data[:, :] == 2
#         # print(onset_data.shape)
#         sustain_data = data[:, :] == 1
#         silence_data = data[:, :] == 0
#         pr = np.stack([onset_data, sustain_data, silence_data],
#                       axis=2).astype(bool)
#         # pr: (32, 128, 3)
#         pr_matrix = utils.piano_roll_to_target(pr)
#         # piano_grid = utils.target_to_3dtarget(pr_matrix, max_note_count=20,
#         #                                       max_pitch=127,
#         #                                       min_pitch=0, pitch_pad_ind=110,
#         #                                       pitch_sos_ind=128,
#         #                                       pitch_eos_ind=129)
#         # try:
#         low, high, _, nn = utils.get_low_high_dur_count(pr_matrix)
#         nn = nn.max()
#         if low < l:
#             l = low
#         if high > h:
#             h = high
#         if nn > num_notes:
#             num_notes = nn

#             # print(low, high)
#         # except ValueError:
#         #     import matplotlib.pyplot as plt
#         #     plt.figure()
#         #     ax = plt.subplot(131)
#         #     ax.imshow(pr[:, :, 0])
#         #     ax = plt.subplot(132)
#         #     ax.imshow(pr[:, :, 1])
#         #     ax = plt.subplot(133)
#         #     ax.imshow(pr[:, :, 2])
#         #     plt.show()
#     return l, h, num_notes


class PolyphonicDataset(Dataset):

    def __init__(self, filepath_data_preprocessed, index_table = True, filepath_dataset = "data/dataset.csv", filepath_metadata = "data/metadata.csv", shift_low=0, shift_high=0, shift_random=0, max_simu_note=16, constant_manual_shift = 0, chop_factor = 0):
        super(PolyphonicDataset, self).__init__()
        self.filepath_data_preprocessed = filepath_data_preprocessed
        self.filepath_dataset = filepath_dataset
        self.filepath_metadata = filepath_metadata
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.data = np.load(self.filepath_data_preprocessed)
        self.index_table = pd.read_csv(self.filepath_data_preprocessed[:-4] + '_index.csv') if index_table else None
        self.dataset = pd.read_csv(self.filepath_dataset)
        self.metadata = pd.read_csv(self.filepath_metadata)
        self.seq_len = self.data.shape[1]
        self.dur_width = math.ceil(math.log2(self.seq_len))
        self.max_simu_note = max_simu_note
        self.shift_random = shift_random
        if self.shift_random != 0: # make "random" transpositions reproducible
            random.seed(2023)
            possible_shifts = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
            self.shifts_list = [0] * (len(self.data) * (self.shift_random + 1)) # we want every observation to be present unshifted once
            for id in range(len(self.data)): # determine "random" transpositions used for each observation
                self.shifts_list[(id*(self.shift_random+1) + 1):((id+1)*(self.shift_random+1))] = random.sample(population=possible_shifts, k=self.shift_random)
        self.constant_manual_shift = constant_manual_shift
        self.chop_factor = chop_factor


    def __len__(self):
        # consider data augmentation here
        if self.shift_random != 0:
            return len(self.data) * (self.shift_random + 1)
        else: 
            return len(self.data) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, id):
        # # separate id into (no, shift) pair
        # no = id // (self.shift_high - self.shift_low + 1)
        # shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low
        # data = self.data[no, :, :]
        # # perform pitch shifting using np.roll and masking
        # data = np.roll(data, shift, axis=1)
        # if shift > 0:
        #     data[:, :shift] = 0
        # elif shift < 0:
        #     data[:, shift:] = 0
        # # if you want to convert data into MIDI message,
        # # insert your converter code after this line.
        # onset_data = data[:, :] == 2
        # sustain_data = data[:, :] == 1
        # silence_data = data[:, :] == 0
        # pr = np.stack([onset_data, sustain_data, silence_data],
        #               axis=2).astype(bool)
        # # pr: (32, 128, 3)
        # pr_matrix = utils.piano_roll_to_target(pr)
        # # 14 - 115, 14
        # # 21 - 105， 11

        if self.shift_random != 0:
            no = id // (self.shift_random + 1)
            shift = self.shifts_list[id]
        elif self.constant_manual_shift != 0:
            no = id
            shift = self.constant_manual_shift
        else:
            no = id // (self.shift_high - self.shift_low + 1)
            shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low

        data = self.data[no, :, :]
        # perform pitch shifting using np.roll and masking
        if shift != 0:
            data = np.roll(data, shift, axis=1)
            if shift > 0:
                data[:, :shift] = 0
            elif shift < 0:
                data[:, shift:] = 0

        piano_grid = utils.target_to_3dtarget(data, max_note_count=self.max_simu_note,
                                              max_pitch=128,
                                              min_pitch=0, pitch_pad_ind=130,
                                              pitch_sos_ind=128,
                                              pitch_eos_ind=129,
                                              seq_len = self.seq_len,
                                              dur_width = self.dur_width,
                                              chop_factor = self.chop_factor)
        return piano_grid.astype(np.int64), id

    def get_obs_id(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return obs_idx

    def get_obs(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.dataset[self.dataset["obs"] == obs_idx]

    def get_corpus(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["corpus"]
    
    def get_piece(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["piece"]

    def get_music_era(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["music_era"]

    def get_century(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["century"]
    
    def get_year(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["year"]

    def get_style(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["style"]

    def get_n_staffs(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_staffs"]

    def get_n_instruments(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_instruments"]

    def get_key(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["key"]

    def get_timesig(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["timesig"]

    def get_seq_len(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_simu_notes"]
    
    def get_n_notes(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_notes"]

    def get_n_notes_per_simu_note(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_notes_per_simu_note"]

    def get_avg_note_pitch(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["avg_note_pitch"]

    def get_pitch_range(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["pitch_range"]

    def get_avg_note_duration(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["avg_note_duration"]

    def get_n_chords(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["n_chords"]

    def get_first_chord(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["first_chord"]

    def get_first_chord_type(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["first_chord_type"]

    def get_first_chord_numeral(self, id):
        no = (id // (self.shift_random + 1)) if (self.shift_random != 0) else (id // (self.shift_high - self.shift_low + 1))
        obs_idx = self.index_table[self.index_table["preprocessed_idx"] == no].iloc[0]["obs_idx"]
        return self.metadata[self.metadata["obs"] == obs_idx].iloc[0]["first_chord_numeral"]



# def reshape_grid_to_plot(piano_grid):
#     y = np.copy(piano_grid)
#     y = piano_grid.transpose([1, 0, 2]).reshape((-1, 6 * 32))
#     return y


if __name__ == '__main__':
    file_path_train = "../data/toy_dataset_pianotree_4_4_train.npy"
    file_path_eval = "../data/toy_dataset_pianotree_4_4_eval.npy"
    train_dataset = PolyphonicDataset(file_path_train, 0, 0) #-3, +3 in OG code
    val_dataset = PolyphonicDataset(file_path_eval, 0, 0)  # DO NOT augment on validation dataset!
    print(f"Length of train dataset: {len(train_dataset)}. Length of eval dataset: {len(val_dataset)}.")
    # print(train_dataset[1124].shape)
    # x = train_dataset[1124]
    # y = reshape_grid_to_plot(x)
    # import matplotlib.pyplot as plt
    # plt.imshow(y)
    # plt.show()
    # for i in range(0, )

    # # print(train_dataset[(2, 3, 4, 5)])
    # data_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    # for i_batch, sample_batched in enumerate(data_loader):
    #     print(i_batch)
    #     print(len(sample_batched))
    #     print(type(sample_batched))
    #     print(sample_batched.shape)
    #     print(sample_batched[0].shape)
