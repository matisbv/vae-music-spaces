# Import packages
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-cf", "--corpus-filter", type=str, default="None",
                    help="Corpus filter to apply to the full dataset before preprocessing")
parser.add_argument("-pf", "--piece-filter", type=str, default="None",
                    help="Piece filter to apply to the full dataset before preprocessing")
parser.add_argument("-tf", "--timesig-filter", type=str, default="None",
                    help="Timesig filter to apply to the full dataset before preprocessing")
parser.add_argument("-sc", "--seqlen-cutoff", type=int, default=0,
                    help="Manual cutoff for maximum sequence length in our dataset. Observations longer than this cutoff will be \
                    discarded. If not provided, no cutoff will be set.")
parser.add_argument("-tp", "--test-proportion", type=float, default=0.1,
                    help="The desired proportion of the size of the test dataset (between 0 and 1).")
args = parser.parse_args()

# Path to full dataset
filepath_dataset = 'data/dataset.csv'

# Dataset name
preprocessed_dataset_name = "dataset"
if args.corpus_filter != "None":
    preprocessed_dataset_name = preprocessed_dataset_name + '_' + args.corpus_filter
if args.piece_filter != "None":
    preprocessed_dataset_name = preprocessed_dataset_name + '_' + args.piece_filter
if args.timesig_filter != "None":
    preprocessed_dataset_name = preprocessed_dataset_name + '_' + args.timesig_filter.replace("/", "-")
if args.seqlen_cutoff != 0:
    preprocessed_dataset_name = preprocessed_dataset_name + '_seqlencutoff' + str(args.seqlen_cutoff)

# Util functions
def preprocess_dataset(dataset, n_obs, longest_seq_n):
    dataset_preprocessed = np.zeros((n_obs, longest_seq_n, 128))
    index_table = pd.DataFrame(columns = ['preprocessed_idx', 'obs_idx'])

    obs_idx_list = sorted(dataset["obs"].unique())
    for preprocessed_idx in tqdm(range(n_obs)):
        obs_preprocessed = np.zeros((longest_seq_n, 128))
        obs = dataset[dataset["obs"] == obs_idx_list[preprocessed_idx]]
        for index, row in obs.iterrows():
            obs_preprocessed[int(row.simu_note), int(row.pitch - 1)] = row.duration
        dataset_preprocessed[preprocessed_idx] = obs_preprocessed
        index_table.loc[len(index_table)] = {'preprocessed_idx': preprocessed_idx, 'obs_idx': obs_idx_list[preprocessed_idx]}

    dataset_preprocessed = dataset_preprocessed.astype(np.int64)

    dataset_preprocessed_train, dataset_preprocessed_eval, index_table_train, index_table_eval = \
        train_test_split(dataset_preprocessed, index_table, test_size = args.test_proportion, random_state=2023)

    index_table_train["preprocessed_idx"] = np.arange(len(index_table_train))
    index_table_eval["preprocessed_idx"] = np.arange(len(index_table_eval))

    return dataset_preprocessed_train, dataset_preprocessed_eval, index_table_train, index_table_eval

# Program
if __name__ == '__main__':
    print("Loading full dataset...")
    dataset = pd.read_csv(filepath_dataset)
    n_obs = len(dataset["obs"].unique())
    print(f"Done! The dataset contains {n_obs} observations.")
    print(f"Applying filters and cutoffs...")
    if args.corpus_filter != "None":
        print(f"Filtering for corpus {args.corpus_filter}...")
        dataset = dataset[dataset["corpus"] == args.corpus_filter]
    if args.piece_filter != "None":
        print(f"Filtering for piece {args.piece_filter}...")
        dataset = dataset[dataset["piece"] == args.piece_filter]
    if args.timesig_filter != "None":
        print(f"Filtering for time signature {args.timesig_filter}...")
        dataset = dataset[dataset["timesig"] == args.timesig_filter]
    if args.seqlen_cutoff != 0:
        print(f"Applying manual seqlen cutoff of T = {args.seqlen_cutoff}...")
        dataset = dataset[dataset["n_simu_notes"] <= args.seqlen_cutoff]
    n_obs = len(dataset["obs"].unique())
    longest_seq_n = dataset["n_simu_notes"].max()
    print(f"Done! The dataset now contains {n_obs} observations.")
    print("Pre-processing the dataset for model training...")
    print(f"The longest sequence length in the dataset is T = {longest_seq_n}. Padding everything to T = {longest_seq_n}.")
    dataset_preprocessed_train, dataset_preprocessed_eval, index_table_train, index_table_eval = \
        preprocess_dataset(dataset, n_obs, longest_seq_n)
    np.save('data/' + preprocessed_dataset_name + '_train.npy', dataset_preprocessed_train)
    np.save('data/' + preprocessed_dataset_name + '_eval.npy', dataset_preprocessed_eval)
    index_table_train.to_csv('data/' + preprocessed_dataset_name + '_train_index.csv', index = False)
    index_table_eval.to_csv('data/' + preprocessed_dataset_name + '_eval_index.csv', index = False)
    print("Done! Preprocessed train and eval datasets, as well as index tables, saved under data/.")