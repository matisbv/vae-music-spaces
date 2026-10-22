# Import packages
import pandas as pd
from tqdm import tqdm
import os
import math

# Path to corpora
filepath_data = 'data/dcml_corpora/'

# Util functions
def create_dataset():
    dataset = pd.DataFrame(columns = ['obs', 'corpus', 'piece', 'timesig', 'n_simu_notes', 'mn', 'staff', 'simu_note', 'pitch', 'duration'])
    obs_idx = 0

    list_corpora = sorted(os.listdir(filepath_data))

    for corpus in list_corpora:
        print(f"Processing corpus {corpus}...")

        list_files = sorted(os.listdir(filepath_data + '/' + corpus + '/notes'))
        list_pieces = [x for x in list_files if x.endswith(".tsv")]

        for piece in tqdm(list_pieces):
            piece_df = pd.read_csv(filepath_data + '/' + corpus + '/notes/' + piece, sep = '\t')

            # Onset approximation
            piece_df["mn_onset_float"] = piece_df["mn_onset"].apply(lambda x: float(eval(x)))
            piece_df["mn_onset_quantized"] = piece_df["mn_onset_float"].apply(lambda x: math.floor(x * 16))

            # Duration approximation
            piece_df["duration_16"] = piece_df["duration"].apply(lambda x: float(eval(x)) * 16)
            piece_df["duration_16_quantized"] = piece_df["duration_16"].apply(lambda x: math.floor(x) if (math.floor(x) > 0) else 1)

            piece_n_bars = max(piece_df['mn'])

            for bar_idx in range(1, piece_n_bars + 1):
                bar_df = piece_df[piece_df['mn'] == bar_idx]

                if len(bar_df) == 0: # for now, we simply ignore empty bars
                    continue

                bar_processed = process_bar(bar_df, corpus_name = corpus, piece_name = piece[:-4], obs_idx = obs_idx)
                dataset = pd.concat([dataset, bar_processed])
                obs_idx += 1
    
    return dataset

def process_bar(bar_notes, corpus_name = "NA", piece_name = "NA", obs_idx = 0):
    dataset = pd.DataFrame(columns = ['obs', 'corpus', 'piece', 'timesig', 'n_simu_notes', 'mn', 'staff', 'simu_note', 'pitch', 'duration'])

    timesig = bar_notes.iloc[0]["timesig"]
    timesig_top, timesig_bottom = [int(x) for x in timesig.split('/')]
    
    n_simu_notes = int(timesig_top * (16 / timesig_bottom))

    for simu_note in range(n_simu_notes):
        notes_simunote = bar_notes[bar_notes["mn_onset_quantized"] == simu_note]
        notes_simunote_sorted_by_pitch = notes_simunote.sort_values("midi")
        for index, row in notes_simunote_sorted_by_pitch.iterrows():
            if (row.duration_16 == 0): continue # we ignore grace notes (approximation error)
            if (row.tied == -1.0 or row.tied == 0.0) and not simu_note == 0: # notes tied to a previous note need further consideration
                previous_notes_same_pitch = dataset[(dataset.staff == row.staff) & (dataset.pitch == row.midi)]
                if len(previous_notes_same_pitch) == 0: # happens if the note is tied to a previous grace note of duration 0 that we ignored
                    dataset.loc[len(dataset)] = {'obs': obs_idx, 'corpus': corpus_name, 'piece': piece_name, \
                                                 'timesig': timesig, 'n_simu_notes': n_simu_notes, 'mn': row.mn, \
                                                    'staff': row.staff, 'simu_note': simu_note, 'pitch': row.midi, \
                                                        'duration': row.duration_16_quantized}
                else: # normal tied note: we increase the duration of the previous note to which the note is tied
                    last_note_same_pitch_simu_note = previous_notes_same_pitch.loc[previous_notes_same_pitch['simu_note'].idxmax()]["simu_note"]
                    dataset.loc[(dataset.simu_note == last_note_same_pitch_simu_note) & (dataset.staff == row.staff) & (dataset.pitch == row.midi), "duration"] += row.duration_16_quantized
            else: # simply add to dataset
                dataset.loc[len(dataset)] = {'obs': obs_idx, 'corpus': corpus_name, 'piece': piece_name, \
                                             'timesig': timesig, 'n_simu_notes': n_simu_notes, 'mn': row.mn, \
                                                'staff': row.staff, 'simu_note': simu_note, 'pitch': row.midi, \
                                                    'duration': row.duration_16_quantized}

    return dataset


# Program
if __name__ == '__main__':
    print("Creating full dataset...")
    dataset = create_dataset()
    dataset.to_csv("data/dataset.csv", index = False)
    n_obs = len(dataset["obs"].unique())
    print("Done! Full dataset saved at data/.")
    print(f"The dataset contains {n_obs} observations.")