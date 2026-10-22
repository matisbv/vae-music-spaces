# Import packages
import pandas as pd
from tqdm import tqdm
import os
import math

# Path to corpora and dataset
filepath_data = 'data/dcml_corpora/'
filepath_dataset = 'data/dataset.csv'

# Util functions
def create_metadata():
    dataset = pd.read_csv(filepath_dataset)
    metadata = pd.DataFrame(columns = ['obs', 'corpus', 'piece', 'music_era', 'century', 'year', 'style', 'n_staffs', 'n_instruments', \
                                       'key', 'mn', 'timesig', 'n_simu_notes', 'n_notes', 'n_notes_per_simu_note', 'avg_note_pitch', 'pitch_range', 
                                       'avg_note_duration', 'n_chords', 'first_chord', 'first_chord_type', 'first_chord_numeral'])

    list_corpora = sorted(os.listdir(filepath_data))

    for corpus in list_corpora:
        print(f"Processing corpus {corpus}...")

        metadata_corpus = pd.read_csv(filepath_data + '/' + corpus + '/metadata.tsv', sep = '\t')

        list_files = sorted(os.listdir(filepath_data + '/' + corpus + '/notes'))
        list_pieces = [x for x in list_files if x.endswith(".tsv")]

        for piece in tqdm(list_pieces):

            # Collect metadata on piece level
            if piece.endswith("_reviewed.tsv"):
                metadata_piece = metadata_corpus[metadata_corpus["fname"] == piece[:-13]]
            else:
                metadata_piece = metadata_corpus[metadata_corpus["fname"] == piece[:-4]]
            year = metadata_piece["composed_end"].iloc[0]
            century = determine_century(year)
            music_era = determine_music_era(year)
            key = metadata_piece["annotated_key"].iloc[0]
            n_staffs = len(dataset[dataset["piece"] == piece[:-4]]["staff"].unique())
            n_instruments = determine_n_instruments(n_staffs, metadata_piece)
            style = determine_style(corpus)

            # Collect harmonies / chords data
            if piece.endswith("_reviewed.tsv"):
                harmonies_path = filepath_data + '/' + corpus + '/harmonies/' + piece[:-13] + '.tsv'
            else:
                harmonies_path = filepath_data + '/' + corpus + '/harmonies/' + piece
            if os.path.exists(harmonies_path):
                harmonies_piece = pd.read_csv(harmonies_path, sep = '\t')

            # Collect bar data and metadata on bar level
            dataset_piece = dataset[dataset["piece"] == piece[:-4]]
            obs_id_list = sorted(dataset_piece["obs"].unique())

            for obs_id in obs_id_list:
                dataset_bar = dataset_piece[dataset_piece["obs"] == obs_id]
                mn = dataset_bar["mn"].iloc[0]

                n_notes = len(dataset_bar)
                n_notes_per_simu_note = len(dataset_bar) / dataset_bar["n_simu_notes"].iloc[0]
                avg_note_pitch = sum(dataset_bar["pitch"]) / n_notes
                pitch_range = max(dataset_bar["pitch"]) - min(dataset_bar["pitch"]) + 1
                avg_note_duration = sum(dataset_bar["duration"]) / n_notes

                if os.path.exists(harmonies_path):
                    harmonies_bar = harmonies_piece[harmonies_piece["mn"] == mn]
                    n_chords = len(harmonies_bar)
                    if n_chords == 0 and mn == 1: # sometimes the first chord is on measure 0, see beethoven n01-1
                        harmonies_bar = harmonies_piece[harmonies_piece["mn"] == 0]
                        last_chord = key + '_' + harmonies_bar["chord"].iloc[-1]
                        last_chord_type = harmonies_bar["chord_type"].iloc[-1]
                        last_chord_numeral = harmonies_bar["numeral"].iloc[-1]
                    if (n_chords == 0) or (str(harmonies_bar["chord"].iloc[0]) == 'nan'): # we use the last recorded chord as the bar's chord
                        n_chords = 1
                        first_chord = last_chord
                        first_chord_type = last_chord_type
                        first_chord_numeral = last_chord_numeral
                    else: # we use the first chord appearing in the bar
                        first_chord = key + '_' + harmonies_bar["chord"].iloc[0]
                        first_chord_type = harmonies_bar["chord_type"].iloc[0]
                        first_chord_numeral = harmonies_bar["numeral"].iloc[0]
                        if str(harmonies_bar["chord"].iloc[-1]) == 'nan':
                            last_chord = first_chord
                            last_chord_type = first_chord_type
                            last_chord_numeral = first_chord_numeral
                        else:
                            last_chord = key + '_' + harmonies_bar["chord"].iloc[-1]
                            last_chord_type = harmonies_bar["chord_type"].iloc[-1]
                            last_chord_numeral = harmonies_bar["numeral"].iloc[-1]
                else:
                    n_chords = 99
                    first_chord = 'UNK'
                    first_chord_type = 'UNK'
                    first_chord_numeral = 'UNK'

                timesig = dataset_bar["timesig"].iloc[0]
                n_simu_notes = dataset_bar["n_simu_notes"].iloc[0]

                metadata.loc[len(metadata)] = {'obs': obs_id, 'corpus': corpus, 'piece': piece, 'music_era': music_era, \
                                               'century': century, 'year': year, 'style': style, 'n_staffs': n_staffs, \
                                               'n_instruments': n_instruments, 'key': key, 'mn': mn, 'timesig': timesig, \
                                                'n_simu_notes': n_simu_notes, 'n_notes': n_notes, 'n_notes_per_simu_note': n_notes_per_simu_note, \
                                                'avg_note_pitch': avg_note_pitch, 'pitch_range': pitch_range, 'avg_note_duration': avg_note_duration, 'n_chords': n_chords, \
                                                'first_chord': first_chord, 'first_chord_type': first_chord_type, 'first_chord_numeral': first_chord_numeral}
    
    return metadata

def determine_music_era(year):
    if year < 1600:
        era =  "1400-1600 : Renaissance"
    elif year < 1750:
        era =  "1600-1750 : Baroque"
    elif year < 1820:
        era = "1750-1820 : Classical"
    elif year < 1900:
        era = "1820-1900 : Romantic"
    else:
        era = "1900-1975 : Modernism"
    return era

def determine_century(year):
    if year < 1600:
        century =  "< 16th"
    elif year < 1700:
        century =  "17th"
    elif year < 1800:
        century = "18th"
    elif year < 1900:
        century = "19th"
    else:
        century = "20th"
    return century

def determine_n_instruments(n_staffs, metadata_piece):
    if n_staffs == 2:
        instr1 = str(metadata_piece["staff_1_instrument"].iloc[0])
        instr2 = str(metadata_piece["staff_2_instrument"].iloc[0])
        if instr1 == instr2:
            n_instruments = 1

    if n_staffs == 3:
        instr1 = str(metadata_piece["staff_1_instrument"].iloc[0])
        instr2 = str(metadata_piece["staff_2_instrument"].iloc[0])
        instr3 = str(metadata_piece["staff_3_instrument"].iloc[0])
        if (instr1.startswith("piano") or instr1.startswith("Piano")) and (instr2.startswith("piano") or instr2.startswith("Piano")) \
            and (instr3.startswith("piano") or instr3.startswith("Piano")): # see Grieg op43n06, Liszt op161no07, Medtner op34n03
            n_instruments = 1
        else:
            n_instruments = 3

    if n_staffs == 4:
        instr1 = str(metadata_piece["staff_1_instrument"].iloc[0])
        instr2 = str(metadata_piece["staff_2_instrument"].iloc[0])
        instr3 = str(metadata_piece["staff_3_instrument"].iloc[0])
        instr4 = str(metadata_piece["staff_4_instrument"].iloc[0])
        if (instr1.startswith("piano") or instr1.startswith("Piano")) and (instr2.startswith("piano") or instr2.startswith("Piano")) \
            and (instr3.startswith("piano") or instr3.startswith("Piano")) and (instr4.startswith("piano") or instr4.startswith("Piano")): # see Liszt op161n04
            n_instruments = 1
        else: # see corelli, Beethoven string quartets
            n_instruments = 4
    
    return n_instruments

def determine_style(corpus):
    if corpus == "ABC":
        style = "String quartet"
    elif corpus == "corelli":
        style = "Trio sonata"
    elif corpus == "chopin_mazurkas":
        style = "Mazurka"
    else:
        style = "Piano piece"

    return style


# Program
if __name__ == '__main__':
    print("Creating metadata table...")
    metadata = create_metadata()
    metadata.to_csv("data/metadata.csv", index = False)
    print("Done! Metadata saved at data/.")