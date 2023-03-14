import os
import sys

SRC_PATH = os.path.join(os.path.dirname(__file__), '../')
MEDLEYDB_PATH = os.path.join(SRC_PATH, 'medleydb')
DATASET_DIR = '../content/pickled_database/'
sys.path.extend([SRC_PATH, MEDLEYDB_PATH])

import numpy as np
import pandas as pd

import medleydb
from pitch_tracker.utils.constants import (F_MIN, HOP_LENGTH, N_FFT, N_MELS,
                                           PICKING_FRAME_SIZE,
                                           PICKING_FRAME_STEP,
                                           PICKING_FRAME_TIME, SAMPLE_RATE,
                                           STEP_FRAME, STEP_TIME, WIN_LENGTH,
                                           N_CLASS, )
from pitch_tracker.utils.audio import load_audio_mono
from pitch_tracker.utils.files import get_file_name, list_file_paths_in_dir, list_folder_paths_in_dir

from pitch_tracker.utils.medleydb_melody import gen_label
from pitch_tracker.utils import dataset

# TODO: convert global variables into parameters and put them inside main()
HOP = 512*5
output_label_dir = f'{SRC_PATH}/content/gen_label/{HOP}/'
AUDIO_PATH = medleydb.AUDIO_PATH

def convert_label():
    available_tracks = list(list_folder_paths_in_dir(AUDIO_PATH))
    available_tracks = [get_file_name(full_path) for full_path in available_tracks]

    mtracks = medleydb.load_multitracks(available_tracks)

    for mtrack in mtracks:
        gen_label(mtrack.track_id, output_label_dir, hop=HOP, overwrite=True, convert_to_midi=True, round_method='round', to_csv=True)
        # gen_label(mtrack.track_id, output_dir, hop=HOP, overwrite=True, convert_to_midi=True, round_method='round', to_csv=False)
        pass

def create_pickled_data():
    label_dir = f'{SRC_PATH}/content/gen_label/{HOP}/Melody2_midi/'
    label_dict = dataset.create_label_dict_from_dir(label_dir)
    
    for k, v in label_dict.items():
        print(k, v.shape)
    
    dataset_paths = dataset.create_dataset_path_dict(label_dir)

    feature_label_gen = dataset.create_feature_label_generator(
        dataset_path_dict=dataset_paths,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        n_class=N_CLASS,
        hop_length=HOP_LENGTH,
        picking_frame_step=PICKING_FRAME_STEP,
        picking_frame_size=PICKING_FRAME_SIZE,
        step_frame=STEP_FRAME,
        step_time=STEP_TIME,
        dist_threshold=0.1,
        empty_threshold=0.3,
    )

    output_pickled_dir = f'{SRC_PATH}/content/pickled_database/'
    passed_songs = dataset.write_feature_label_to_disk_by_frame(
        feature_label_gen, output_pickled_dir, categorize_by_subdir = True)
    failed_songs = [label for label in dataset_paths if label not in passed_songs]
    print(failed_songs)
    del failed_songs

def main():
    convert_label()
    create_pickled_data()

if __name__ == '__main__':
    main()
