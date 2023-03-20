import os
import sys

SRC_PATH = os.path.join(os.path.dirname(__file__), '../')
MEDLEYDB_PATH = os.path.join(SRC_PATH, 'medleydb')
DATASET_DIR = f'{SRC_PATH}/content/pickled_database/'
sys.path.extend([SRC_PATH, MEDLEYDB_PATH])

import argparse

import numpy as np
import pandas as pd

import medleydb
from pitch_tracker.utils import dataset
from pitch_tracker.utils.audio import load_audio_mono
from pitch_tracker.utils.constants import (F_MIN, HOP_LENGTH, N_CLASS, N_FFT,
                                           N_MELS, PICKING_FRAME_SIZE,
                                           PICKING_FRAME_STEP,
                                           PICKING_FRAME_TIME, SAMPLE_RATE,
                                           STEP_FRAME, STEP_TIME, WIN_LENGTH)
from pitch_tracker.utils.files import (get_file_name, list_file_paths_in_dir,
                                       list_folder_paths_in_dir)
from pitch_tracker.utils.medleydb_melody import gen_label

# TODO: convert global variables into parameters and put them inside main()
AUDIO_PATH = medleydb.AUDIO_PATH

def convert_label(stft_hop_size, step_frame, output_label_dir:str):
    available_tracks = list(list_folder_paths_in_dir(AUDIO_PATH))
    available_tracks = [get_file_name(full_path) for full_path in available_tracks]

    mtracks = medleydb.load_multitracks(available_tracks)

    for mtrack in mtracks:
        # this `hop` prefers to the number of element between each analysis frame
        # not to be confused with STFT hop
        gen_label(mtrack.track_id, output_label_dir, hop=stft_hop_size*step_frame, overwrite=True, convert_to_midi=True, round_method='round', to_csv=True)
        # gen_label(mtrack.track_id, output_dir, hop=HOP, overwrite=True, convert_to_midi=True, round_method='round', to_csv=False)
        pass

def create_pickled_data(stft_hop_size, step_frame):
    label_dir = f'{SRC_PATH}/content/gen_label/{stft_hop_size}_{step_frame}/Melody2_midi/'
    label_dict = dataset.create_label_dict_from_dir(label_dir)
    
    for k, v in label_dict.items():
        print(k, v.shape)
    
    dataset_paths = dataset.create_dataset_path_dict(label_dir)
    step_time = step_frame * (stft_hop_size/SAMPLE_RATE)
    picking_frame_step = int(PICKING_FRAME_SIZE/2)

    feature_label_gen = dataset.create_feature_label_generator(
        dataset_path_dict=dataset_paths,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        n_class=N_CLASS,
        hop_length=stft_hop_size,
        picking_frame_step=picking_frame_step,
        picking_frame_size=PICKING_FRAME_SIZE,
        step_frame=step_frame,
        step_time=step_time,
        dist_threshold=0.1,
        empty_threshold=0.3,
    )

    output_pickled_dir = f'{SRC_PATH}/content/pickled_database/{stft_hop_size}_{step_frame}/'
    passed_songs = dataset.write_feature_label_to_disk_by_frame(
        feature_label_gen, output_pickled_dir, categorize_by_subdir = True)
    failed_songs = [label for label in dataset_paths if label not in passed_songs]
    print(failed_songs)
    del failed_songs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stft_hop_size', type=int, default=512,
                        help='STFT Hop size for label conversion and pickled data creation')
    parser.add_argument('--step_frame', type=int, default=5,
                        help='number of STFT frames to form an onset/analysis frame')
    args = parser.parse_args()

    output_label_dir = f'{SRC_PATH}/content/gen_label/{args.stft_hop_size}_{args.step_frame}/'
    convert_label(args.stft_hop_size, args.step_frame, output_label_dir)
    create_pickled_data(args.stft_hop_size, args.step_frame)

if __name__ == '__main__':
    main()
