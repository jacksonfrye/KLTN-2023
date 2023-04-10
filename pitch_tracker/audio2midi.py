import os
import sys

from pitch_tracker.classifier import MelodyExtractor
from pitch_tracker import THESIS_2023_MODEL_PATH

SRC_PATH = os.path.join(os.path.dirname(__file__))
MEDLEYDB_PATH = os.path.join(SRC_PATH, 'medleydb')
DATASET_DIR = f'{SRC_PATH}/content/pickled_database/'
sys.path.extend([SRC_PATH, MEDLEYDB_PATH])

import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert audio files to MIDI files.')
    parser.add_argument('src', type=str, help='Path to directory of audio files or list of audio file paths')
    parser.add_argument('--out_dir', type=str, default='./', help='Output directory for MIDI files')
    parser.add_argument('--vb', type=float, default=0.0, help='Voicing bias for melody extraction')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on')
    parser.add_argument('--model_path', type=str, default=THESIS_2023_MODEL_PATH, help='Path to model')

    args = parser.parse_args()

    melody_extractor = MelodyExtractor(model_path=args.model_path, device=args.device)
    melody_extractor.export_to_midis(args.src, args.out_dir, voicing_bias=args.vb)

if __name__ == '__main__':
    main()
