# %%
import os
import sys

SRC_PATH = os.path.join(os.path.dirname(__file__), '../')
MEDLEYDB_PATH = os.path.join(SRC_PATH, 'medleydb')
DATASET_DIR = '../content/pickled_database/'
sys.path.extend([SRC_PATH, MEDLEYDB_PATH])

import argparse
import yaml

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pitch_tracker.ml.earlystopping import EarlyStopping
from pitch_tracker.ml.model.net import Audio_CNN
from pitch_tracker.ml.train_model import train_model
from pitch_tracker.utils import files
from pitch_tracker.utils.dataset import AudioDataset

DEVICE = "cuda" if torch.cuda.is_available() \
    else "mps" if torch.backends.mps.is_available() \
    else "cpu"
print(f"Using {DEVICE} device")


# %%
#prepare dataset
def prepare_dataset():
    # split 60/20/20
    dataset_paths = list(files.list_folder_paths_in_dir(DATASET_DIR))
    train_set, validation_set = train_test_split(
        dataset_paths,
        test_size=0.40,
        random_state=1,
        shuffle=True)
    validation_set, test_set = train_test_split(
        validation_set,
        test_size=0.50,
        random_state=1,
        shuffle=True)
    
    print(f'train_song_set: {len(train_set)}')
    print(f'validation_song_set: {len(validation_set)}')
    print(f'test_song_set: {len(test_set)}')
    
    train_dataset = AudioDataset(train_set)
    validation_dataset = AudioDataset(validation_set)
    test_dataset = AudioDataset(test_set)

    return train_dataset, validation_dataset, test_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=f'{SRC_PATH}/scripts/config/model_config.yml',
                        help='Path to the config file in .yml format')
    args = parser.parse_args()

    # Load config from file
    try:
        with open(args.config, 'r') as f:
            p = yaml.safe_load(f)
    except:
        print('Cannot find the config path, using hardcoded config')
        p = {
            # dataset
            'batch_size': 8,
            # fit
            'n_epochs': 5,
            'learning_rate': 1e-3,
            # early stopping
            'es_patience': 10,
            'es_verbose': True,
            'es_dir_path': './checkpoints',
            # lr scheduler
            'ls_patience': 8,
            'ls_factor': 0.2,
            # misc
            'device': DEVICE,
        }


    # prepare dataset & dataloader
    train_dataset, validation_dataset, test_dataset = prepare_dataset()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=p['batch_size'],
        shuffle=True
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=p['batch_size'],
        shuffle=True
    )
    
    # BCE loss doesn't work well.
    loss_fn = nn.CrossEntropyLoss().to(p['device'])
    model = Audio_CNN().to(p['device'])
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p['learning_rate']
    )
    early_stopping = EarlyStopping(
        patience=p['es_patience'],
        verbose=p['es_verbose'],
        dir_path=p['es_dir_path']
    )
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=p['ls_patience'],
        factor=p['ls_factor']
    )

    train_model(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=p['n_epochs'],
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        device=p['device'],
    )


if __name__ == '__main__':
    main()
 