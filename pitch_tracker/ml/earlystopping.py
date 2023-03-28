"""
Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""
from datetime import datetime
import os
import numpy as np
import torch


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.
    The checkpoints are saved with the following format: {yyyymmdd-HHMMSS}_checkpoint_{val_loss}.pt

    Args:
        patience (int): Number of epochs to wait for improvement before terminating training. Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement. Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        dir_path (str): Directory path for saving model checkpoints. Default: './checkpoints/'
        trace_func (function): Function used for printing messages. Default: print

    Attributes:
        counter (int): Number of epochs without improvement.
        best_score (float): Best validation loss score encountered.
        early_stop (bool): Whether early stopping should be performed.
        val_loss_min (float): Minimum validation loss encountered.

        CHECKPOINT_PREFIX (str): Prefix for checkpoint filenames.
        CHECKPOINT_FORMAT (str): File format for saving checkpoints.
        CHECKPOINT_TIMESTAMP (str): Timestamp format for checkpoint filenames.

    Methods:
       __call__(self, val_loss, model):
            Update early stopping state based on current validation loss and model.

       save_checkpoint(self, val_loss, model):
            Save current model if validation loss has improved.

    """
    def __init__(self, patience:int=7, verbose:bool=False, delta:float=0, dir_path:str='./checkpoints/', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dir_path = dir_path
        self.trace_func = trace_func
        

        self.CHECKPOINT_PREFIX = 'checkpoint'
        self.CHECKPOINT_FORMAT = 'pt'
        self.CHECKPOINT_TIMESTAMP = f'%Y%m%d-%H%M%S'

    def __call__(self, val_loss:float, model):
        """Update early stopping state based on current validation loss and model.

        Args:
            val_loss (float): Current validation loss value.
            model: Model to save if validation loss has improved.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save current model if validation loss has improved.

        Args:
            val_loss (float): Current validation loss value.
            model: Model to save if validation loss has improved.

        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        os.makedirs(self.dir_path, exist_ok=True)

        time_stamp = datetime.now().strftime(self.CHECKPOINT_TIMESTAMP)
        checkpoint_name = f'{time_stamp}_{self.CHECKPOINT_PREFIX}_{val_loss:.6f}.{self.CHECKPOINT_FORMAT}'
        model_path = os.path.join(self.dir_path, checkpoint_name)
        
        torch.save(model.state_dict(), model_path)
        
        self.val_loss_min = val_loss
