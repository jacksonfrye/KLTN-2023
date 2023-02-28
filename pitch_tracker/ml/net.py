from collections import OrderedDict
from functools import partial
from typing import Tuple, Union

import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pitch_tracker.utils import dataset, files
from pitch_tracker.utils.constants import (F_MIN, HOP_LENGTH, N_CLASS, N_FFT,
                                           N_MELS, PICKING_FRAME_SIZE,
                                           PICKING_FRAME_STEP,
                                           PICKING_FRAME_TIME, SAMPLE_RATE,
                                           STEP_FRAME, STEP_TIME, WIN_LENGTH)
from pitch_tracker.utils.dataset import AudioDataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv2d_block1 = create_conv2d_block(
            conv2d_input=(1,256,3),
            maxpool_kernel_size=3,
        )
        
        self.conv2d_block2 = create_conv2d_block(
            conv2d_input=(256,256,3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block3 = create_conv2d_block(
            conv2d_input=(256,210,3),
            maxpool_kernel_size=3,
        )
        
        self.flatten_layer = torch.nn.Flatten(2)
        self.dense_layer = nn.Linear(74, 128)
        self.output_layer = nn.Linear(128, 88)
        
    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        # x = self.unflatten_layer(x)
        # x = self.reshape_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.output_layer(x)
        # x = self.softmax_layer(x)

        return x

def train_model(model, dataloader, loss_fn, optimizer, device:str):
    total_batches = len(dataloader)
    total_size = 0
    running_loss = 0
    total_correct = 0
    model.train()
    for batch, (X, (y1, y2, y3)) in enumerate(dataloader):
        X, y3 = X.to(device), y3.to(device)

        # Compute prediction error
        y_pred = model(X)
        loss = loss_fn(y_pred, y3)
        running_loss += loss.item()
        
        pos_neg_arr = (y_pred.argmax(2) == y3.argmax(2)).flatten()
        batch_target_size = pos_neg_arr.numel()
        batch_correct = torch.nonzero(pos_neg_arr).numel()
        batch_accuracy = batch_correct/batch_target_size


        total_size += batch_target_size
        total_correct += batch_correct

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            print(f"[{batch+1:>5d}/{total_batches:>5d}]  Batch Accuracy: {(100*batch_accuracy):>0.1f}%, current loss: {running_loss/(batch+1):>7f}")

    avg_loss = running_loss / total_batches
    avg_accuracy = total_correct/total_size
    print(f"[{batch+1:>5d}/{total_batches:>5d}]  Avg Accuracy: {(100*avg_accuracy):>0.1f}%, Avg loss: {avg_loss:>7f}")

def test_model(model, dataloader, loss_fn, device:str):
    num_batches = len(dataloader)
    n_size = 0
    test_loss = 0
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for X, (y1,y2,y3) in dataloader:
            X, y3 = X.to(device), y3.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y3).item()

            pos_neg_arr = (y_pred.argmax(2) == y3.argmax(2)).flatten()
            n_size += pos_neg_arr.numel()
            n_correct += torch.nonzero(pos_neg_arr).numel()
            
    test_loss /= num_batches
    acc = n_correct / n_size
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def create_conv2d_block(
        conv2d_input: Tuple[int,int,Union[Tuple[int,int], int]],
        maxpool_kernel_size: Union[Tuple[int,int], int, None],):
    in_channels, out_channels, (kernel_size) = conv2d_input
    
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
    relu = nn.ReLU()
    batch_norm = nn.BatchNorm2d(out_channels)
    maxpool_2d = nn.MaxPool2d(maxpool_kernel_size) if maxpool_kernel_size else None
    
    conv2d_block = nn.Sequential(
        OrderedDict([
            ('conv2d', conv2d),
            ('relu', relu),
            ('batch_norm', batch_norm),  
        ])
    )

    if maxpool_2d:
        conv2d_block.add_module('maxpool2d', maxpool_2d)
    
    return conv2d_block