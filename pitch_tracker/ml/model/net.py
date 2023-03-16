from collections import OrderedDict
from functools import partial
from typing import Tuple, Union
from math import floor

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


class Audio_CNN(nn.Module):
    def __init__(self):
        super(Audio_CNN, self).__init__()
        self.conv2d_block1 = create_conv2d_block(
            conv2d_input=(1, 256, 3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block2 = create_conv2d_block(
            conv2d_input=(256, 256, 3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block3 = create_conv2d_block(
            conv2d_input=(256, 210, 3),
            maxpool_kernel_size=3,
        )

        self.flatten_layer = torch.nn.Flatten(2)
        self.dense_layer = nn.Linear(74, 128)
        self.output_layer = nn.Linear(128, 88)

    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.output_layer(x)

        return x


class Audio_CRNN(nn.Module):
    def __init__(self):
        super(Audio_CRNN, self).__init__()
        self.conv2d_block1 = create_conv2d_block(
            conv2d_input=(1, 256, 3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block2 = create_conv2d_block(
            conv2d_input=(256, 256, 3),
            maxpool_kernel_size=3,
        )

        self.conv2d_block3 = create_conv2d_block(
            conv2d_input=(256, 210, 3),
            maxpool_kernel_size=3,
        )

        self.flatten_layer = torch.nn.Flatten(2)

        self.gru = nn.GRU(
            input_size=74,
            hidden_size=64,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.dense_layer = nn.LazyLinear(128)
        self.output_layer = nn.LazyLinear(88)

    def forward(self, x):
        x = self.conv2d_block1(x)
        x = self.conv2d_block2(x)
        x = self.conv2d_block3(x)
        x = self.flatten_layer(x)
        x, h_n = self.gru(x)
        
        x = self.dense_layer(x)
        x = self.output_layer(x)

        return x


def create_conv2d_block(
        conv2d_input: Tuple[int, int,Union[Tuple[int,int], int]],
        padding: Union[Tuple[int,int], int, str] = 0,
        maxpool_kernel_size: Union[Tuple[int, int], int, None] = None,
        ) -> nn.Sequential:
    """
    Creates a 2D convolutional block with ReLU activation and batch normalization.

    Args:
        conv2d_input (tuple): A tuple containing the number of input channels,
            the number of output channels and the kernel size for the 2D convolutional layer.
            The kernel size can be an integer or a tuple of two integers.
        padding (int or tuple or str): The padding value. Can take 'valid'
            (similar to no padding/0) or 'same'. Default to 0.
        maxpool_kernel_size (int or tuple or None): The size of the window to take a max over for
            the MaxPool2d layer. Can be an integer or a tuple of two integers. If None,
            no MaxPool2d layer is added to the block. Default to None.

    Returns:
        nn.Sequential: A sequential container that holds all layers in the block.
    """
    in_channels, out_channels, (kernel_size) = conv2d_input

    conv2d = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding)
    relu = nn.ReLU()
    batch_norm = nn.BatchNorm2d(out_channels)
    maxpool_2d = nn.MaxPool2d(
        maxpool_kernel_size) if maxpool_kernel_size else None

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


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int,int]] = 1,
    stride: Union[int, Tuple[int,int]] = 1,
    pad: Union[int, Tuple[int,int]] = 0,
    dilation: Union[int, Tuple[int,int]] = 1,
    ) -> Tuple[int, int]:
    """
    Calculates the output shape of a 2D convolutional layer.

    Args:
        h_w (tuple): A tuple containing the height and width of the input tensor.
        kernel_size (int or tuple): The size of the kernel for the 2D convolutional layer.
            Default is 1.
        stride (int or tuple): The stride for the 2D convolutional layer. Default is 1.
        pad (int or tuple): The padding for the 2D convolutional layer. Default is 0.
        dilation (int or tuple): The dilation for the 2D convolutional layer. Default is 1.

    Returns:
        tuple: A tuple containing the height and width of the output tensor.
    """
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(
            stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = floor((h_w[0] + sum(pad[0]) - dilation[0]* \
              (kernel_size[0]-1) - 1) / stride[0] + 1)
    w = floor((h_w[1] + sum(pad[1]) - dilation[1]* \
              (kernel_size[1]-1) - 1) / stride[1] + 1)

    return h, w
