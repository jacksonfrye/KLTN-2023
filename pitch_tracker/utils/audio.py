"""
Copyright (c) 2012-2014 Department of Computational Perception,
Johannes Kepler University, Linz, Austria and Austrian Research Institute for
Artificial Intelligence (OFAI), Vienna, Austria.
All rights reserved.

https://github.com/CPJKU/madmom/blob/main/madmom/audio/signal.py#L169

"""

import torchaudio
import torch



def load_audio_mono(file_path, sample_rate):
    """Load the audio file with target sample rate, down mix to mono.

    Args:
        file_path (str): the audio file path
        sample_rate (int): target sample rate

    Return:
        torch.Tensor: Audio array
        int: sample rate 
    """    
    signal, sr = torchaudio.load(file_path, channels_first=False)
    # resampling
    if sr != sample_rate:
        signal = resample(signal, sr, sample_rate)
        sr = sample_rate
    # downmix
    if signal.shape[1] > 1:
        signal = remix_torch(signal, 1)
    return signal, sr

def resample(signal, old_sr, new_sr):
    resampler = torchaudio.transforms.Resample(old_sr, new_sr)
    signal = resampler(signal).mean(dim=0)
    return signal


def remix_torch(signal, num_channels):
    """
    This modified for torch, original source: https://github.com/CPJKU/madmom/blob/main/madmom/audio/signal.py#L169

    Remix the signal to have the desired number of channels.

    Parameters
    ----------
    signal : numpy array
        Signal to be remixed.
    num_channels : int
        Number of channels.

    Returns
    -------
    numpy array
        Remixed signal (same dtype as `signal`).

    Notes
    -----
    This function does not support arbitrary channel number conversions.
    Only down-mixing to and up-mixing from mono signals is supported.

    The signal is returned with the same dtype, thus rounding errors may occur
    with integer dtypes.

    If the signal should be down-mixed to mono and has an integer dtype, it
    will be converted to float internally and then back to the original dtype
    to prevent clipping of the signal. To avoid this double conversion,
    convert the dtype first.

    """
    # convert to the desired number of channels
    if num_channels == signal.ndim or num_channels is None:
        # return as many channels as there are.
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        # down-mix to mono
        # Note: to prevent clipping, the signal is converted to float first
        #       and then converted back to the original dtype
        # TODO: add weighted mixing
        return torch.mean(signal, axis=-1).type(signal.dtype)
    elif num_channels > 1 and signal.ndim == 1:
        # up-mix a mono signal simply by copying channels
        return torch.tile(signal[:, None], num_channels)
    else:
        # any other channel conversion is not supported
        raise NotImplementedError("Requested %d channels, but got %d channels "
                                  "and channel conversion is not implemented."
                                  % (num_channels, signal.shape[1]))
