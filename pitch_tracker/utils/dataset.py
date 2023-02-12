import csv
from typing import Dict, Any, Union


import numpy as np
import librosa
import torch
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import amplitude_to_DB


from pitch_tracker.utils import files
from pitch_tracker.utils.audio import load_audio_mono


def create_label_dict(label_folder: str) -> Dict[str, np.ndarray]:
    """Creates a dictionary that maps label names to their corresponding note messages in numpy arrays.

    Args:
        label_folder (str): The directory where the label files are located.

    Returns:
        dict: A dictionary that maps label names to their note messages as numpy arrays.
    """
    result = {}
    file_paths = files.list_file_paths_in_dir(label_folder)
    for label_path in file_paths:
        with open(label_path, 'r') as f:
            csv_reader = csv.reader(f)
            note_messages = []
            for line in csv_reader:
                stime = float(line[0])
                etime = float(line[1])
                pitch = float(line[2])
                note_messages.append((stime, etime, pitch))
        note_messages = np.asarray(note_messages, dtype=np.float32)

        label_name = files.get_file_name(label_path, include_ext=False)
        result[label_name] = note_messages
    return result


def build_pick_features_and_time(
    STFT_features: np.ndarray,
    picking_frame_step:int,
    picking_frame_size:int,
    step_frame:int,
    step_time:int):

    n_frames = STFT_features.shape[0]
    n_win_frames = int(np.ceil(n_frames / step_frame / picking_frame_step))

    times_1D = np.arange(0, n_win_frames * picking_frame_size) * step_time
    pick_features = np.zeros((n_win_frames, picking_frame_size * step_frame, STFT_features.shape[1]))
    pick_times = np.zeros((n_win_frames, picking_frame_size))

    for i, j in enumerate(range(0, n_frames, picking_frame_step*step_frame)):
        pick_times[i, :] = times_1D[int(j/step_frame):int(j/step_frame)+picking_frame_size]
        end_idx = j + picking_frame_size * step_frame
        end_idx = end_idx if end_idx < n_frames else n_frames
        copy_size = end_idx - j
        pick_features[i, :copy_size, :] = STFT_features[range(j, end_idx), :]

    return pick_features, pick_times



def create_label_generator(note_messages: np.ndarray, ftimes, dist_threshold, empty_threshold, step_time):
    for t_frame in ftimes:
        stime = t_frame[0] - step_time
        etime = t_frame[-1] + step_time
        note_messages_in_frame = _get_note_messages_in_frame(note_messages, stime, etime)
        onset_labels = _get_onset_label(note_messages_in_frame, t_frame)
        duration_labels = _get_duration_label(t_frame,
                                              note_messages_in_frame,
                                              onset_labels,
                                              dist_threshold,
                                              empty_threshold)
        pitch_labels = _get_pitch_label(t_frame, note_messages_in_frame)

        yield [onset_labels, duration_labels, pitch_labels]

def extract_stft_feature(
    y:np.ndarray,
    n_fft:int,
    hop_length:int,
    mean:float=0.0,
    var:float=1.0) -> np.ndarray:
    
    stft_feature = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    log_compress = librosa.core.power_to_db(np.square(np.abs(stft_feature))).T
    return (log_compress - mean) / var

def extract_melspectrogram_feature(
    y:np.ndarray,
    n_fft:int,
    hop_length:int,
    n_mels:int,
    sample_rate:int=44100,
    win_length:int = None,
    fmin:float=0.0,
    mean:float=0.0,
    var:float=1.0,
    backend:str='librosa') -> Union[np.ndarray, torch.Tensor]:
    """extract the melspectrogram feature from the audio signal.

    Args:
        y (np.ndarray): the audio signal, shape: [n_channels, n_samples] or [n_samples]
        n_fft (int): the length of the FFT window.
        hop_length (int): the number of samples between successive frame.
        n_mels (int): the number of mel bands.
        sample_rate (int, optional): the sample rate of the audio signal. Defaults to 44100.
        win_length (int, optional): the length of the windows function, not to be confused with n_fft. Defaults to None.
        fmin (float, optional): the lowest bound of the frequency. Defaults to 0.0.
        mean (float, optional): the mean of the dataset, used for normalizing. Defaults to 0.0.
        var (float, optional): the var of the dataset, used for normalizing. Defaults to 1.0.
        backend (str, optional): either `librosa` or `torchaudio`. Defaults to 'librosa'.

    Returns:
        np.ndarray | torch.Tensor: the melspectrogram feature, shape: [n_channels, n_mels, n_frames] 
    """
    if backend == 'torch':
        melspectrogram_extractor = MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            pad_mode='constant',
            center=True,
            f_min=librosa.midi_to_hz(21),
            f_max=None,
            n_mels=n_mels,
            power=2.0,
            norm="slaney",
        )
        melspectrogram_feature = melspectrogram_extractor(y)
        # TODO: use torchaudio.function.amplitude_to_DB to convert the mel-feature to DB scale.
        # replace this:
        # melspectrogram_feature = melspectrogram_feature.numpy()
        # with
        log_compress = amplitude_to_DB(melspectrogram_feature, multiplier=10., amin=1e-10, db_multiplier=1.0)
    
    else:    
        melspectrogram_feature = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            pad_mode='constant',
            center=True,
            fmin=fmin,
            fmax=None,
            n_mels=n_mels,
            power=2.0,
            htk=True)
        log_compress = librosa.power_to_db(melspectrogram_feature)
    return (log_compress - mean) / var

def create_feature_label_generator(
    audio_dir:str, 
    label_dir:str, 
    sample_rate:int, 
    n_fft:int,
    hop_length:int,
    picking_frame_step:int,
    picking_frame_size:int,
    step_frame:int,
    step_time:int,
    dist_threshold:float,
    empty_threshold:float):

    label_dict = create_label_dict(label_dir)

    for label_name in label_dict:
        audio_path = files.get_full_path_by_file_name(audio_dir, label_name)
        if audio_path is None:
            print(f'No audio:{label_name}')
            continue

        end_time_label = label_dict[label_name][-1, 1] - 0.1
        # logger.info(f'{label_name} end: {end_time_label}')
        
        y, sr = load_audio_mono(audio_path, sample_rate)
        eindex = int(end_time_label * sr)
        
        STFT_features = extract_stft_feature(y[:eindex], n_fft=n_fft, hop_length=hop_length)

        pick_features, pick_times = build_pick_features_and_time(STFT_features,
                                                                 picking_frame_step,
                                                                 picking_frame_size,
                                                                 step_frame,
                                                                 step_time)

        label_generator = create_label_generator(label_dict[label_name],
                                                 pick_times,
                                                 dist_threshold=dist_threshold,
                                                 empty_threshold=empty_threshold)

        feature_generator = (feature for feature in pick_features)
        
        # feature_and_label = _intergen(feature_generator, label_generator)
        feature_and_label = ((feature, label) for feature, label in zip(feature_generator, label_generator))
        
        yield label_name, feature_and_label

# def _intergen():
#     # yeild x, y for 2 generator, in which x, y are list

#     pass

def _get_note_messages_in_frame(note_messages, stime, etime):
    stimes = note_messages[:, 0]
    note_messages_in_frame_idices = np.flatnonzero((stimes >= stime) & (stimes <= etime))
    notes_in_frame = note_messages[note_messages_in_frame_idices, :]
    return notes_in_frame

def _get_onset_label(
    notes_in_frame,
    t_frame,
    picking_frame_size,
    onset_time_threshold):
    
    onset_labels = np.zeros(picking_frame_size)
    for of in notes_in_frame[:, 0]:
        diff = np.abs(of - t_frame)
        onset_idx = diff < onset_time_threshold
        onset_labels[onset_idx] = 1

    return onset_labels

def _get_duration_label(t_frame, notes_in_frame, onset_labels, dist_threshold, empty_threshold, picking_frame_size):
    duration_labels = np.zeros(picking_frame_size)
    for i, t in enumerate(t_frame):
        if onset_labels[i]:
            duration_labels[i] = 1
            continue
        stime = t + dist_threshold
        etime = t - dist_threshold
        is_in_notes = np.flatnonzero(
            (notes_in_frame[:, 0] <= stime) & (notes_in_frame[:, 1] >= etime))
        if is_in_notes.size > 0:
            duration_labels[i] = 1

        num_non_silence = np.flatnonzero(duration_labels).size
        if num_non_silence < empty_threshold:
            # Has too many silence frames so we keep (skip?) this super frame
            continue

    return duration_labels

def _get_pitch_label(t_frame, notes_in_frame, picking_frame_size, n_class, pre_midi_start):
    pitch_labels = np.zeros((picking_frame_size, n_class))
    for i, t in enumerate(t_frame):
        idx = np.flatnonzero(
                (notes_in_frame[:, 0] <= t) & (notes_in_frame[:, 1] > t))
        if (idx.size > 0):
            last_label_idx = idx[-1]
            NoteLvl = int(
                    notes_in_frame[last_label_idx, 2] - pre_midi_start)
            pitch_labels[i, NoteLvl] = 1
        else:
            pitch_labels[i, 0] = 1
    return pitch_labels
