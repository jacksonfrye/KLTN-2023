import mido
import pandas as pd
import numpy as np
from functools import partial


def build_note_messages(csv_path: str, ticks_per_beat: int = 480, tempo: int = 500_000, base_velocity: int = 50):
    df = pd.read_csv(csv_path, header=None)
    if len(df.columns) == 3:
        df[3] = base_velocity
    s2t = partial(mido.second2tick, ticks_per_beat=ticks_per_beat, tempo=tempo)
    note_messages = df.values
    note_messages[:, 0:2] = np.apply_along_axis(s2t, 1, note_messages[:, 0:2])
    note_messages = note_messages.astype(int)
    return note_messages


def _merge_consecutive_trues(arr: np.ndarray) -> list:
    # Find the indices where the array changes from True to False or vice versa
    change_indices = np.where(np.diff(arr))[0]
    # Add 1 to the end indices to get the correct index in the original array
    change_indices[1::2] += 1
    # If the first element is True, add a start index of 0
    if arr[0]:
        change_indices = np.hstack(([0], change_indices))
    # If the last element is True, add an end index of len(arr)-1
    if arr[-1]:
        change_indices = np.hstack((change_indices, [len(arr)-1]))
    # Reshape the change indices into pairs of start and end indices
    start_end_pairs = change_indices.reshape(-1, 2)
    # Convert the start and end indices to a list of tuples
    result = [(start, end) for start, end in start_end_pairs if start != end]

    return result


def try_merge_continuous_note_messages(note_messages, continuous_distance_threshold: int = 0):
    distance_between_messages = np.absolute(
        note_messages[:-1, 1] - note_messages[1:, 0])
    pitch_difference = note_messages[:-1, 2] - note_messages[1:, 2]
    pitch_merge_mask = pitch_difference == 0
    distance_merge_mask = distance_between_messages < continuous_distance_threshold
    should_merge_next_mask = np.bitwise_and(
        pitch_merge_mask, distance_merge_mask)
    merged = []
    i = 0
    while i < len(note_messages)-1:
        current_message = note_messages[i]
        if not should_merge_next_mask[i]:
            merged.append(current_message)
            i += 1
            continue
        current_start = current_message[0]
        current_pitch = current_message[2]
        max_velocity = current_message[3]

        # for loop check how many next messages to merge:
        j = i+1
        while j+1 < len(should_merge_next_mask):
            if not should_merge_next_mask[j+1]:
                j += 1
                break
            j += 1
        next_message = note_messages[j]
        next_end = next_message[1]
        next_velocity = next_message[3]
        max_velocity = max(max_velocity, next_velocity)
        new_message = np.array(
            [current_start, next_end, current_pitch, max_velocity], dtype=int)
        merged.append(new_message)
        i = j+1
    # append last message if it doesn't get merged
    if i == len(note_messages)-1:
        merged.append(note_messages[-1])

    return np.array(merged)


def convert_to_midi(note_messages, ticks_per_beat: int = 480):
    midi = mido.MidiFile()
    midi.ticks_per_beat = 480
    track = mido.MidiTrack()
    midi.tracks.append(track)
    last_note_end_tick = 0
    for start_tick, end_tick, midi_value, velocity in note_messages:
        last_note_tick_distance = start_tick - last_note_end_tick
        duration_tick = end_tick - start_tick

        # Create note_on and note_off messages using the values
        note_on = mido.Message('note_on', note=midi_value,
                               velocity=velocity, time=last_note_tick_distance)
        note_off = mido.Message(
            'note_off', note=midi_value, velocity=velocity, time=duration_tick)

        track.append(note_on)
        track.append(note_off)
        last_note_end_tick = end_tick

    return midi
# %%
# song_path = '/Users/tien.d/workspace/GITHUB/mono_pitch_tracker/content/gen_label/512_5_176/Melody2_midi/ClaraBerryAndWooldog_TheBadGuys.csv'
# note_messages = build_note_messages(song_path)
# merged = try_merge_continuous_note_messages(note_messages, 5)
# merged.shape,note_messages.shape
