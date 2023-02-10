import csv
from typing import Dict

import numpy as np

from ..utils.files import get_file_name, list_file_paths_in_dir


def create_label_dict(label_folder: str) -> Dict[str, np.ndarray]:
    """Creates a dictionary that maps label names to their corresponding note messages in numpy arrays.

    Args:
        label_folder (str): The directory where the label files are located.

    Returns:
        dict: A dictionary that maps label names to their note messages as numpy arrays.
    """
    result = {}
    file_paths = list_file_paths_in_dir(label_folder)
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

        label_name = get_file_name(label_path, include_ext=False)
        result[label_name] = note_messages
    return result