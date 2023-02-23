import csv
import functools
import operator
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple


def get_file_name(file_path:str, include_ext:bool=False):
    _, file_name_with_ext = os.path.split(file_path)
    if include_ext:
        return file_name_with_ext
    file_name, _ = os.path.splitext(file_name_with_ext)
    return file_name

def list_file_paths_in_dir(dir_path:str, include_ext=None):
    file_paths = (os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path))
    file_paths = (file_path for file_path in file_paths if os.path.isfile(file_path))
    return file_paths

def list_folder_paths_in_dir(dir_path:str):
    folder_paths = (os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path))
    folder_paths = (folder_path for folder_path in folder_paths if os.path.isdir(folder_path))
    return folder_paths

def list_all_file_paths_in_dir(dir_path:str, skip_level:int=0, exts:Tuple[str]=('')):
    list_file_paths = []
    path_generator = os.walk(dir_path)
    for _ in range(skip_level):
        next(path_generator)
    for root, ds, fs in path_generator:
        list_file_paths.extend([os.path.join(root, f) for f in fs if f.endswith(exts)])
    
    return list_file_paths

def flatten_list(a:list):
    return functools.reduce(operator.concat, a)

def save_pickle(obj, file_path, dir_path=None, is_overwrite=True):
    """Save the object to a pickled file, currently using pickle protocol 5

    Parameters
    ----------
    obj: Python object
        the object to save to file

    file_path : str
        the path to the file (with filename) to save this obj to

    dirpath: str
        the path to the directory to save this obj to, the file name of the obj is `obj_ddmmyyyy_hhmmss.pkl`
        with ddmmyyyy_hhmmss is the current datetime when saving the obj

    is_overwrite: bool
        whether to overwrite the obj to this file_path, if exists
        if not overwrite the file, the file name of the obj to save is `obj_ddmmyyyy_hhmmss.pkl`
        with ddmmyyyy_hhmmss is the current datetime when saving the obj

    Returns
    -------
    pathlib.Path
        the Path instance of the saved object, None if cannot save
    """
    if obj is None:
        return None
    if not file_path:
        if not dir_path:
            return None
        elif not isinstance(dir_path, str):
            raise TypeError(f'dir_path is a str, not {type(dir_path).__name__}')
    elif not isinstance(file_path, str):
        raise TypeError(f'file_path is a str, not {type(file_path).__name__}')

    protocol = pickle.HIGHEST_PROTOCOL
    obj_file = Path(file_path)

    os.makedirs(obj_file.parent.as_posix(),exist_ok=True)

    if not obj_file.exists():
        with obj_file.open(mode='wb') as f:
            pickle.dump(obj, f, protocol=protocol)
        return obj_file
    
    if obj_file.exists() and is_overwrite:
        with obj_file.open(mode='wb') as f:
            pickle.dump(obj, f, protocol=protocol)
        return obj_file
    else:
        print(f'{obj_file.name} is already existed.')
        # obj_filename = obj_file.stem + '_' + datetime.now().strftime('%d%m%Y_%H%M%S') + \
        #     obj_file.suffix
        # modified_obj_file = obj_file.with_name(obj_filename)
        # with modified_obj_file.open(mode='wb') as f:
        #     pickle.dump(obj, f, protocol=protocol)
        # return modified_obj_file
    
    # # file_path not exists, get the valid parent directory,
    # # or check if file_path is a directory already or use the dir_path
    # obj_dir = obj_file if obj_file.is_dir() else (
    #     obj_file.parent if obj_file.parent.is_dir() else Path(dir_path)
    # )
    # # get the name of file_path to be used as filename
    # obj_filename = obj_file.name if not obj_file.is_dir() else ''
    # if obj_dir.is_dir():
    #     obj_filename = ('obj_' + datetime.now().strftime('%d%m%Y_%H%M%S') +
    #                     '.pkl' if not obj_filename else obj_filename)
    #     obj_file = obj_dir.joinpath(obj_filename)
    #     with obj_file.open(mode='wb') as f:
    #         pickle.dump(obj, f, protocol=protocol)
    #     return obj_file
    return None

def load_pickle(file_path):
    """Save an object from a pickled file

    Parameters
    ----------
    file_path : str
        the path to the file to load this obj to

    Returns
    -------
    object or None
        the loaded object or None if cannot load
    """
    if file_path is None:
        return None
    elif not isinstance(file_path, str):
        raise TypeError(f'file_path is a str, not {type(file_path).__name__}')

    obj_file = Path(file_path)

    if obj_file.is_file():
        # try:
        with obj_file.open(mode='rb') as f:
            obj = pickle.load(f)
            return obj
        # except ValueError:
        #     import pickle5
        #     with obj_file.open(mode='rb') as f:
        #         obj = pickle5.load(f)
        #         return obj
    return None