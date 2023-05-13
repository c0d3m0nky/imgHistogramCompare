import json
import pickle
from pathlib import Path
from typing import Union
import glob
import os


def dump_json(obj, path: Union[str | Path]):
    with open(path, "w") as f:
        f.truncate()
        f.write(json.dumps(obj, indent=2))


def pickle_obj(obj, path: Union[str | Path]):
    with open(path, "wb") as f:
        f.truncate()
        pickle.dump(obj, f)


def unpickle_obj(path: Union[str | Path]):
    with open(path, "rb") as f:
        return pickle.load(f)



def glob_images(root: str):
    return glob.glob(root + '/*.png') + glob.glob(root + '/*.jpg') + glob.glob(root + '/*.jpeg')


def set_normal_priority():
    v = os.nice(0)
    os.nice(v * -1)
