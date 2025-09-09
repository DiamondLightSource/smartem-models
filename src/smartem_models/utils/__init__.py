import os
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
import tifffile
import yaml


@lru_cache(maxsize=1)
def get_config():
    config_path = os.getenv("SMARTEM_MODELS_CONFIGURATION")
    if not config_path:
        raise ValueError("Failed to load configuration from environment variable SMARTEM_MODELS_CONFIGURATION")
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    return config


def read_img(img_path: Path, normalise: bool = True, crop: tuple[int] | None = None) -> np.array:
    if img_path.suffix == ".mrc":
        data = mrcfile.read(img_path)
    elif img_path.suffix in (".tiff", ".tif"):
        data = tifffile.imread(img_path)
    else:
        raise ValueError(f"Input images must be in MRC or TIFF format. Format {img_path.suffix} unrecognised")
    if not crop:
        crop = (np.min(data.shape), np.min(data.shape))
    left = (data.shape[0] - crop[0]) // 2
    top = (data.shape[1] - crop[1]) // 2
    right = left + crop[0]
    bottom = top + crop[1]
    data = data[left:right, top:bottom]
    if normalise:
        mean = np.mean(data)
        sdev = np.std(data)
        sigma_min = mean - 3 * sdev
        sigma_max = mean + 3 * sdev
        data = np.ndarray.copy(data)
        data[data < sigma_min] = sigma_min
        data[data > sigma_max] = sigma_max
        data = data - data.min()
        data = data / data.max()
        data = data * 255
        data = data.astype("uint8")
    return data


def read_square_img(img_path: Path) -> np.array:
    if img_path.suffix == ".mrc":
        data = mrcfile.read(img_path)
    elif img_path.suffix in (".tiff", ".tif"):
        data = tifffile.imread(img_path)
    else:
        raise ValueError(f"Input images must be in MRC or TIFF format. Format {img_path.suffix} unrecognised")
    return data
