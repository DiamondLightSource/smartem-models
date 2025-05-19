from pathlib import Path

import mrcfile
import numpy as np
import tifffile


def read_img(img_path: Path, normalise: bool = True) -> np.array:
    if img_path.suffix == ".mrc":
        data = mrcfile.read(img_path)
    elif img_path.suffix in (".tiff", ".tif"):
        data = tifffile.imread(img_path)
    else:
        raise ValueError(f"Input images must be in MRC or TIFF format. Format {img_path.suffix} unrecognised")
    if normalise:
        mean = np.mean(data)
        sdev = np.std(data)
        sigma_min = mean - 3 * sdev
        sigma_max = mean + 3 * sdev
        data = np.ndarray.copy(data)
        data[data < sigma_min] = sigma_min
        data[data > sigma_max] = sigma_max
        data = data - data.min()
        data = data * 255 / data.max()
        data = data.astype("uint8")
    return data
