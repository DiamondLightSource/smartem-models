"""
Created on Sun Nov 20 14:04:56 2022

@author: jaehoon cha
@email: jaehoon.cha@stfc.ac.uk
"""

import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smartem_models.utils import read_img


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


class To1DTensor:
    def __call__(self, sample):
        return torch.from_numpy(sample)


class smartem(Dataset):
    def __init__(self, path_dir, data_name, grid_id=None):
        self.data_path = os.path.join(path_dir, f"{data_name}.npz")
        self.data_zip = np.load(self.data_path)
        self.data = self.data_zip["imgs"]  # -1~1
        self.labs = self.data_zip["labs"]

        if grid_id is not None:
            idxs = np.where(self.labs == grid_id)[0]
            self.data = self.data[idxs]
            self.labs = self.labs[idxs]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.randint(0, self.__len__() - 1)

        sample = self.data[idx].astype(np.float32)
        sample2 = self.data[idx2].astype(np.float32)

        sample = 2.0 * (sample / 255.0) - 1.0
        sample2 = 2.0 * (sample2 / 255.0) - 1.0

        sample = torch.from_numpy(np.expand_dims(sample, axis=0))
        sample2 = torch.from_numpy(np.expand_dims(sample2, axis=0))

        labs = self.labs[idx]

        labs = torch.from_numpy(np.expand_dims(labs, axis=0))

        sample = {"x1": sample, "x2": sample2, "lab": labs}

        return sample


class SquareDataset(Dataset):
    def __init__(self, grid_squares: dict[int, Path]):
        self.labels, self.img_paths = zip(*grid_squares.items(), strict=True)
        self.imgs = [read_img(p) for p in self.img_paths]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.randint(0, self.__len__() - 1)

        sample = self.imgs[idx].astype(np.float32)
        sample2 = self.imgs[idx2].astype(np.float32)

        sample = 2.0 * (sample / 255.0) - 1.0
        sample2 = 2.0 * (sample2 / 255.0) - 1.0

        sample = torch.from_numpy(np.expand_dims(sample, axis=0))
        sample2 = torch.from_numpy(np.expand_dims(sample2, axis=0))

        labs = self.labels[idx]

        labs = torch.from_numpy(np.expand_dims(labs, axis=0))

        sample = {"x1": sample, "x2": sample2, "lab": labs}

        return sample
