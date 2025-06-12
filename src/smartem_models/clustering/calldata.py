"""
Created on Sun Nov 20 14:04:56 2022

@author: jaehoon cha
@email: jaehoon.cha@stfc.ac.uk
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smartem_models.utils import read_img


def atoi(text):
    return int(text) if text.isdigit() else text


def prepare_image(im: np.array) -> torch.Tensor:
    prepared_im = 2 * (im.astype(np.float32) / 255) - 1
    prepared_im = torch.from_numpy(np.expand_dims(prepared_im, axis=0))
    return prepared_im


class SquareDataset(Dataset):
    def __init__(self, grid_squares: dict[int, Path], transform=None):
        self.labels, self.img_paths = zip(*grid_squares.items(), strict=True)
        self.imgs = [read_img(p) for p in self.img_paths]
        self.transform = transform

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
        if self.transform:
            sample = self.transform(sample)
            sample2 = self.transform(sample2)

        labs = self.labels[idx]

        labs = torch.from_numpy(np.expand_dims(labs, axis=0))

        sample = {"x1": sample, "x2": sample2, "lab": labs}

        return sample
