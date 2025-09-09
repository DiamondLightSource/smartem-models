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

from smartem_models.utils import read_img, read_square_img


def atoi(text):
    return int(text) if text.isdigit() else text


def prepare_image(im: np.array) -> torch.Tensor:
    prepared_im = 2 * (im.astype(np.float32) / 255) - 1
    prepared_im = torch.from_numpy(np.expand_dims(prepared_im, axis=0))
    return prepared_im


class HoleDataset(Dataset):
    def __init__(
        self,
        grid_square_imgs: list[Path],
        foil_hole_positions: list[list[tuple[int, int]]],
        diameter,
        transform=None,
        subset_size: int | None = None,
    ):
        self.img_positions = []
        for gs in foil_hole_positions:
            self.img_positions.extend(gs)

        if subset_size is not None:
            if len(self.img_positions) > subset_size:
                chosen_indices = np.random.choice(range(len(self.img_positions)), size=subset_size)
                self.img_positions = [p for i, p in enumerate(self.img_positions) if i in chosen_indices]

        self.labels = range(len(self.img_positions))
        self.imgs = [read_square_img(p) for p in grid_square_imgs]
        self.index_map = []
        for i, gs in enumerate(foil_hole_positions):
            self.index_map.extend([i for _ in gs])
        self.transform = transform
        self.diameter = diameter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.randint(0, self.__len__() - 1)

        img_idx = self.index_map[idx]
        img_idx2 = self.index_map[idx2]

        x, y = self.img_positions[idx]
        sample = self.imgs[img_idx]
        x2, y2 = self.img_positions[idx2]
        sample2 = self.imgs[img_idx2]

        yleft = y - self.diameter // 2
        yright = y + self.diameter // 2
        xleft = x - self.diameter // 2
        xright = x + self.diameter // 2
        xleft = xleft if xleft >= 0 else 0
        xright = xright if xright >= 0 else 0
        yleft = yleft if yleft >= 0 else 0
        yright = yright if yright >= 0 else 0

        yleft2 = y2 - self.diameter // 2
        yright2 = y2 + self.diameter // 2
        xleft2 = x2 - self.diameter // 2
        xright2 = x2 + self.diameter // 2
        xleft2 = xleft2 if xleft2 >= 0 else 0
        xright2 = xright2 if xright2 >= 0 else 0
        yleft2 = yleft2 if yleft2 >= 0 else 0
        yright2 = yright2 if yright2 >= 0 else 0

        sample = sample[yleft:yright, xleft:xright].astype(np.float32)
        sample2 = sample2[yleft2:yright2, xleft2:xright2].astype(np.float32)

        if sample.shape[0] != sample.shape[1]:
            if sample.shape[0] > sample.shape[1]:
                sample = np.pad(sample, ((0, 0), (0, sample.shape[0] - sample.shape[1])))
            else:
                sample = np.pad(sample, ((0, sample.shape[1] - sample.shape[0]), (0, 0)))

        if sample2.shape[0] != sample2.shape[1]:
            if sample2.shape[0] > sample2.shape[1]:
                sample2 = np.pad(sample2, ((0, 0), (0, sample2.shape[0] - sample2.shape[1])))
            else:
                sample2 = np.pad(sample2, ((0, sample2.shape[1] - sample2.shape[0]), (0, 0)))

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


class SquareAtlasMagDataset(Dataset):
    def __init__(
        self, atlas_image_path: Path, grid_square_positions: dict[str, tuple[int, int]], size: int, transform=None
    ):
        self.atlas_img = read_img(atlas_image_path)
        self.size = size
        self.img_positions = []
        self.labels = []
        for i, (k, v) in enumerate(grid_square_positions.items()):
            self.img_positions.append((k, v))
            self.labels.append(i)

        self.transform = transform

    def __len__(self):
        return len(self.img_positions)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2 = random.randint(0, self.__len__() - 1)

        x, y = self.img_positions[idx][1]
        x2, y2 = self.img_positions[idx2][1]

        yleft = y - self.size // 2
        yright = y + self.size // 2
        xleft = x - self.size // 2
        xright = x + self.size // 2
        xleft = xleft if xleft >= 0 else 0
        xright = xright if xright >= 0 else 0
        yleft = yleft if yleft >= 0 else 0
        yright = yright if yright >= 0 else 0

        yleft2 = y2 - self.size // 2
        yright2 = y2 + self.size // 2
        xleft2 = x2 - self.size // 2
        xright2 = x2 + self.size // 2
        xleft2 = xleft2 if xleft2 >= 0 else 0
        xright2 = xright2 if xright2 >= 0 else 0
        yleft2 = yleft2 if yleft2 >= 0 else 0
        yright2 = yright2 if yright2 >= 0 else 0

        sample = self.atlas_img[yleft:yright, xleft:xright].astype(np.float32)
        sample2 = self.atlas_img[yleft2:yright2, xleft2:xright2].astype(np.float32)

        if sample.shape[0] != sample.shape[1]:
            if sample.shape[0] > sample.shape[1]:
                sample = np.pad(sample, ((0, 0), (0, sample.shape[0] - sample.shape[1])))
            else:
                sample = np.pad(sample, ((0, sample.shape[1] - sample.shape[0]), (0, 0)))

        if sample2.shape[0] != sample2.shape[1]:
            if sample2.shape[0] > sample2.shape[1]:
                sample2 = np.pad(sample2, ((0, 0), (0, sample2.shape[0] - sample2.shape[1])))
            else:
                sample2 = np.pad(sample2, ((0, sample2.shape[1] - sample2.shape[0]), (0, 0)))

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
