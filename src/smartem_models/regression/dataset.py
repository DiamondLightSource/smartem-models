from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smartem_models.utils import read_img


class FoilHoleDataset(Dataset):
    def __init__(
        self, grid_squares: dict[str, Path], foil_hole_positions: dict[str, list[tuple[int, int, int]]], transform=None
    ):
        self.gs_imgs = {k: read_img(p) for k, p in grid_squares.items()}
        self.img_positions = []
        for k, v in foil_hole_positions.items():
            self.img_positions.extend((k, fh) for fh in v)

        self.transform = transform

    def __len__(self):
        return len(self.img_positions)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y, d = self.img_positions[idx][1]
        image = self.gs_imgs[self.img_positions[idx][0]][y - d // 2 : y + d // 2, x - d // 2 : x + d // 2]
        # If image is grayscale, add a channel dimension.
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        # If single channel, replicate to make 3 channels.
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        if self.transform:
            image = torch.from_numpy(image)
            image = self.transform(image)
        return image, 0
