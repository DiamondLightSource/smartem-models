from functools import lru_cache

import numpy as np
import scipy.stats


def grid_hist(grid: tuple[tuple[float, float, float]], coords: list[tuple], values: list[float]) -> np.array:
    grid_shape = tuple(int((g[1] - g[0]) / g[2]) for g in grid)
    indices = np.arange(np.prod(grid_shape))
    unravelled_indices = np.unravel_index(indices, grid_shape)
    flattened_hist = [[] for _ in indices]
    found_coord_indices = []
    for i in indices:
        for j, coord in enumerate(coords):
            if j not in found_coord_indices:
                ui = tuple(u[i] for u in unravelled_indices)
                if all(x >= grid[n][0] + ui[n] * grid[n][2] for n, x in enumerate(coord)) and all(
                    x <= grid[n][0] + (ui[n] + 1) * grid[n][2] for n, x in enumerate(coord)
                ):
                    flattened_hist[i].append(values[j])
                    found_coord_indices.append(j)
    most_hits = np.max([len(h) for h in flattened_hist])
    flattened_hist = np.array([np.pad(h, (0, most_hits - len(h)), constant_values=np.nan) for h in flattened_hist])
    return flattened_hist.reshape((*grid_shape, most_hits))


def init_distributions(hist: np.array, num_steps: int = 10) -> np.array:
    means = np.nanmean(hist, axis=-1)
    sdevs = np.nanstd(hist, axis=-1)
    # if there was only one (non-nan value in the array take the standard deviation to be 0.5
    sdevs = np.where(sdevs == 0, 0.5, sdevs)
    dist_constructor = scipy.stats.truncnorm(loc=means, scale=sdevs, a=-means / sdevs, b=(1 - means) / sdevs)
    res = np.zeros((num_steps, *means.shape))
    dist_step = 1 / num_steps
    for i in range(num_steps):
        res[i] = dist_constructor.pdf(dist_step * i + (dist_step / 2))
    return np.moveaxis(res, 0, -1)


@lru_cache(maxsize=100)
def _binary_bin_search(grid: tuple[tuple[float, float, float]], coord: tuple) -> tuple:
    bin_index = [0 for _ in coord]
    for i, (axis, x) in enumerate(zip(grid, coord, strict=False)):
        edges = np.arange(axis[0], axis[1] + axis[2], axis[2])
        indices = list(range(len(edges)))
        while len(edges) > 1:
            if x <= edges[len(edges) // 2]:
                indices = indices[: len(edges) // 2]
                edges = edges[: len(edges) // 2]
            else:
                indices = indices[len(edges) // 2 :]
                edges = edges[len(edges) // 2 :]
        bin_index[i] = indices[0]
    return tuple(bin_index)


def update_distributions(
    dist: np.array, grid: tuple[tuple[float, float, float]], coord: tuple, quality: bool
) -> np.array:
    index = _binary_bin_search(grid, coord)
    bin_dist = dist[index]
    step = 1 / len(bin_dist)
    midpoints = np.arange(step, 1 + step, step)
    probs = np.array([p if quality else 1 - p for p in midpoints])
    update_unnormalised = bin_dist * probs * step
    update = update_unnormalised / np.sum(update_unnormalised)
    dist[index] = update / step
    return dist


def score(dist: np.array, grid: tuple[tuple[float, float, float]], coord: tuple) -> float:
    index = _binary_bin_search(grid, coord)
    bin_dist = dist[index]
    step = 1 / len(bin_dist)
    midpoints = np.arange(step, 1 + step, step)
    return np.sum(step * midpoints * bin_dist)
