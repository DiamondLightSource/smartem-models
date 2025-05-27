import numpy as np
import scipy.stats


def grid_hist(grid: tuple[tuple[float, float, float]], coords: list[tuple], values: list[float]) -> np.array:
    if not all((g[1] - g[0]) % g[2] == 0 for g in grid):
        raise ValueError
    grid_shape = (int((g[1] - g[0]) / g[2]) for g in grid)
    indices = np.arange(np.prod(grid_shape))
    unravelled_indices = np.unravel_index(indices, grid_shape)
    flattened_hist = [[] for _ in indices]
    found_coord_indices = []
    for i in indices:
        for j, coord in enumerate(coords):
            if j not in found_coord_indices:
                ui = unravelled_indices[i]
                if all(x >= grid[n][0] + ui[n] * grid[n][2] for n, x in enumerate(coord)) and all(
                    x <= grid[n][0] + (ui[n] + 1) * grid[n][2] for n, x in enumerate(coord)
                ):
                    flattened_hist[i].append(values[j])
                    found_coord_indices.append(j)
    most_hits = np.max([len(h) for h in flattened_hist])
    flattened_hist = np.array([np.pad(h, (0, most_hits - len(h)), constant_values=np.nan) for h in flattened_hist])
    return flattened_hist.reshape((*grid_shape, most_hits))


def init_distributions(grid: np.array, num_steps: int = 10) -> np.array:
    means = np.nanmean(grid, axis=-1)
    sdevs = np.nanstd(grid, axis=-1)
    # if there was only one (non-nan value in the array take the standard deviation to be 0.5
    sdevs = np.where(sdevs == 0, 0.5, sdevs)
    dist_constructor = scipy.stats.truncnorm(loc=means, scale=sdevs, a=-means / sdevs, b=1 - (means / sdevs))
    res = np.zeros((num_steps, *means.shape))
    dist_step = 1 / num_steps
    for i in range(num_steps):
        res[i] = dist_constructor(dist_step * i + (dist_step / 2))
    return np.moveaxis(res, 0, -1)
