import numpy as np
import scipy.stats


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


def update_distribution(dist: np.array, quality: bool) -> np.array:
    step = 1 / len(dist)
    midpoints = np.arange(step, 1 + step, step)
    probs = np.array([p if quality else 1 - p for p in midpoints])
    update_unnormalised = dist * probs * step
    update = update_unnormalised / np.sum(update_unnormalised)
    dist = update / step
    return dist


def update_distribution_from_prob(dist: np.array, quality: float) -> np.array:
    step = 1 / len(dist)
    midpoints = np.arange(step, 1 + step, step)
    probs = np.array([p * quality + (1 - p) * (1 - quality) for p in midpoints])
    probs[probs < 1e-4] = 1e-4
    update_unnormalised = dist * probs * step
    update = update_unnormalised / np.sum(update_unnormalised)
    dist = update / step
    return dist


def score(dist: np.array, index: int) -> float:
    bin_dist = dist[index]
    step = 1 / len(bin_dist)
    midpoints = np.arange(step, 1 + step, step)
    return np.sum(step * midpoints * bin_dist)
