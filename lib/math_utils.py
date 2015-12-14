import numpy as np


def merge_where_nan(target, filler):
    """
    Mutates `target` by replacing its nan values by values from `filler`.
    This is a very simple polyfill for `numpy.copyto`. (for numpy<1.7)
    """
    try:
        np.copyto(target, filler, where=np.isnan(target))
    except AttributeError:
        isnan = np.isnan(target)
        target[isnan] = filler[isnan]


def median_clip(data, clip_sigma=3., limit_ratio=1e-3, max_iterations=5):
    """
    Computes an iteratively sigma-clipped median on a `data` set.

    data : ndarray
        Input data.
    clip_sigma : float
        Sigma at which to clip.
    limit_ratio : float
        When the proportion of rejected pixels is less than this fraction, the
        iterations stop.
    max_iterations : int
        Ceiling on the number of clipping iterations.

    Returns a tuple of the median, the sigma, and the number of iterations.
    """

    # Make sure data is safe
    data = data[(np.isnan(data) == False) * np.isfinite(data)]

    median = np.median(data)
    iteration = 0
    finished = False
    while not finished:
        iteration += 1
        lastct = median
        median = np.median(data)
        sigma = np.std(data)

        # Reduce data set
        index = np.nonzero(np.abs(data - median) < clip_sigma * sigma)
        if np.size(index) > 0:
            data = data[index]

        if (abs(median - lastct) / abs(lastct) < limit_ratio) \
                or (iteration >= max_iterations):
            finished = True

    median = np.median(data)
    sigma = np.std(data)

    return median, sigma, iteration