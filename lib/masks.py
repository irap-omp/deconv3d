
import numpy as np
from hyperspectral import HyperspectralCube as Cube


def read_hyperspectral_cube(cube):
    if isinstance(cube, basestring):
        cube = Cube.from_fits(cube)
    if not isinstance(cube, Cube):
        raise TypeError("Provided cube is not a HyperspectralCube")
    if cube.is_empty():
        raise ValueError("Provided cube is empty")

    return cube


def above_percentile(cube, percentile=30):
    """
    Will create a mask that will only allow the pixels *above* the `percentile`.
    """
    cube = read_hyperspectral_cube(cube)
    img = np.nansum(cube, axis=0)  # flatten along spectral axis

    p = np.nanpercentile(img, percentile)
    mask = np.copy(img)
    mask[img >= p] = 1
    mask[img < p] = 0

    return mask

