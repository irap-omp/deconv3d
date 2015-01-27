## GENERAL PACKAGES ############################################################
import os
import numpy
import numpy as np
from hyperspectral import HyperspectralCube as Cube

## LOCAL PACKAGES ##############################################################

from instruments import Instrument

## OPTIONAL PACKAGES ###########################################################
try:
    import bottleneck as bn
except ImportError:
    import numpy as bn
try:
    from astropy.io.fits import Header
except ImportError:
    from pyfits import Header


class Run:
    """

    """
    def __init__(self, cube, instrument, max_iterations=10000):

        # Set up the input data cube
        if isinstance(cube, basestring):
            cube = Cube.from_fits(cube)
        if not isinstance(cube, Cube):
            # try:  # todo: implement Cube.from_mpdaf() first
            #     import mpdaf
            #     if isinstance(cube, mpdaf.obj.Cube):
            #         if variance is None:
            #             variance = Cube(data=cube.var)
            #         cube = Cube.from_mpdaf(cube)
            #     else:
            #         raise TypeError("Provided cube is not a HyperspectralCube "
            #                         "nor mpdaf's Cube")
            # except ImportError:
            #     raise TypeError("Provided cube is not a HyperspectralCube")
            raise TypeError("Provided cube is not a HyperspectralCube")
        if cube.is_empty():
            raise ValueError("Provided cube is empty")
        self.cube = cube

        # Set up the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("Provided instrument is not an Instrument")
        self.instrument = instrument

        # Initialize
        cube_shape = cube.data.shape
        cube_width = cube_shape[2]
        cube_height = cube_shape[1]
        cube_depth = cube_shape[0]
        current_iteration = 0

    ## SIMULATOR ###############################################################

    # def

    @staticmethod
    def gaussian(x, a, c, w):
        """
        Returns `g(x)`, `g` being a gaussian described by the other parameters :

        a: Amplitude
        c: Center
        w: Width

        If `x` is an `ndarray`, the return value will be an `ndarray` too.
        """
        return a * np.exp(-1. * (x - c) ** 2 / (2. * w ** 2))

    ## PROBABILITIES ###########################################################

    def likelihood(self):
        """
        See http://en.wikipedia.org/wiki/Likelihood_function
        """
        pass

    def prior(self):
        """
        See http://en.wikipedia.org/wiki/Prior_probability
            http://en.wikipedia.org/wiki/A_priori_probability
        """
        pass

    def posterior(self):
        """
        See http://en.wikipedia.org/wiki/Posterior_probability
        """
        pass
