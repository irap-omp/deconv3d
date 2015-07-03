from numpy.core.numeric import Infinity
import numpy as np
from rtnorm import rtnorm


class LineModel:
    """
    Interface for the model of the spectral line.
    Your line models should extend this class, and implement its methods.
    Duck typing is very good, but this is cleaner, faster and more maintainable,
    as we have a lot of methods to check for. LBYL > EAFP here.

    See `SingleGaussianLineModel` below for an implementation example.
    """

    def __init__(self):  # PEP compliance
        pass

    def parameters(self):
        """
        Returns a list of strings, which are the (unique!) names of the
        parameters of your line model.
        """
        raise NotImplementedError()

    def min_boundaries(self, runner):
        """
        Returns a list of the (default) minimum boundaries of the parameters of
        your line model.
        """
        raise NotImplementedError()

    def max_boundaries(self, runner):
        """
        Returns a list of the (default) maximum boundaries of the parameters of
        your line model.
        """
        raise NotImplementedError()

    def post_jump(self, runner, old_parameters, new_parameters):
        """
        The model may want to mutate the `new_parameters` right after the Cauchy
        jumping, for example to apply Gibbs within MH.
        The `old_parameters` are provided for convenience, you should not mutate
        them.
        """
        pass

    def modelize(self, runner, x, parameters):
        """
        Returns a list of the same size as the input list `x`, containing the
        values of this line model for the provided `parameters`.
        """
        raise NotImplementedError()


class SingleGaussianLineModel(LineModel):
    """
    A single gaussian curve, defined by its three usual parameters.
    This is the default line model that deconv3d uses.
    """

    def parameters(self):
        return ['a', 'c', 'w']

    def min_boundaries(self, runner):
        return [0, 0, 0]

    def max_boundaries(self, runner):
        """
        The FSF is normalized, so we need to adjust the maximum of our amplitude
        accordingly.
        """
        cube = runner.cube
        fsf = runner.fsf
        fsf_max = np.amax(fsf)
        a_max = np.amax(cube.data)
        if fsf_max > 0:
            a_max = a_max / fsf_max
        return [a_max, cube.data.shape[0]-1, cube.data.shape[0]]

    def post_jump(self, runner, old_parameters, new_parameters):
        """
        Mutates the `new_parameters` right after the Cauchy jumping, to apply
        Gibbs within MH for the amplitude.
        Note: we don't use this, this was an attempt to abstractify Gibbs.
        """
        # fixme
        mu = 0.0
        sigma = 1.0
        # new_parameters[0] = rtnorm(0, Infinity, mu=mu, sigma=sigma)[0]

    def modelize(self, runner, x, parameters):
        """
        This model is a simple gaussian curve.
        """
        return self.gaussian(x, parameters[0], parameters[1], parameters[2])

    @staticmethod
    def gaussian(x, a, c, w):
        """
        Returns `g(x)`, `g` being a gaussian described by the other parameters :

        a: Amplitude
        c: Center
        w: Standard deviation, aka. RMS Width

        If `x` is an `ndarray`, the return value will be an `ndarray` too.
        """
        return a * np.exp(-1. * (x - c) ** 2 / (2. * w ** 2))