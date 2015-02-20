import numpy as np


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

    def min_boundaries(self, cube):
        """
        Returns a list of the (default) minimum boundaries of the parameters of
        your line model.
        """
        raise NotImplementedError()

    def max_boundaries(self, cube):
        """
        Returns a list of the (default) maximum boundaries of the parameters of
        your line model.
        """
        raise NotImplementedError()

    def modelize(self, x, parameters):  # unsure about the name of this method
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

    def min_boundaries(self, cube):
        return [0, 0, 0]

    def max_boundaries(self, cube):
        return [np.amax(cube.data), cube.data.shape[0]-1, cube.data.shape[0]]

    def modelize(self, x, parameters):
        return self.gaussian(x, parameters[0], parameters[1], parameters[2])

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