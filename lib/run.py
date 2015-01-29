## GENERAL PACKAGES ############################################################
from docutils.nodes import paragraph
import numpy as np
from hyperspectral import HyperspectralCube as Cube

## LOCAL PACKAGES ##############################################################
from instruments import Instrument

## OPTIONAL PACKAGES ###########################################################
from lib.convolution import convolve_1d

try:
    import bottleneck as bn
except ImportError:
    import numpy as bn
try:
    from astropy.io.fits import Header
except ImportError:
    from pyfits import Header


class LineModel:
    """
    Interface for the model of the spectral line.
    """

    def parameters(self):
        raise NotImplementedError("Implement parameters()")


class SingleGaussianLineModel(LineModel):

    def __init__(self):
        pass

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


## MH WITHIN GIBBS RUNNER ######################################################
class Run:
    """
    todo
    """
    def __init__(
        self,
        cube, instrument,
        model=SingleGaussianLineModel,
        max_iterations=10000,
        min_acceptance_rate=0.05
    ):

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

        # Set up the model to match against, from a class name or an instance
        if isinstance(model, LineModel):
            self.model = model
        else:
            self.model = model()
            if not isinstance(self.model, LineModel):
                raise TypeError("Provided model is not a TheoreticalModel")

        # Initialize
        cube_shape = cube.data.shape
        cube_width = cube_shape[2]
        cube_height = cube_shape[1]
        cube_depth = cube_shape[0]
        current_iteration = 0
        current_acceptance_rate = 0.

        # Initial parameters
        min_boundaries = np.asarray(self.model.min_boundaries(cube))
        max_boundaries = np.asarray(self.model.max_boundaries(cube))
        parameters = min_boundaries + (max_boundaries - min_boundaries) * np.random.rand(len(self.model.parameters()))

        # Prepare the chain
        types_param = ['float32' for i in range(len(self.model.parameters()))]
        dtype = zip(self.model.parameters(), types_param)
        chain = np.zeros(max_iterations, dtype=dtype)  # intensive operation !

        # Initial error/difference between simulation and data
        sim = np.zeros_like(cube.data)

        lsf = instrument.lsf.as_vector(cube)
        lsf_fft = None  # memoization holder for performance

        fsf = instrument.fsf.as_image(cube)

        if fsf.shape[0] % 2 == 0 or fsf.shape[1] % 2 == 0:
            raise ValueError("FSF *must* be of odd dimensions")

        for (y, x) in self.spaxel_iterator():

            # Raw line model
            line = self.model.modelize(range(0, cube_depth), parameters)

            # Spectral convolution
            if lsf_fft is None:
                line_conv, lsf_fft = convolve_1d(line, lsf)
            else:
                line_conv, _ = convolve_1d(line, lsf_fft, compute_fourier=False)

            # Spatial convolution, we iterate over the neighboring spaxels and
            # add the line model, scaled by the FSF.
            fsf_half_height = (fsf.shape[1]-1)/2
            fsf_half_width = (fsf.shape[0]-1)/2
            for fsf_y in range(-fsf_half_height, +fsf_half_height+1):
                yy = y + fsf_y
                yyy = fsf_y + fsf_half_height
                if yy < 0 or yy >= cube_height:
                    continue
                for fsf_x in range(-fsf_half_width, +fsf_half_width+1):
                    xx = x + fsf_x
                    xxx = fsf_x + fsf_half_width
                    if xx < 0 or xx >= cube_width:
                        continue
                    sim[:, yy, xx] += line_conv * fsf[yyy, xxx]

        print sim

        # import matplotlib.pyplot as plt
        # plt.plot(line)
        # plt.ylabel('line')
        # plt.show()

        # Loop as many times as specified, as long as the acceptance is OK
        while \
                current_iteration < max_iterations - 1 \
                and \
                (
                    current_acceptance_rate > min_acceptance_rate or
                    current_acceptance_rate == 0.
                ):

            # Loop through all pixels
            for (y, x) in self.spaxel_iterator():
                # fixme
                pass


            current_iteration += 1

    def spaxel_iterator(self):
        """
        Creates a generator that will yield all (y, x) doublets, for iteration.
        This generator iterates over the pixel "coordinates" column by column.
        Override this to implement your own iteration logic.
        """
        h = self.cube.data.shape[1]
        w = self.cube.data.shape[2]
        for y in range(0, h):
            for x in range(0, w):
                yield (y, x)

    ## SIMULATOR ###############################################################

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

    # WIP

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
