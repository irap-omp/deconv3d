# coding=utf-8

## GENERAL PACKAGES ############################################################
import numpy as np
import logging
from os.path import splitext
from math import log, isnan
from hyperspectral import HyperspectralCube as Cube
from matplotlib import pyplot as plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deconv3d')

## LOCAL PACKAGES ##############################################################
from instruments import Instrument
from convolution import convolve_1d
from math_utils import median_clip
from line_models import LineModel, SingleGaussianLineModel

## OPTIONAL PACKAGES ###########################################################
try:
    import bottleneck as bn
except ImportError:
    import numpy as bn
try:
    from astropy.io.fits import Header
except ImportError:
    from pyfits import Header


## MH WITHIN GIBBS RUNNER ######################################################
class Run:
    """
    todo
    """
    def __init__(
        self,
        cube, instrument,
        variance=None,
        model=SingleGaussianLineModel,
        max_iterations=10000,
        min_acceptance_rate=0.05
    ):

        # Assign the logger to a property for convenience
        self.logger = logger

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
        self.data_cube = cube.data

        # Collect informations about the cube
        cube_shape = cube.data.shape
        cube_width = cube_shape[2]
        cube_height = cube_shape[1]
        cube_depth = cube_shape[0]

        # Flag valid spaxels : we use only them for iteration and summation
        # This is an image, not a cube. By default, we flag as invalid all
        # spaxels that have a nan value somewhere in the spectrum.
        # 0 : invalid
        # 1 : valid
        self.validity_flag = np.ones((cube_height, cube_width))
        self.validity_flag[np.isnan(np.sum(self.data_cube, 0))] = 0

        # Set up the variance
        if variance is not None:
            if isinstance(variance, basestring):
                variance = Cube.from_fits(variance)
            if not isinstance(variance, Cube):
                raise TypeError("Provided variance is not a Cube")
            if variance.data is None:
                self.logger.warning("Provided variance cube is empty")
            self.logger.info("Using provided variance : %s" % variance)
            variance_cube = variance.data
            self.logger.info("Replacing zeros in the variance cube by 1e12")
            variance_cube = np.where(variance_cube == 0.0, 1e12, variance_cube)
            #variance = np.where(np.isnan(cube_data), 1e12, variance)
        else:
            # Clip data, and collect standard deviation sigma
            sub_data = np.copy(cube.data[2:-2, 2:-4, 2:4])
            _, clip_sigma, _ = median_clip(sub_data, 2.5)
            # Adjust sigma if it is zero, as we'll divide with it later on
            if clip_sigma == 0:
                clip_sigma = 1e-20  # arbitrarily low value
            variance_cube = clip_sigma ** 2

        # Save the variance cube and the standard deviation cube (error cube)
        self.variance_cube = variance_cube             # cube of sigma^2
        self.error_cube = np.sqrt(self.variance_cube)  # cube of sigmas

        # Set up the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("Provided instrument is not an Instrument")
        self.instrument = instrument

        # Set up the spread functions from the instrument
        self.lsf = self.instrument.lsf.as_vector(self.cube)
        self.fsf = self.instrument.fsf.as_image(self.cube)

        # Set up the model to match against, from a class name or an instance
        if isinstance(model, LineModel):
            self.model = model
        else:
            self.model = model()
            if not isinstance(self.model, LineModel):
                raise TypeError("Provided model is not a LineModel")

        current_iteration = 0
        current_acceptance_rate = 0.

        # Set up the FSF and the LSF, from the instrument
        lsf = instrument.lsf.as_vector(cube)
        fsf = instrument.fsf.as_image(cube)
        if fsf.shape[0] % 2 == 0 or fsf.shape[1] % 2 == 0:
            raise ValueError("FSF *must* be of odd dimensions")

        # Parameters boundaries
        min_boundaries = np.array(self.model.min_boundaries(cube))
        max_boundaries = np.array(self.model.max_boundaries(cube))

        self.logger.info("Min boundaries : %s" %
                         dict(zip(self.model.parameters(), min_boundaries)))
        self.logger.info("Max boundaries : %s" %
                         dict(zip(self.model.parameters(), max_boundaries)))

        # Parameter jumping amplitude
        jumping_amplitude = np.array(np.sqrt(
            (max_boundaries - min_boundaries) ** 2 / 12.
        ) * len(self.model.parameters()) / np.size(cube.data))

        # Prepare the chain of parameters
        # One set of parameters per iteration, per spaxel
        parameters_count = len(self.model.parameters())
        parameters = np.ndarray(
            (max_iterations, cube_height, cube_width, parameters_count)
        )

        # Prepare the array of contributions
        # We only store "last iteration" contributions, or the RAM explodes.
        contributions = np.ndarray((
            cube_height, cube_width,  # spaxel coordinates
            cube_depth, cube_height, cube_width  # contribution data cube
        ))

        # Initial parameters, picked at random between boundaries
        for (y, x) in self.spaxel_iterator():
            p_new = \
                min_boundaries + (max_boundaries - min_boundaries) * \
                np.random.rand(len(self.model.parameters()))
            parameters[0][y][x] = p_new

        # Initial error/difference between simulation and data
        sim = np.zeros_like(cube.data)

        print("Iteration #%d" % current_iteration)

        lsf_fft = None  # memoization holder for performance
        for (y, x) in self.spaxel_iterator():

            contribution, lsf_fft = self.contribution_of_spaxel(
                x, y, parameters[0][y][x],
                cube_width, cube_height, cube_depth,
                fsf, lsf, lsf_fft=lsf_fft
            )

            sim = sim + contribution
            contributions[y, x, :, :, :] = contribution

        err = cube.data - sim
        current_iteration += 1

        # Loop as many times as specified, as long as the acceptance is OK
        while \
                current_iteration < max_iterations \
                and \
                (
                    current_acceptance_rate > min_acceptance_rate or
                    current_acceptance_rate == 0.
                ):

            print("Iteration #%d" % current_iteration)

            # Loop through all spaxels
            for (y, x) in self.spaxel_iterator():

                # Compute a new set of parameters for this spaxel
                p_old = np.array(parameters[current_iteration-1][y][x].tolist())
                p_new = self.jump_from(p_old, jumping_amplitude)

                # Check if new parameters are within the boundaries
                # It happens quite often that parameters are out of boundaries,
                # so we do not log anything because it slows the script a lot.
                too_low = np.array(p_new < min_boundaries)
                too_high = np.array(p_new > max_boundaries)
                if too_low.any() or too_high.any():
                    # print "New proposed parameters are out of boundaries."
                    parameters[current_iteration][y][x] = p_old
                    continue

                # Compute the contribution of the new parameters
                contribution, lsf_fft = self.contribution_of_spaxel(
                    x, y, p_new,
                    cube_width, cube_height, cube_depth,
                    fsf, lsf, lsf_fft=lsf_fft
                )

                # Remove contribution of parameters of previous iteration
                ul = err + contributions[y, x]
                # Add contribution of new parameters
                res = ul - contribution

                # Minimum acceptance ratio, picked randomly between -∞ and 0
                min_acceptance_ratio = log(np.random.rand())

                # Actual acceptance ratio
                acceptance_ratio = \
                    - 1./2. * np.sum((res/variance_cube)**2) \
                    + 1./2. * np.sum((err/variance_cube)**2)

                if min_acceptance_ratio < acceptance_ratio:
                    parameters[current_iteration][y][x] = p_new
                    contributions[y, x] = contribution
                    err = res
                else:
                    parameters[current_iteration][y][x] = p_old

            current_iteration += 1

        # Output
        self.parameters = parameters

    ## ITERATORS ###############################################################

    def spaxel_iterator(self):
        """
        Creates a generator that will yield all (y, x) doublets, for iteration.
        This generator iterates over the spaxel indices column by column,
        skipping spaxels flagged as invalid.

        Override this to implement your own iteration logic.
        """
        h = self.cube.data.shape[1]
        w = self.cube.data.shape[2]
        for y in range(0, h):
            for x in range(0, w):
                if self.validity_flag[y,x] == 1:
                    yield (y, x)

    ## MCMC ####################################################################

    def jump_from(self, parameters, amplitude):
        """
        Draw new parameters, using a proposal density (aka jumping distribution)
        that is defaulted to Cauchy jumping.

        Override this to provide your own proposal density.
        """
        size = len(parameters)
        random_uniforms = np.random.uniform(-np.pi / 2., np.pi / 2., size=size)
        return parameters + amplitude * np.tan(random_uniforms)

    def extract_parameters(self):
        """
        Extracts the best fit parameters from the chain of parameters.
        This takes the mean after removing the burnout.
        Returns a 2D array of parameters. (one parameter set per spaxel)
        """

        # Remove the burnout (let's say the first 20%)
        i = self.parameters.shape[0]
        parameters_burned_out = self.parameters[int(i/5.):, ...]

        return np.mean(parameters_burned_out, 0)

    ## SIMULATOR ###############################################################

    def simulate(self, shape, parameters):
        """
        wip
        """

        sim = np.zeros(shape)

        lsf_fft = None  # memoization holder for performance
        for (y, x) in self.spaxel_iterator():

            contribution, lsf_fft = self.contribution_of_spaxel(
                x, y, parameters[y][x],
                shape[2], shape[1], shape[0],
                self.fsf, self.lsf, lsf_fft=lsf_fft
            )

            sim = sim + contribution

        return sim

    def contribution_of_spaxel(self, x, y, parameters,
                               cube_width, cube_height, cube_depth,
                               fsf, lsf, lsf_fft=None):
        """
        The contribution cube of the line described by `parameters` in the
        spaxel (x, y), after convolution by the `lsf` and `fsf`.

        Returns a cube of shape (cube_width, cube_height, cube_depth), mostly
        empty (zeros), and with the spatially spread contribution of the line
        located at pixel (x, y).
        """

        # Initialize output cube
        sim = np.zeros((cube_depth, cube_height, cube_width))

        # Raw line model
        line = self.model.modelize(range(0, cube_depth), parameters)

        # Spectral convolution: using the Fast Fourier Transform of the LSF
        if lsf_fft is None:
            line_conv, lsf_fft = convolve_1d(line, lsf)
        else:
            line_conv, _ = convolve_1d(line, lsf_fft, compute_fourier=False)

        # Spatial convolution: we iterate over the neighboring spaxels and
        # add the line model, scaled by the FSF.
        fsf_half_height = (fsf.shape[1]-1)/2
        fsf_half_width = (fsf.shape[0]-1)/2
        for fsf_y in range(-fsf_half_height, +fsf_half_height+1):
            yy = y + fsf_y
            yyy = fsf_y + fsf_half_height
            if yy < 0 or yy >= cube_height:
                continue  # spaxel is out of the cube -- EDGE EFFECTS !?
            for fsf_x in range(-fsf_half_width, +fsf_half_width+1):
                xx = x + fsf_x
                xxx = fsf_x + fsf_half_width
                if xx < 0 or xx >= cube_width:
                    continue  # spaxel is out of the cube -- EDGE EFFECTS !?
                sim[:, yy, xx] += line_conv * fsf[yyy, xxx]

        return sim, lsf_fft

    ## PLOTS ###################################################################

    def plot_images(self, filepath=None):
        """
        Plot a mosaic of images of the cropped (along z) cubes,
        and then either show it or save it to a file.

        filepath: string
            If specified, will write the plot to a file instead of showing it.
            The file will be created at the provided absolute or relative filepath.
            The extension of the file must be either png or pdf.
        z_crop: None|int
            The maximum and total length of the crop (in pixels) along z,
            centered on the galaxy's z position.
            If you provide zero or an even value (2n),
            the closest bigger odd value will be used (2n+1).
            By default, will not crop.
        """

        if filepath is not None:
            name, extension = splitext(filepath)
            supported_extensions = ['.png', '.pdf']
            if not extension in supported_extensions:
                raise ValueError("Extension '%s' is not supported, "
                                 "you may use one of %s",
                                 extension, ', '.join(supported_extensions))


        p = self.extract_parameters()
        convolved_cube = self.simulate(self.data_cube.shape, p)
        self._plot_images(self.data_cube, convolved_cube)

        if filepath is None:
            plot.show()
        else:
            plot.savefig(filepath)

    def _plot_images(self, data_cube, convolved_cube):

        fig = plot.figure(1, figsize=(16, 9))
        plot.clf()
        plot.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.05,
                             top=0.95, left=0.05, right=0.95)

        # MEASURE
        sub = fig.add_subplot(2, 2, 1)
        sub.set_title('Measured')
        measured_cube = data_cube[:, :, :]
        measured_image = (measured_cube.sum(0) / measured_cube.shape[0])
        plot.imshow(measured_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # CONVOLVED
        sub = fig.add_subplot(2, 2, 2)
        sub.set_title('Convolved')
        convolved_image = (convolved_cube.sum(0) / convolved_cube.shape[0])
        plot.imshow(convolved_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # FSF
        sub = fig.add_subplot(2, 2, 3)
        sub.set_title('FSF')
        fsf_image = self.fsf
        plot.imshow(fsf_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        return fig