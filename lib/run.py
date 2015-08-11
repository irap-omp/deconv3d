# coding=utf-8

## GENERAL PACKAGES ############################################################
import sys
from logging import Logger
import math
import numpy as np
import logging
from os.path import splitext
from math import log
from hyperspectral import HyperspectralCube as Cube
from matplotlib import pyplot as plot
from lib.rtnorm import rtnorm

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
    from astropy.io.fits import Header, getdata
except ImportError:
    from pyfits import Header, getdata

## SOME CONSTANTS ##############################################################
CIRCLE = np.pi * 2.
CIRCLE_4TH = np.pi / 2.
MAXIMUM = sys.maxint


## MCMC RUNNER #################################################################
class Run:
    """
    This is the main runner of the deconvolution.
    Use it like this :
    ```
    cube = Cube.from_fits('my_fits.fits')
    inst = MUSE()
    run = Run(cube, inst, max_iterations=1000)
    run.plot_chain()
    ```

    cube: str | hyperspectral.HyperspectralCube
        The path to a FITS file, or a `hyperspectral.HyperspectralCube` object,
        containing the source data you want to deconvolve.
    instrument: Instrument
        The instrument object to use. Use `MUSE()`, for example.
    mask: ndarray
        An image of the spatial size of the above cube, filled with zeroes and
        ones. The runner will only try to deconvolve the spaxels where this mask
        is set to 1. The default mask is filled with ones, ie. transparent.
        This mask is also automatically opacified where there are NaN values in
        the input cube.
    variance: ndarray
        A variance cube of the same dimensions as the input cube.
    model: Type | LineModel
        The model object (or classname) of the lines you want to use in the
        simulation. This is defaulted to SingleGaussianLineModel, a simple model
        of a single gaussian line that has three parameters defining it.
    initial_parameters: str | ndarray | None
        Either a filepath to a .npy file holding a ndarray, or the ndarray
        itself. Should be a 3D array of shape :
        (cube_height, cube_width, model_parameters_count).
        If it is a 1D array of the length of the model parameters, it will be
        broadcasted to each spaxel.
        When not defined, the initial parameters will be picked at random
        between their respective boundaries.
    jump_amplitude: float | ndarray
        You can tweak the Cauchy jump using this coefficient.
        You can provide a ndarray of the dimensions of the parameters of your
        line model, to customize the jump amplitude of each parameter.
        The default value of this is 1, which means no tweaking.
    max_iterations: int
        The number of iterations after which the chain will end.
    """
    #@profile
    def __init__(
        self,
        cube,
        instrument,
        mask=None,
        variance=None,
        model=SingleGaussianLineModel,
        initial_parameters=None,
        jump_amplitude=1.0,
        max_iterations=10000,
        min_acceptance_rate=0.01
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

        # Mask
        if mask is None:
            mask = np.ones((cube_height, cube_width))
        if isinstance(mask, basestring):
            mask = getdata(mask)
        self.mask = mask

        # Flag invalid spaxels : we won't them for iteration and summation
        # By default, we flag as invalid all spaxels that have a NaN value
        # somewhere in the spectrum.
        self.mask[np.isnan(np.sum(self.data_cube, 0))] = 0

        # Count the number of spaxels we're going to parse
        spaxels_count = np.sum(self.mask)

        # Set up the variance
        if variance is not None:
            if isinstance(variance, basestring):
                variance = Cube.from_fits(variance)
            if isinstance(variance, Cube):
                if variance.data is None:
                    self.logger.warning("Provided variance cube is empty")
                self.logger.info("Using provided variance : %s" % variance)
                variance_cube = variance.data
                self.logger.info("Replacing zeros in the variance cube by 1e12")
                variance_cube = np.where(variance_cube == 0.0, 1e12, variance_cube)
            elif isinstance(variance, np.ndarray):
                variance_cube = variance
            else:
                raise TypeError("Provided variance is not a Cube")
        else:
            # Clip data, and collect standard deviation sigma
            sub_data = np.copy(cube.data[2:-2, 2:-4, 2:4])
            _, clip_sigma, _ = median_clip(sub_data, 2.5)
            # Adjust sigma if it is zero, as we'll divide with it later on
            if clip_sigma == 0:
                clip_sigma = 1e-20  # arbitrarily low value
            variance_cube = np.array(clip_sigma ** 2)

        # Save the variance cube and the standard deviation cube (error cube)
        self.variance_cube = variance_cube             # cube of sigmas²
        self.error_cube = np.sqrt(self.variance_cube)  # cube of sigmas

        # Set up the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("Provided instrument is not an Instrument")
        self.instrument = instrument

        # Set up the spread functions from the instrument
        self.lsf = self.instrument.lsf.as_vector(self.cube)
        self.fsf = self.instrument.fsf.as_image(self.cube)
        if self.fsf.shape[0] % 2 == 0 or self.fsf.shape[1] % 2 == 0:
            raise ValueError("FSF *must* be of odd dimensions")

        # Assert that the spread functions are normalized (we can remove this)
        #assert np.nansum(self.fsf) == 1.0
        #assert np.nansum(self.lsf) == 1.0

        # Collect the shape of the FSF
        fh = self.fsf.shape[0]
        fw = self.fsf.shape[1]
        # The FSF *must* be odd-shaped, so these are integers
        fhh = (fh - 1) / 2  # FSF half height
        fhw = (fw - 1) / 2  # FSF half width

        # Set up the model to match against, from a class name or an instance
        if isinstance(model, LineModel):
            self.model = model
        else:
            self.model = model()
            if not isinstance(self.model, LineModel):
                raise TypeError("Provided model is not a LineModel")

        # Parameters boundaries
        min_boundaries = np.array(self.model.min_boundaries(self))
        max_boundaries = np.array(self.model.max_boundaries(self))

        self.logger.info("Min boundaries : %s" %
                         dict(zip(self.model.parameters(), min_boundaries)))
        self.logger.info("Max boundaries : %s" %
                         dict(zip(self.model.parameters(), max_boundaries)))

        # Assert that boundaries are consistent
        if (min_boundaries > max_boundaries).any():
            raise ValueError("Boundaries are inconsistent: min > max.")

        # Collect information about the model
        parameters_count = len(self.model.parameters())

        # Parameter jumping amplitude
        jumping_amplitude = np.ones(parameters_count) * (0.5 * jump_amplitude)

        # Do we even need to do MH within Gibbs ?
        gpi = self.model.gibbs_parameter_index()
        do_gibbs = False if gpi is None else True

        if do_gibbs:
            self.logger.info("MH within Gibbs enabled for parameter %d." % gpi)
            # Make sure the Gibbs'd parameter has a jumping amplitude of 0
            jumping_amplitude[gpi] = 0

        # Prepare the chain of parameters
        # One set of parameters per iteration, per spaxel
        parameters = np.ndarray(
            (max_iterations, cube_height, cube_width, parameters_count)
        )

        # Prepare the chain of "likelihoods"?
        likelihoods = np.ndarray((max_iterations, cube_height, cube_width))

        # Prepare the array of contributions
        # We only store "last iteration" contributions, or the RAM explodes.
        contributions = np.ndarray((
            cube_height, cube_width,  # spaxel coordinates
            cube_depth, cube_height, cube_width  # contribution data cube
        ))

        cur_iteration = 0  # Holder for the # of the current iteration
        current_acceptance_rate = 0.

        # Initial parameters
        if initial_parameters is not None:
            # ... defined by the user
            if isinstance(initial_parameters, basestring):
                initial_parameters = np.load(initial_parameters)
            initial_parameters = np.array(initial_parameters)

            parameters[0] = initial_parameters
        else:
            # ... picked at random between boundaries
            for (y, x) in self.spaxel_iterator():
                p_new = \
                    min_boundaries + (max_boundaries - min_boundaries) * \
                    np.random.rand(parameters_count)
                # p_new[0] = 0.  # fixme: set amplitude to start at 0
                parameters[0][y][x] = p_new

        # Initial simulation
        sim = np.zeros_like(cube.data)

        self.logger.info("Iteration #1")

        lsf_fft = None  # memoization holder for performance
        for (y, x) in self.spaxel_iterator():

            contribution, lsf_fft = self.contribution_of_spaxel(
                x, y, parameters[0][y][x],
                cube_width, cube_height, cube_depth,
                self.fsf, self.lsf, lsf_fft=lsf_fft
            )

            sim = sim + contribution
            contributions[y, x, :, :, :] = contribution

        # Initial error/difference between simulation and data
        err_old = cube.data - sim
        cur_iteration += 1

        # Accepted iterations counter (we accepted the whole first iteration)
        accepted_count = spaxels_count

        # Loop as many times as specified, as long as the acceptance is OK
        while \
                cur_iteration < max_iterations \
                and \
                (
                    current_acceptance_rate > min_acceptance_rate or
                    current_acceptance_rate == 0.
                ):

            # Acceptance rate
            max_accepted_count = spaxels_count * cur_iteration
            if max_accepted_count > 0:
                current_acceptance_rate = \
                    float(accepted_count) / float(max_accepted_count)

            self.logger.info(
                "Iteration #%d / %d, %2.0f%%" %
                (cur_iteration+1, max_iterations, 100 * current_acceptance_rate)
            )

            # Loop through all spaxels
            for (y, x) in self.spaxel_iterator():

                # Compute a new set of parameters for this spaxel
                p_old = np.array(parameters[cur_iteration-1][y][x].tolist())
                p_new = self.jump_from(p_old, jumping_amplitude)

                # Now, post-process the parameters, if the model requires it to
                self.model.post_jump(self, p_old, p_new)

                # Check if new parameters are within the boundaries
                # It happens quite often that parameters are out of boundaries,
                # so we do not log anything because it slows the script a lot.
                out_of_bounds = False
                too_low = np.array(p_new < min_boundaries)
                too_high = np.array(p_new > max_boundaries)
                if too_low.any() or too_high.any():
                    #print "New proposed parameters are out of boundaries."
                    out_of_bounds = True
                    if not do_gibbs:
                        parameters[cur_iteration][y][x] = p_old
                        continue

                # Compute the contribution of the new parameters
                contribution, lsf_fft = self.contribution_of_spaxel(
                    x, y, p_new,
                    cube_width, cube_height, cube_depth,
                    self.fsf, self.lsf, lsf_fft=lsf_fft
                )

                # Remove contribution of parameters of previous iteration
                ul = err_old + contributions[y, x]
                # Add contribution of new parameters
                err_new = ul - contribution

                # Compute the limits of the affected section
                y_min = max(y-fhh, 0)
                y_max = min(y+fhh+1, cube_height)
                x_min = max(x-fhw, 0)
                x_max = min(x+fhw+1, cube_width)

                # Actual acceptance ratio
                # We optimize by computing only around the spatial area that was
                # modified, aka. the area of the FSF around our current spaxel.
                err_new_part = err_new[:, y_min:y_max, x_min:x_max]
                err_old_part = err_old[:, y_min:y_max, x_min:x_max]

                # Variance may be a scalar, an image or a plain cube
                # Maybe we'll force variance to be a cube at init so that we may
                # remove this. Not sure which is faster.
                var_part = variance_cube
                if len(variance_cube.shape) == 2:
                    var_part = variance_cube[y_min:y_max, x_min:x_max]
                elif len(variance_cube.shape) == 3:
                    var_part = variance_cube[:, y_min:y_max, x_min:x_max]

                # Two sums of small arrays (shaped at most like the FSF) is
                # faster than one sum of a big array (shaped like the cube).
                ar_part_old = 0.5 * bn.nansum(err_old_part ** 2 / var_part)
                ar_part_new = 0.5 * bn.nansum(err_new_part ** 2 / var_part)

                cur_acceptance = ar_part_old - ar_part_new

                # I don't know what I'm doing
                likelihoods[cur_iteration][y][x] = cur_acceptance

                # Minimum acceptance ratio, picked randomly between -∞ and 0
                min_acceptance = log(np.random.rand())

                # Save new parameters only if acceptance ratio is acceptable
                if min_acceptance < cur_acceptance and not out_of_bounds:
                    contributions[y, x, :, :, :] = contribution
                    err_old = err_new
                    accepted_count += 1
                    p_end = p_new.copy()
                else:
                    # Otherwise the new parameters are the same as the old ones
                    p_end = p_old.copy()

                # Save the parameters
                parameters[cur_iteration][y][x] = p_end

                if do_gibbs:
                    # GIBBS
                    # fixme: make sure this works ?
                    # fixme: some of these maths are tied to the fact that the
                    # gibbsed value is the amplitude ; fix it.

                    # Collect some values we'll need
                    gibbsed_value = parameters[cur_iteration][y][x][gpi]
                    # Compute some subcubes we'll need
                    ul_part = ul[:, y_min:y_max, x_min:x_max]
                    ek_part = contributions[y, x, :, y_min:y_max, x_min:x_max]
                    if gibbsed_value != 0:
                        ek_part = ek_part / gibbsed_value
                    else:
                        # The contribution of the previous iteration is empty,
                        # as the amplitude is zero.
                        # In order to make the following math work, we need
                        # to have a non-null contribution, with amplitude 1.
                        p_one = parameters[cur_iteration][y][x].copy()
                        p_one[gpi] = 1.
                        # print "Params One", p_one
                        contribution_one, _ = self.contribution_of_spaxel(
                            x, y, p_one,
                            cube_width, cube_height, cube_depth,
                            self.fsf, self.lsf, lsf_fft=None
                        )
                        ek_part = contribution_one[:, y_min:y_max, x_min:x_max]
                    ra = float(max_boundaries[gpi] ** 2)  # apriori variance
                    ro = ra / (1. + ra * np.sum(ek_part ** 2 / var_part))
                    mu = ro * np.sum(ek_part * ul_part / var_part)
                    # Pick from a random truncated normal distribution
                    r = rtnorm(min_boundaries[gpi], max_boundaries[gpi],
                               mu=mu, sigma=np.sqrt(ro))[0]
                    # print "Amplitude sigma", np.sqrt(ro)
                    # print "Amplitude", p_end[gpi], " -> ", r
                    # And now re-compute everything
                    p_end[gpi] = r

                    # It's costly to recompute the contribution ! fixme
                    #contribution_test, _ = self.contribution_of_spaxel(
                    #    x, y, p_end,
                    #    cube_width, cube_height, cube_depth,
                    #    self.fsf, self.lsf, lsf_fft=None
                    #)
                    # We can use the subcubes we already computed
                    contribution = np.zeros(cube.shape)
                    contribution[:, y_min:y_max, x_min:x_max] = ek_part * r

                    err_new = ul - contribution  # fixme: optimize
                    # And, finally, write it
                    contributions[y, x, :, :, :] = contribution
                    err_old = err_new
                    parameters[cur_iteration][y][x] = p_end

            # We compute the error based on the error of the previous iteration
            # and therefore, due to numerical instability, small errors slowly
            # creep in. To squash them, we sometimes recompute a "fresh" error.
            # The error creep is usually in the 1e-14 order.
            if cur_iteration % 1000 == 0:
                err_dbg = self._compute_error_in_one_step(
                    self.data_cube,
                    parameters[cur_iteration, :, :],
                    self.fsf, self.lsf
                )
                # diff = np.abs(err_dbg - err_old)
                # dbg_error_creep.append(np.amax(diff))
                # print "Corrected error creep of ", np.amax(diff)
                err_old = err_dbg
                # assert np.allclose(err_dbg, err_old, atol=0., rtol=1e-06)
            ##################

            cur_iteration += 1

        # Output
        self.parameters_chain = parameters
        self.likelihoods = likelihoods
        self.parameters = self.extract_parameters()
        #print "Extracted Parameters : %s" % str(self.parameters)
        self.convolved_cube = Cube(
            data=self.simulate_convolved(cube_shape, self.parameters),
            meta=self.cube.meta
        )
        self.clean_cube = Cube(
            data=self.simulate_clean(cube_shape, self.parameters),
            meta=self.cube.meta
        )

    ## ITERATORS ###############################################################

    def spaxel_iterator(self):
        """
        Creates a generator that will yield all (y, x) doublets, for iteration.
        This generator iterates over the spaxel indices column by column,
        skipping spaxels not present in the mask.

        Override this to implement your own iteration logic.
        """
        h = self.cube.data.shape[1]
        w = self.cube.data.shape[2]
        for y in range(0, h):
            for x in range(0, w):
                if self.mask[y, x] == 1:
                    yield (y, x)

    ## MCMC ####################################################################

    def jump_from(self, parameters, amplitude):
        """
        Draw new parameters, using a proposal density (aka jumping distribution)
        that is defaulted to Cauchy jumping.

        Override this to provide your own proposal density.
        """
        size = len(parameters)
        random_uniforms = np.random.uniform(-CIRCLE_4TH, CIRCLE_4TH, size=size)
        return parameters + amplitude * np.tan(random_uniforms)

    def extract_parameters(self, percentage=20.):
        """
        Extracts the mean parameters from the chain of parameters from the last
        `percentage`% parameters.
        Returns a 2D array of parameters sets. (one parameter set per spaxel)
        Access it Y first, then X : params[Y][X].
        """

        s = (100. - percentage) * self.parameters_chain.shape[0] / 100.
        parameters_burned_out = self.parameters_chain[int(s):, ...]

        return np.mean(parameters_burned_out, 0)

    ## SIMULATOR ###############################################################

    def simulate_clean(self, shape, parameters):
        """
        Returns a cube of `shape`, containing the simulation (the sum of all the
        lines) for the given map of parameters sets.

        shape:
            The desired shape of the output cube.
        parameters: np.ndarray
            There is one parameters set per spaxel, so this should be a 3D array
            holding one parameters set per spaxel, of shape (Y, X, len(params)).
        """

        # Initialize an empty output cube
        sim = np.zeros(shape)
        z = shape[0]

        # For each spaxel, add its line
        for (y, x) in self.spaxel_iterator():

            # Contribution of the line, not convolved
            line = self.model.modelize(self, range(0, z), parameters[y, x])

            sim[:, y, x] = line

        return sim

    def simulate_convolved(self, shape, parameters):
        """
        Returns a cube containing the simulation (the sum of all the convolved
        lines) for the given map of parameters sets.

        shape:
            The desired shape of the output cube.
        parameters: np.ndarray
            There is one parameters set per spaxel, so this should be a 2D array
            of parameters sets.
        """

        # Initialize an empty output cube
        sim = np.zeros(shape)

        # Memoize the Fast Fourier Transform of the Line Spead Function
        lsf_fft = None  # (for performance)

        # For each spaxel, add its contribution
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
        located at spaxel (x, y).

        parameters:
            A single set of parameters for the model.
        """

        # Initialize output cube
        sim = np.zeros((cube_depth, cube_height, cube_width))

        # Raw line model
        line = self.model.modelize(self, range(0, cube_depth), parameters)

        # Spectral convolution: using the Fast Fourier Transform of the LSF
        if lsf_fft is None:
            line_conv, lsf_fft = convolve_1d(line, lsf)
        else:
            line_conv, _ = convolve_1d(line, lsf_fft, compute_fourier=False)

        # Collect the shape of the FSF
        fh = fsf.shape[0]
        fw = fsf.shape[1]
        # The FSF *must* be odd-shaped, so these are integers
        fhh = (fh-1)/2  # FSF half height
        fhw = (fw-1)/2  # FSF half width

        # Create the contribution cube, through spatial convolution
        # This odd syntax simply extrudes the FSF with the line
        local_contrib = fsf * line_conv[:, np.newaxis][:, np.newaxis]

        # Copy the contribution into a cube sized like the input cube
        # We only copy the portions that intersect spatially
        sim[
            :,
            max(y-fhh, 0):min(y+fhh+1, cube_height),
            max(x-fhw, 0):min(x+fhw+1, cube_width)
        ]\
            = local_contrib[
                :,
                max(fhh-y, 0):min(cube_height+fhh-y, fh),
                max(fhw-x, 0):min(cube_width+fhw-x,  fw),
            ]

        return sim, lsf_fft

    ## SAVES ###################################################################

    def save(self, name, clobber=False):
        """
        Save the run data in various files whose filenames are prepended by
        `<name>_`.
        Here's a list of the generated files :
          - <name>_parameters.npy
            A 3D array, holding a parameters set for each spaxel.
            This can be used as input for `initial_parameters`.
          - <name>_images.png
            A mosaic of the relevant cubes, flattened along the spectral axis.

        name: string
            An absolute or relative name that will be used as prefix for the
            save files.
            Eg: 'my_run', or '/home/me/science/my_run'.
        clobber: bool
            When set to true, will OVERWRITE existing files.
        """
        self.save_parameters_npy("%s_parameters.npy" % name)
        self.save_matlab("%s_matlab.mat" % name)
        self.plot_images("%s_images.png" % name)
        self.plot_chain(filepath="%s_chain.png" % name)

        self.convolved_cube.to_fits("%s_convolved_cube.fits" % name, clobber)
        self.clean_cube.to_fits("%s_clean_cube.fits" % name, clobber)

        np.save("%s_fsf.npy" % name, self.fsf)
        np.save("%s_lsf.npy" % name, self.lsf)

    def save_parameters_npy(self, filepath):
        """
        Write the extracted parameters map to a file located at `filepath`.
        Will clobber an existing file. The filepath's extension should be `npy`.
        The saved map is the 3D array holding a parameters set for each spaxel.
        """

        np.save(filepath, self.extract_parameters())

    def save_matlab(self, filepath):
        """
        Write some data to a file located at `filepath`.
        Will clobber an existing file. The filepath's extension should be `mat`,
        for Matlab©.

        The `mat` file will contain :
        - parameters:
            The 3D array holding a parameters set for each spaxel.
            Shape: (cube_height, cube_width, parameters_count)
        - chain:
            The whole chain of parameters.
            Shape: (max_iterations, cube_height, cube_width, parameters_count)

        This requires `scipy`.

        filepath: string
            Will write the Matlab© file to this file path.
            It should end with `.mat`.
            You can also provide a `file`-like object instead of a `string`.
        """
        try:
            from scipy.io import savemat
            savemat(filepath, dict(
                parameters=self.extract_parameters(),
                chain=self.parameters_chain
            ))
        except ImportError:
            logger.error("The `scipy` package is required to save for matlab.")

    ## PLOTS ###################################################################

    def plot_chain(self, x=None, y=None, filepath=None, bound=False):
        """
        Plot the MCMC chain of the spaxel described by the indices (`x`, `y`).
        If `x` or `y` are not specified, they will default to the center of the
        spatial image if it is odd-sized, and `floor(size/2)` is it is
        even-shaped.

        filepath: string
            If specified, will write the plot to a file instead of showing it.
            The file will be created at the provided filepath, be it absolute or
            relative. The extension of the file must be either png or pdf.
        bound: bool
            Should we force the plot's Y axis to stretch to boundaries ?
            It may visually flatten the chain if its walk is not as broad as the
            boundaries' space.
        """

        self._check_image_filepath(filepath)

        if x is None:
            x = math.floor(self.cube.shape[2] / 2.)
        if y is None:
            y = math.floor(self.cube.shape[1] / 2.)

        chain = self.parameters_chain[:, y, x, :]

        # print chain
        # [[  2.27769676e-19   1.57019398e+01   7.37263527e+00]
        #  [  2.27769676e-19   1.57019398e+01   7.37263527e+00]]

        chain_transposed = np.transpose(chain)

        plot.clf()  # clear current figure

        names = self.model.parameters()
        rows = 2
        cols = len(names)
        bmin = self.model.min_boundaries(self)
        bmax = self.model.max_boundaries(self)

        # First row: the model's parameters
        for i in range(len(names)):
            plot.subplot2grid((rows, cols), (0, i % cols))
            plot.plot(chain_transposed[i])
            if bound:
                plot.ylim(bmin[i], bmax[i])
            plot.title(names[i], fontsize='small')

        # Second row: the reduced chi
        plot.subplot2grid((rows, cols), (1, 0), colspan=cols)
        plot.plot(self.likelihoods[:, y, x])
        plot.title('acceptance ratio', fontsize='small')

        if filepath is None:
            plot.show()
        else:
            plot.savefig(filepath)

    def plot_images(self, filepath=None):
        """
        Plot a mosaic of relevant images :
            - the cropped (along z) cubes,
            - the FSF
            - the mask
        and then either show it or save it to a file.

        filepath: string
            If specified, will write the plot to a file instead of showing it.
            The file will be created at the provided absolute or relative path.
            The extension of the file must be either `png` or `pdf`.
        z_crop: None|int
            The maximum and total length of the crop (in pixels) along z,
            centered on the galaxy's z position.
            If you provide zero or an even value (2n),
            the closest bigger odd value will be used (2n+1).
            By default, will not crop.
        """

        self._check_image_filepath(filepath)

        p = self.extract_parameters()
        convolved_cube = self.simulate_convolved(self.data_cube.shape, p)
        clean_cube = self.simulate_clean(self.data_cube.shape, p)
        self._plot_images(self.data_cube, convolved_cube, clean_cube)

        if filepath is None:
            plot.show()
        else:
            plot.savefig(filepath)

    def _plot_images(self, data_cube, convolved_cube, clean_cube):

        fig = plot.figure(1, figsize=(16, 9))
        plot.clf()
        plot.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.05,
                             top=0.95, left=0.05, right=0.95)

        # MEASURE
        sub = fig.add_subplot(2, 3, 1)
        sub.set_title('Measured')
        measured_cube = data_cube[:, :, :]
        measured_image = (measured_cube.sum(0) / measured_cube.shape[0])
        plot.imshow(measured_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # CONVOLVED
        sub = fig.add_subplot(2, 3, 2)
        sub.set_title('Simulation Convolved')
        convolved_image = (convolved_cube.sum(0) / convolved_cube.shape[0])
        plot.imshow(convolved_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # FSF
        sub = fig.add_subplot(2, 3, 3)
        sub.set_title('FSF')
        fsf_image = self.fsf
        plot.imshow(fsf_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # CLEAN
        sub = fig.add_subplot(2, 3, 4)
        sub.set_title('Simulation Clean')
        clean_image = (clean_cube.sum(0) / clean_cube.shape[0])
        plot.imshow(clean_image, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        # MASK
        sub = fig.add_subplot(2, 3, 5)
        sub.set_title('Mask')
        plot.imshow(self.mask, interpolation='nearest', origin='lower')
        plot.xticks(fontsize=8)
        plot.yticks(fontsize=8)
        colorbar = plot.colorbar()
        colorbar.ax.tick_params(labelsize=8)

        return fig

    def _check_image_filepath(self, filepath):
        """
        Make sure that the `filepath` is a valid filepath for images.
        """
        if filepath is not None:
            name, extension = splitext(filepath)
            supported_extensions = ['.png', '.pdf']
            if not extension in supported_extensions:
                raise ValueError("Extension '%s' is not supported, "
                                 "you may use one of %s",
                                 extension, ', '.join(supported_extensions))

    def _compute_error_in_one_step(self, data, params, psf, lsf):
        """
        Debug method to hunt an annoying bug.
        Convolves in one step.
        """
        # Initialize an empty output cube
        shape = np.shape(data)
        sim = np.zeros(shape)
        z = shape[0]

        lsf_fft = None

        # For each spaxel, add its convolved line
        for (y, x) in self.spaxel_iterator():
            # Contribution of the line
            line = self.model.modelize(self, range(0, z), params[y, x])
            # Convolved via the LSF
            if lsf_fft is None:
                line_conv, lsf_fft = convolve_1d(line, lsf)
            else:
                line_conv, _ = convolve_1d(line, lsf_fft, compute_fourier=False)
            # Add it to the simulation
            sim[:, y, x] = line_conv

        # Now convolve everything via the PSF
        from scipy.signal import convolve2d
        for _z in range(0, z):
            sim[_z, :, :] = convolve2d(sim[_z, :, :], psf, mode='same')

        return data - sim



# Some test code for the profiler
# Add the @profile annotation, and run `kernprof -v -l lib/run.py`
# if __name__ == "__main__":
    # from instruments import MUSE
    # cube_test = Cube.from_fits('tests/input/test_cube_01.fits')
    # inst_test = MUSE()
    # run_test = Run(cube_test, inst_test, max_iterations=100)