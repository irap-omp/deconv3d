# coding=utf-8

import math
import numpy as np
from astropy import units as u

#
# This file contains the FieldSpreadFunction and LineSpreadFunction interfaces
# as well as some basic implementations of these interfaces :
#   - Gaussian FSF
#   - Moffat FSF
#   - Gaussian LSF
#   - MUSE LSF (only if mpdaf module is available)
#
# The instrument will use both 2D FSF and 1D LSF
# to create a 3D PSF with which it will convolve the cubes.
#


## FIELD SPREAD FUNCTIONS ######################################################


class FieldSpreadFunction:
    """
    This is the *interface* all Field Spread Functions (PSF) should implement.
    """

    def as_image(self, for_cube):
        """
        Should return this FSF as a 2D image shaped [for_cube].

        for_cube: HyperspectralCube

        :rtype: np.ndarray
        """
        raise NotImplementedError()


class NoFieldSpreadFunction(FieldSpreadFunction):
    """
    A field spread function that does not spread anything,
    and whose PSF will return the cube unchanged.
    """
    def __init__(self):
        pass

    def as_image(self, for_cube):
        """
        Return the identity PSF, chock-full of ones.
        """
        shape = for_cube.shape[1:]
        return np.ones(shape)


class ImageFieldSpreadFunction(FieldSpreadFunction):
    """
    A custom field spread function using a provided 2D image.
    This class is handy when you already have computed your FSF image externally
    and want to use it as-is.
    """
    def __init__(self, image_2d):
        self.image_2d = image_2d

    def as_image(self, for_cube):
        return self.image_2d

    def __str__(self):
        return """Custom Image PSF"""


class GaussianFieldSpreadFunction(FieldSpreadFunction):
    """
    The default Gaussian Field Spread Function.

    fwhm: float [in arcsec]
        Full Width Half Maximum in arcsec, aka. "seeing".
    pa: float [default is 0.]
        Position Angle, the clockwise rotation from Y of ellipse,
        in angular degrees.
    ba: float [default is 1.0]
        Axis ratio of the ellipsis, b/a ratio (y/x).
    """
    def __init__(self, fwhm=None, pa=0, ba=1.0):
        self.fwhm = fwhm
        self.pa = pa
        self.ba = ba

    def __str__(self):
        return """Gaussian PSF :
    fwhm = {i.fwhm} "
    pa   = {i.pa} °
    ba   = {i.ba}""".format(i=self)

    def as_image(self, for_cube, xo=None, yo=None):
        # Get the FWHM in pixels (we assume the pixels are squares!)
        fwhm = self.fwhm / for_cube.get_step(1).to('arcsec').value
        stddev = fwhm / (2 * math.sqrt(2 * math.log(2)))

        # Guess the shape of the FSF so that we hold the data within three
        # standard deviations (i think, this has to be confimed)
        size = math.ceil(6.*stddev)
        if size % 2 == 0:
            size += 1
        shape = (size, size)

        if xo is None:
            xo = (shape[1] - 1) / 2 - (shape[1] % 2 - 1)
        if yo is None:
            yo = (shape[0] - 1) / 2 - (shape[0] % 2 - 1)

        y, x = np.indices(shape)
        r = self._radius(xo, yo, x, y)

        fsf = np.exp(-0.5 * (r / stddev) ** 2)

        return fsf / fsf.sum()

    def _radius(self, xo, yo, x, y):
        """
        Computes the radii, taking into account the variance and the elliptic
        shape.
        """
        dx = xo - x
        dy = yo - y
        # Rotation matrix around z axis
        # R(90)=[[0,-1],[1,0]] so clock-wise y -> -x & x -> y
        radian_pa = np.radians(self.pa)
        dx_p = dx * np.cos(radian_pa) - dy * np.sin(radian_pa)
        dy_p = dx * np.sin(radian_pa) + dy * np.cos(radian_pa)

        return np.sqrt(dx_p ** 2 + dy_p ** 2 / self.ba ** 2)


class MoffatFieldSpreadFunction(GaussianFieldSpreadFunction):
    """
    The Moffat Field Spread Function.

    fwhm: float [arcsec]
        Moffat's distribution alpha variable : http://en.wikipedia.org/wiki/Moffat_distribution
    beta: float
        Moffat's distribution beta variable : http://en.wikipedia.org/wiki/Moffat_distribution

    pa: float [default is 0.]
        Position Angle, the clockwise rotation from Y of ellipse,
        in angular degrees.
    ba: float [default is 1.0]
        Axis ratio of the ellipsis, b/a ratio (y/x).
    """

    def __init__(self, fwhm=None, beta=None, pa=None, ba=None):
        self.fwhm = fwhm
        self.beta = beta
        GaussianFieldSpreadFunction.__init__(self, fwhm, pa, ba)

    def __str__(self):
        return """Moffat PSF :
  fwhm         = {i.fwhm} "
  beta         = {i.beta} 
  pa           = {i.pa} °
  ba           = {i.ba}""".format(i=self)

    def as_image(self, for_cube, xo=None, yo=None):
        # Get the FWHM in pixels (we assume the pixels are squares!)
        fwhm = self.fwhm / for_cube.get_step(1).to('arcsec').value
       
        shape = for_cube.shape[1:]

        if xo is None:
            xo = (shape[1] - 1) / 2 - (shape[1] % 2 - 1)
        if yo is None:
            yo = (shape[0] - 1) / 2 - (shape[0] % 2 - 1)

        y, x = np.indices(shape)
        r = self._radius(xo, yo, x, y)

        alpha = fwhm / (2.*np.sqrt(2.**(1./beta)-1) ) 
        beta = self.beta
        psf = (1. + (r / alpha) ** 2) ** (-beta)

        return psf / psf.sum()


## LINE SPREAD FUNCTIONS #######################################################


class LineSpreadFunction:
    """
    This is the interface all Line Spread Functions (LSF) should implement.
    """

    def as_vector(self, for_cube):
        """
        Should return this LSF as a 1D vector shaped `for_cube`, meaning that
        it should have the length of the cube's spectral axis.

        for_cube: HyperspectralCube

        :rtype: ndarray
        """
        raise NotImplementedError()


class VectorLineSpreadFunction(LineSpreadFunction):
    """
    A custom line spread function using a provided 1D `vector`
    that should have the same length as the cube's (z).
    This class is handy when you already have computed your vector externally
    and want to use it as a LSF.
    Should be centered around ``zo = (zsize - 1) / 2 - (zsize % 2 - 1)``.
    """

    def __init__(self, vector):
        self.vector = vector

    def as_vector(self, for_cube):
        return self.vector

    def __str__(self):
        return """Custom Vector LSF"""


class GaussianLineSpreadFunction(LineSpreadFunction):
    """
    A line spread function that spreads as a gaussian.
    We assume the centroid is in the middle.

    fwhm: float
        Full Width Half Maximum, in µm.
    """
    def __init__(self, fwhm):
        self.fwhm = fwhm

    def __str__(self):
        return """Gaussian LSF : fwhm = {i.fwhm} µm \n""".format(i=self)

    def as_vector(self, for_cube):
        # Std deviation from FWHM
        sigma = self.fwhm / 2.35482 / for_cube.get_step(0).to(u.um).value
        # Resulting vector length is the spectral depth of the cube
        depth = for_cube.shape[0]
        # Assymmetric range around 0
        z_center_index = (depth - 1) / 2 - (depth % 2 - 1)
        z_range = np.arange(depth) - z_center_index
        # Compute gaussian (we assume peak is at 0, ie. µ=0)
        if sigma == 0:
            lsf_1d = np.zeros(depth)
            lsf_1d[z_center_index] = 1.
        else:
            # fixme µ is 0, check if that's is a problem
            lsf_1d = self.gaussian(z_range, 0, sigma)
        # Normalize and serve
        return lsf_1d / lsf_1d.sum()

    @staticmethod
    def gaussian(x, mu, sigma):
        """
        Non-normalized gaussian function.

        x : float|numpy.ndarray
            Input value(s)
        mu : float
            Position of the peak on the x-axis
        sigma : float
            Standard deviation

        :rtype: Float value(s) after transformation, of the same shape as input x.
        """
        return np.exp((x - mu) ** 2 / (-2. * sigma ** 2))


class MUSELineSpreadFunction(LineSpreadFunction):
    """
    A line spread function that uses MPDAF's LSF.
    See http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/user_manual_PSF.html

    .. warning::
        This requires the ``mpdaf`` module, which only works for odd arrays.
    model: string
        See ``mpdaf.MUSE.LSF``'s ``type`` parameter.
        Note: `type` is a restricted keyword, so we use `model` instead.
    """
    def __init__(self, model="qsim_v1"):
        self.model = model
        try:
            from mpdaf.MUSE import LSF
        except ImportError:
            raise ImportError("You need to install the mpdaf module "
                              "to use MUSELineSpreadFunction.")
        self.lsf = LSF(type=self.model)

    def __str__(self):
        return """MUSE LSF : model = '{i.model}'""".format(i=self)

    def as_vector(self, cube):
        # Resulting vector shape
        depth = cube.shape[0]
        odd_depth = depth if depth % 2 == 1 else depth+1
        # Get LSF 1D from MPDAF
        wavelength_aa = cube.z_central * 1e4  # units from microns to AA
        z_step_aa = cube.z_step * 1e4
        lsf_1d = self.lsf.get_LSF(lbda=wavelength_aa, step=z_step_aa, size=odd_depth)
        # That LSF is of an odd depth, truncate it if necessary
        if depth % 2 == 0:
            lsf_1d = lsf_1d[:-1]
        # Normalize and serve
        return lsf_1d / lsf_1d.sum()
