from astropy.units import Unit
from hyperspectral import HyperspectralCube, Axis
import numpy as np

from convolution import convolve_3d_same
from spread_functions import \
    LineSpreadFunction, FieldSpreadFunction, \
    GaussianLineSpreadFunction, GaussianFieldSpreadFunction


class Instrument:
    """
    This is a generic instrument class to use directly or to extend.
    Usually, you'll use a child class of this, like `MUSE`.

    fsf: FieldSpreadFunction
        The 2D Field Spread Function instance to use in the convolution.
    lsf: LineSpreadFunction
        The 1D Line Spread Function instance to use in the convolution.
        Will be used in the convolution to spread the FSF 2D image through the
        spectral third axis, z.
    """

    def __init__(self, lsf, fsf):

        # LINE SPREAD FUNCTION
        if not isinstance(lsf, LineSpreadFunction):
            raise ValueError("lsf should be an instance of LineSpreadFunction")
        self.lsf = lsf

        # FIELD SPREAD FUNCTION
        if not isinstance(fsf, FieldSpreadFunction):
            raise ValueError("fsf should be an instance of FieldSpreadFunction")
        self.fsf = fsf

        # MEMOIZED POINT SPREAD FUNCTION (AND ITS FFT)
        self.psf = None  # 3D ndarray, product of 2D FSF and 1D LSF
        self.psf_fft = None  # Fast Fourier Transform of above (also a ndarray)

    def convolve(self, cube):
        """
        Convolve the provided data cube using the 3D Point Spread Function
        made from the convolution of the 2D FSF and 1D LSF.
        Should transform the input cube and then return it.

        .. note::
            The 3D Point Spread Function and its Fast Fourier Transform are
            memoized for optimization, so if you change the instrument's
            parameters after a first run, the PSF will not reflect your changes.
            Delete ``psf`` and ``psf_fft`` to clear the memory.

        cube: HyperspectralCube
        """
        # PSF's FFT is memoized for optimization
        if self.psf_fft is None:
            # Apply LSF to FSF
            self.psf = Instrument.compute_psf(
                self.fsf.as_image(cube),
                self.lsf.as_vector(cube),
                cube
            )
            cube_convolved, self.psf_fft = convolve_3d_same(
                cube.data,
                self.psf,
                compute_fourier=True
            )
        else:
            cube_convolved, __ = convolve_3d_same(
                cube.data,
                self.psf_fft,
                compute_fourier=False
            )

        cube.data = cube_convolved

        return cube

    @staticmethod
    def compute_psf(fsf2d, lsf1d, for_cube):
        """
        Apply the 1D LSF to provided 2D PSF image,
        and return the resulting 3D PSF cube.
        The LSF extrudes the FSF image along the z-axis (the spectral one).
        """
        # Get cube shape (z,y,x)
        shape = for_cube.shape
        # Resize in 3D (duplicate along x and y)
        lsf3d = np.repeat(lsf1d, shape[1] * shape[2]).reshape(shape)
        # Extrude PSF image along z
        psf3d = np.resize(fsf2d, shape)
        # Apply LSF
        psf3d = lsf3d * psf3d

        return psf3d

    def __str__(self):
        return """
fsf = {i.fsf}
lsf = {i.lsf}
""".format(i=self)


class MUSE(Instrument):
    """
    The MUSE instrument.
    """

    def __init__(self, lsf=None, fsf=None,
                 lsf_fwhm=0.0002675,
                 fsf_fwhm=1.0, fsf_pa=0., fsf_ba=1.0):

        if lsf is None:
            lsf = GaussianLineSpreadFunction(fwhm=lsf_fwhm)
        if fsf is None:
            fsf = GaussianFieldSpreadFunction(fwhm=fsf_fwhm,
                                              pa=fsf_pa,
                                              ba=fsf_ba)

        Instrument.__init__(self, lsf=lsf, fsf=fsf)

    def build_cube(self, data):

        meta = {
            'CDELT1': 5.5555555555555e-05,
            'CDELT2': 5.5555555555555e-05,
            'CDELT3': 1.25,
            'CRVAL1': 1.0,
            'CRVAL2': 1.0,
            'CRVAL3': 6564.0,
            'CRPIX1': 1.0,
            'CRPIX2': 1.0,
            'CRPIX3': 15.0,
            'CUNIT1': 'deg     ',
            'CUNIT2': 'deg     ',
            'CUNIT3': 'Angstrom',
            'CTYPE1': 'RA---TAN',
            'CTYPE2': 'DEC--TAN',
        }
        x = Axis('x', meta['CRVAL1'], meta['CDELT1'], Unit(meta['CUNIT1']))
        y = Axis('y', meta['CRVAL2'], meta['CDELT2'], Unit(meta['CUNIT2']))
        z = Axis('z', meta['CRVAL3'], meta['CDELT3'], Unit(meta['CUNIT3']))

        from astropy.io import fits
        header = fits.Header()
        for name in meta:
            header[name] = meta[name]

        return HyperspectralCube(data=data, meta={'fits': header}, x=x, y=y, z=z)
