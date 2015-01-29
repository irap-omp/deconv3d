# coding=utf-8
import numpy as np
import logging

# Try to use the faster https://pypi.python.org/pypi/pyFFTW
try:
    from pyfftw.interfaces.numpy_fft import rfftn, irfftn, fftshift
except ImportError:
    logging.info("Convolution: install pyFFTW for better performances.")
    from numpy.fft import rfftn, irfftn, fftshift


def convolve_3d_same(cube, psf, compute_fourier=True):
    """
    Convolve a 3D cube with the given PSF.
    PSF can be the PSF data or its (Fast) Fourier Transform.

    This convolution has edge effects.

    cube: The cube we want to convolve
    psf: The Point Spread Function or its Fast Fourier Transform
    compute_fourier: bool
        If True, then it will compute the FFT transform of the PSF.
        If False, it will assume that the given PSF is a FFT already.
    """

    # Pad to power of 2
    padded_cube, cube_slices = pad_cube(cube, axes=[0, 1, 2])

    size = np.array(np.shape(padded_cube)[slice(0, 3)])

    if compute_fourier:
        padded_psf, psf_slices = pad_cube(psf, axes=[0, 1, 2])
        fft_psf = rfftn(padded_psf, s=size, axes=[0, 1, 2])
    else:
        fft_psf = psf

    fft_img = rfftn(padded_cube, s=size, axes=[0, 1, 2])

    # Convolution
    fft_cube = np.real(fftshift(irfftn(fft_img * fft_psf, s=size, axes=[0, 1, 2]), axes=[0, 1, 2]))

    # Remove padding
    cube_conv = fft_cube[cube_slices]

    return cube_conv, fft_psf


def pad_cube(cube, axes=None):
    """
    Computes padding needed for a cube to make sure it has a power of 2 shape
    along the dimensions of passed axes (by default [0,1]).
    Returns the padded cube and cube slices,
    which are the indices of the actual data in the padded cube.
    """

    if axes is None:
        axes = [0, 1]

    # Compute padding size for each axis
    old_shape = np.shape(cube)
    new_shape = np.array(old_shape)
    for axis in axes:
        zdim = cube.shape[axis]
        s = np.binary_repr(zdim - 1)
        s = s[:-1] + '0'
        new_shape[axis] = 2 ** len(s)

    cube_padded = np.zeros(new_shape)
    cube_slices = np.empty(len(old_shape), slice).tolist()

    for i, v in enumerate(old_shape):
        cube_slices[i] = slice(0, old_shape[i])

    for axis in axes:
        diff = new_shape[axis] - old_shape[axis]
        if diff & 1:
            half = diff / 2 + 1
        else:
            half = diff / 2
        cube_slices[axis] = slice(half, old_shape[axis] + half)

    # Copy cube contents into padded cube
    cube_padded[cube_slices] = cube.copy()

    return cube_padded, cube_slices


def convolve_1d(data, psf, compute_fourier=True, axis=0):
    """
    Convolve data with PSF only along one dimension specified by axis (default: 0)
    PSF can be the PSF data or its Fourier transform
    if compute_fourier then compute the fft transform of the PSF.
    if False then assumes that the fft is given.
    """

    axis = np.array([axis])

    # Compute needed padding
    cubep, boxcube = padding(data, axes=axis)

    # Get the size of the axis
    size = np.array(np.shape(cubep)[slice(axis, axis + 1)])

    if compute_fourier:
        psfp, boxpsf = padding(psf, axes=axis)
        fftpsf = np.fft.rfftn(psfp, s=size, axes=axis)
    else:
        fftpsf = psf

    fftimg = np.fft.rfftn(cubep, s=size, axes=axis)

    # Convolution
    fft = np.fft.fftshift(np.fft.irfftn(fftimg * fftpsf, s=size, axes=axis), axes=axis).real

    # Remove padding
    cube_conv = fft[boxcube]

    return cube_conv, fftpsf


def padding(cube, axes=None):
    """
    Computes padding needed for a cube to make sure it has
    a power of 2 shape along dimensions of passed axes (default [0,1])
    Returns padded cube and cube slices,
    which are the indices of the actual data in the padded cube.
    """

    if axes is None:
        axes = [0, 1]

    # Compute padding size for each axis
    old_shape = np.shape(cube)
    new_shape = np.array(old_shape)
    for axis in axes:
        zdim = cube.shape[axis]
        s = np.binary_repr(zdim - 1)
        s = s[:-1] + '0'
        new_shape[axis] = 2 ** len(s)

    cube_padded = np.zeros(new_shape)
    cube_slices = np.empty(len(old_shape), slice).tolist()

    for i, v in enumerate(old_shape):
        cube_slices[i] = slice(0, old_shape[i])

    for axis in axes:
        diff = new_shape[axis] - old_shape[axis]
        if diff & 1:
            half = diff / 2 + 1
        else:
            half = diff / 2
        cube_slices[axis] = slice(half, old_shape[axis] + half)

    # Copy cube contents into padded cube
    cube_padded[cube_slices] = cube.copy()

    return cube_padded, cube_slices