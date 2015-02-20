
## GENERAL PACKAGES ############################################################
import os
from os.path import abspath, dirname, join
import numpy as np
import unittest
from hyperspectral import HyperspectralCube as Cube

## LOCAL PACKAGES ##############################################################
from deconv3d import Run, MUSE


## ACTUAL TESTS ################################################################
from lib.line_models import SingleGaussianLineModel


class RunTest(unittest.TestCase):

    longMessage = True

    root_folder = abspath(join(dirname(abspath(__file__)), os.pardir))
    fits_folder = join(root_folder, 'tests/input')

    fits_muse_filename = join(fits_folder, 'test_cube_01.fits')

    def test_init_with_empty_cube(self):
        cube = Cube()
        inst = MUSE()

        self.assertTrue(cube.is_empty())

        with self.assertRaises(ValueError):
            run = Run(cube, inst)

    def test_init_with_muse_cube(self):
        cube = Cube.from_fits(self.fits_muse_filename)
        inst = MUSE()

        self.assertFalse(cube.is_empty())

        run = Run(cube, inst, max_iterations=22222)

        run.plot_images('test_run.png')
        run.save_parameters('test_run.npy')

    def test_initial_parameters(self):
        cube = Cube.from_fits(self.fits_muse_filename)
        inst = MUSE()

        m = SingleGaussianLineModel()
        p = \
            m.min_boundaries(cube) + \
            (m.max_boundaries(cube) - m.min_boundaries(cube)) * \
            np.random.rand(len(m.parameters()))

        run = Run(cube, inst, initial_parameters=p, max_iterations=1)





