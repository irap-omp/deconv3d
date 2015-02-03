
## GENERAL PACKAGES ############################################################
import os
from os.path import abspath, dirname, join
import numpy
import unittest
from hyperspectral import HyperspectralCube as Cube

## LOCAL PACKAGES ##############################################################
from deconv3d import Run, MUSE


## ACTUAL TESTS ################################################################
class RunTest(unittest.TestCase):

    longMessage = True

    root_folder = abspath(join(dirname(abspath(__file__)), os.pardir))
    fits_folder = join(root_folder, 'tests/input')

    fits_muse_filename = join(fits_folder, 'subcube_MUSE.fits')

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

        run = Run(cube, inst, max_iterations=100)

        print run.parameters[99]


