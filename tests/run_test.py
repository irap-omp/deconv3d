
## GENERAL PACKAGES ############################################################
from os import pardir, remove
from os.path import abspath, dirname, join, isfile
import numpy as np
import unittest
from hyperspectral import HyperspectralCube as Cube

## LOCAL PACKAGES ##############################################################
from deconv3d import Run, MUSE


## ACTUAL TESTS ################################################################
from lib.line_models import SingleGaussianLineModel


class RunTest(unittest.TestCase):

    longMessage = True

    root_folder = abspath(join(dirname(abspath(__file__)), pardir))
    fits_folder = join(root_folder, 'tests/input')

    fits_muse_filename = join(fits_folder, 'test_cube_01.fits')

    data_galpak1_filename = join(fits_folder, 'GalPaK_cube_1101_size4.08_flux1e-16_incl60_vmax199_disp80_seeing1.0_PAm50.fits')
    mask_galpak1_filename = join(fits_folder, 'GalPaK_cube_1101_size4.08_flux1e-16_incl60_vmax199_disp80_seeing1.0_PAm50_mask.fits')

    data_galpak2_filename = join(fits_folder, 'input_snrx10GalPaK_cube_1101_size4.08_flux1e-16_incl60_vmax199_disp80_seeing1.00_PA_m50.fits')

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

        run = Run(cube, inst,
                  initial_parameters='run_001.npy',
                  max_iterations=22222)

        run.save('run_003', clobber=True)

    def test_save(self):
        cube = Cube.from_fits(self.fits_muse_filename)
        inst = MUSE()

        run = Run(cube, inst, max_iterations=2)
        run.save('test', clobber=True)

        self.assertTrue(isfile('test_parameters.npy'))
        remove('test_parameters.npy')
        self.assertTrue(isfile('test_images.png'))
        remove('test_images.png')
        self.assertTrue(isfile('test_convolved_cube.fits'))
        remove('test_convolved_cube.fits')
        self.assertTrue(isfile('test_clean_cube.fits'))
        remove('test_clean_cube.fits')

    def test_initial_parameters(self):
        cube = Cube.from_fits(self.fits_muse_filename)
        inst = MUSE()

        m = SingleGaussianLineModel()
        nump = len(m.parameters())
        minp = np.array(m.min_boundaries(cube))
        maxp = np.array(m.max_boundaries(cube))

        ## FROM A 1D ARRAY
        p1d = minp + (maxp - minp) * np.random.rand(nump)

        run = Run(cube, inst, initial_parameters=p1d, max_iterations=1)
        # Note: p1d is broacasted on each spaxel in this assertion
        self.assertTrue((run.extract_parameters() == p1d).all())

        ## FROM A 3D ARRAY
        p3d = np.resize(p1d, (cube.shape[1], cube.shape[0], nump))
        run = Run(cube, inst, initial_parameters=p1d, max_iterations=1)
        self.assertTrue((run.extract_parameters() == p3d).all())

        ## FROM A NPY FILE
        np.save('test.npy', p3d)
        run = Run(cube, inst, initial_parameters='test.npy', max_iterations=1)
        self.assertTrue((run.extract_parameters() == p3d).all())
        remove('test.npy')

    def test_plot_chain(self):
        cube = Cube.from_fits(self.fits_muse_filename)
        inst = MUSE()

        run = Run(cube, inst, max_iterations=666)

        run.plot_chain()

    def test_with_galpak1_data(self):
        cube = Cube.from_fits(self.data_galpak1_filename)
        inst = MUSE()

        self.assertFalse(cube.is_empty())

        run = Run(
            cube, inst,
            mask=self.mask_galpak1_filename,
            max_iterations=44444
        )

        run.save('run_galpak1', clobber=True)

    def test_with_galpak2_data(self):
        # FIX THESE DAMN HEADERS
        from astropy.io.fits import setval
        setval(self.data_galpak2_filename, keyword='CDELT1', value=5.5555555555555e-05, ext=0)
        setval(self.data_galpak2_filename, keyword='CDELT2', value=5.5555555555555e-05, ext=0)
        setval(self.data_galpak2_filename, keyword='CRVAL1', value=1.0, ext=0)
        setval(self.data_galpak2_filename, keyword='CRVAL2', value=1.0, ext=0)
        setval(self.data_galpak2_filename, keyword='CRPIX1', value=1.0, ext=0)
        setval(self.data_galpak2_filename, keyword='CRPIX2', value=1.0, ext=0)
        setval(self.data_galpak2_filename, keyword='CUNIT1', value='deg     ', ext=0)
        setval(self.data_galpak2_filename, keyword='CUNIT2', value='deg     ', ext=0)
        setval(self.data_galpak2_filename, keyword='CTYPE1', value='RA---TAN', ext=0)
        setval(self.data_galpak2_filename, keyword='CTYPE2', value='DEC--TAN', ext=0)
        ########################

        cube = Cube.from_fits(self.data_galpak2_filename)
        inst = MUSE()

        self.assertFalse(cube.is_empty())

        run = Run(
            cube, inst,
            #mask=self.mask_galpak1_filename,
            max_iterations=100000
        )

        run.save('run_galpak2', clobber=True)

    def test_numpy_extrude(self):
        a2d = np.array([[0, 1],
                        [2, 0]])
        a1d = np.array([1, 2, 3])

        b3d = np.array([
            [[0, 1],
             [2, 0]],
            [[0, 2],
             [4, 0]],
            [[0, 3],
             [6, 0]],
        ])

        # Yep. Not trivial, but fast
        a3d = a2d * a1d[:, np.newaxis][:, np.newaxis]

        self.assertEqual(0, np.sum(a3d-b3d))