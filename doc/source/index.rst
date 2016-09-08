.. Deconv3D documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive, whatever that means. o.O

Welcome to Deconv3D's documentation!
====================================

.. include:: common.rst

`Deconv3D <http://deconv3d.irap.omp.eu>`_ is a tool to extract Galaxy
Parameters and Kinematics from 3-Dimensional data, using reverse deconvolution
with the Bayesian analysis procedure Markov Chain Monte Carlo.


Usage
*****

The simplest way to use deconv3d is as follows: ::

        import deconv3d
        run = deconv3d.Run(
            cube='my_truncated_muse_cube.fits',
            instrument=deconv3d.MUSE(),
        )


A full parameterized usage looks like this: ::

        import deconv3d
        run = deconv3d.Run(
            cube='my_truncated_muse_cube.fits',
            instrument=deconv3d.MUSE(),
            mask=None,
            variance=None,
            model=SingleGaussianLineModel,
            initial_parameters=None,
            jump_amplitude=0.1,
            gibbs_apriori_variance=None,
            max_iterations=100000,
            keep_one_in=1,
            write_every=10000,
            min_acceptance_rate=0.01
        )


Masking
-------

Performance improves drastically if you mask the voxels you want to run the fitting on.
The mask should be a 2D numpy.ndarray of the spatial size of the cube.

The runner will only try to deconvolve the spaxels where this
mask is set to 1. The default mask is filled with ones, transparency.
This mask is also automatically opacified where there are NaN values in
the input cube.

Authors
-------

- Bouché N., Carfanta H., Schroetter I., Michel-Dansac L., Contini T., ApJ, tbd "A Bayesian parametric tool for extracting kinematics from 3D data"
- Contacts: `Herve.Carfantan@irap.omp.eu <mailto:Herve.Carfantan@irap.omp.eu>`_


Table of Contents
-----------------

.. include:: table_of_contents.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
