
import unittest
import matplotlib.pyplot as plot
import numpy as np

import sys
sys.path.append('.')  # T_T

from lib.rtnorm import rtnorm


class RtnormTest(unittest.TestCase):

    longMessage = True

    def test_histogram(self):
        """
        This should plot a histogram looking like a gaussian
        ... It does.
        """
        # CONFIGURATION (play with different values)
        samples = int(1e6)
        minimum = 1.
        maximum = 17.
        center = 7.
        stddev = 5.

        # VARIABLES FROM RANDOM TRUNCATED NORMAL DISTRIBUTION
        variables = rtnorm(minimum, maximum, mu=center, sigma=stddev,
                           size=samples)

        # PLOT THEIR HISTOGRAM
        plot.hist(variables, bins=400)
        plot.show()

    def test_sanity(self):
        """
        Simple sanity test for the random truncated normal distribution.
        """
        from sys import maxint

        # Generate an array with one number
        r = rtnorm(0, maxint)

        self.assertTrue(isinstance(r, np.ndarray))
        self.assertTrue(len(r) == 1)
        self.assertTrue((r > 0).all())
        self.assertTrue((r < maxint).all())

        # Generate an array with 42 numbers
        r = rtnorm(0, maxint, size=42)

        self.assertTrue(isinstance(r, np.ndarray))
        self.assertTrue(len(r) == 42)
        self.assertTrue((r > 0).all())
        self.assertTrue((r < maxint).all())
