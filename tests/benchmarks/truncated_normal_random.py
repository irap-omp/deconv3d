import benchmark

from numpy.core.numeric import Infinity
from scipy.stats import truncnorm

from lib.rtnorm import rtnorm

# Benchmark Report
# ================
#
# BenchmarkTruncatedNormalRandom
# ------------------------------
#
#   name | rank |  runs |     mean |        sd | timesBaseline
# -------|------|-------|----------|-----------|--------------
# rtnorm |    1 | 1e+04 | 1.48e-05 | 2.802e-06 |           1.0
#  scipy |    2 | 1e+04 | 4.45e-05 |  0.000194 | 3.00621002978
#
# Each of the above 20000 runs were run in random, non-consecutive order by
# `benchmark` v0.1.5 (http://jspi.es/benchmark) with Python 2.7.6
# Linux-3.13.0-55-generic-x86_64 on 2015-06-25 11:09:05.


class BenchmarkTruncatedNormalRandom(benchmark.Benchmark):

    each = 10000  # configure number of runs

    def setUp(self):
        # Can also specify tearDown, eachSetUp, and eachTearDown
        pass

    def test_rtnorm(self):
        b = rtnorm(0, Infinity)

    def test_scipy(self):
        b = truncnorm.rvs(0, Infinity)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")

