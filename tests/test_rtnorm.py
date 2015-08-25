
# This should plot a histogram looking like a gaussian
# ... It does.

## CONFIGURATION (play with different values)

samples = int(1e6)

minimum = 0.
maximum = 15.
center = 7.
stddev = 5.


## VARIABLES FROM RANDOM TRUNCATED NORMAL DISTRIBUTION

from lib.rtnorm import rtnorm

variables = rtnorm(minimum, maximum, mu=center, sigma=stddev, size=samples)


## PLOT THEIR HISTOGRAM

import matplotlib.pyplot as plot

plot.hist(variables, bins=400)
plot.show()

