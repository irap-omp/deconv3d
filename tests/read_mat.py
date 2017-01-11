# Private and dirty testing of deconv3d
# Dans data14forAntoine
#    FSF             15x15                 1800  double
#    data_noise      24x30x21            120960  double
#    varNoise        24x30x21            120960  double
#
# Dans Parametres_theoriques
#    Parametres_theoriques      24x30x3  17280 double
# premier plan : amplitude,
# second plan position
# troisieme plan dispersion

import datetime

filename = 'input/data14forAntoine.mat'
run_name = 'saves/run_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import sys
sys.path.append('/home/agoutenoir/Code/MUSE')

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Grab data from the matlab file
mat = scipy.io.loadmat(filename)
measured = mat['data_noise']
var = mat['varNoise']
fsf = mat['FSF']

# Rearrange index to be Z/Y/X
# Warning: may be Z/X/Y, to confirm (it does not matter right now)
measured = np.transpose(measured)
var = np.transpose(var)
fsf = np.transpose(fsf)

# test: add more noise
# measured = measured + np.random.randn(*measured.shape)

print "3D Shapes in (Z, Y, X) order :"
print "  Data :    ", measured.shape
print "  Variance :", var.shape
print "2D Shapes in (Y, X) order :"
print "  FSF :     ", fsf.shape


print "Theoretical parameters :"
dat = scipy.io.loadmat('input/Parametres_theoriques.mat')['Parametres_theoriques']

a = np.transpose(dat[:, :, 0])
c = np.transpose(dat[:, :, 1])
w = np.transpose(dat[:, :, 2])

print "  a shape", a.shape
print "  a max", np.amax(a)
print "  a min", np.amin(a)
print "  c shape", c.shape
print "  c max", np.amax(c)
print "  c min", np.amin(c)
print "  w shape", w.shape
print "  w max", np.amax(w)
print "  w min", np.amin(w)

real_params = np.dstack((a, c, w))
print "Expected params shape:", real_params.shape

init_params = real_params.copy()
# init_params[:, :, 0] = 0.  # amplitude = 0


from deconv3d import Run, MUSE, logger, above_percentile, \
    VectorLineSpreadFunction
from hyperspectral import HyperspectralCube as Cube
import logging

logger.setLevel(logging.DEBUG)


# # Flat LSF (not gaussian)
# zsize = measured.shape[0]
# v = np.zeros(zsize)
# # We need to offset the peak by -1,
# # probably because matlab indexation is 1-based
# v[(zsize - 1) / 2 - (zsize % 2 - 1) - 1] = 1.
# lsf = VectorLineSpreadFunction(v)
# instrument = MUSE(fsf_fwhm=0.8841, lsf=lsf)

# Emma's LSF
lsf_fwhm = 2.355 * 0.9345 * 0
instrument = MUSE(fsf_fwhm=0.8841, lsf_fwhm=lsf_fwhm)

cube = instrument.build_cube(measured)

run = Run(
    cube=cube,
    variance=var,
    instrument=instrument,
    # jump_amplitude=[0.0, 0.1, 0.1],
    # initial_parameters=init_params,
    gibbs_apriori_variance=5.,
    # mask=above_percentile(cube, 60),
    max_iterations=20000,
    keep_one_in=10
)

run.save(run_name)


# Compare measure to expected
deconved = run.convolved_cube.data.copy()
expected = run.simulate_convolved(cube.shape, real_params)

# PLOT THE DIFFERENCE
# shown = np.abs(measured - expected).sum(0) / expected.shape[0]
# plt.imshow(shown, interpolation='nearest', origin='lower')
# plt.colorbar()
# plt.show()


# PLOT SOME LINES
# (X, Y)
pixels = [
    (5, 5),
    (10, 10),
    (7, 12),
    (13, 17),
    (11, 9),
]

zsize = measured.shape[0]
plt.subplots_adjust(wspace=0.25, hspace=0.55, bottom=0.05, top=0.95, left=0.05, right=0.95)
for i in range(len(pixels)):
    plt.subplot(len(pixels), 1, i+1)
    pixel = pixels[i]
    x = pixel[0]
    y = pixel[1]
    plt.plot(range(0, zsize), deconved[:, y, x], 'b', label='our model (clean)')
    plt.plot(range(0, zsize), expected[:, y, x], 'r', label='expected (clean)')
    plt.plot(range(0, zsize), measured[:, y, x], 'g', label='measure (convolved)')
    plt.title("X={x}, Y={y}".format(x=x, y=y))
plt.legend()
plt.savefig(run_name + '_lines.png')


# TESTING THE FSF => SAME (close enough)
# emma_fsf = fsf[1:-1, 1:-1]
# print "FSF ARE EQUAL:", np.allclose(emma_fsf, run.fsf)
# import matplotlib.pyplot as plt
# plt.imshow(emma_fsf, interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.imshow(run.fsf, interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.imshow(np.abs(emma_fsf - run.fsf), interpolation='nearest')
# plt.colorbar()
# plt.show()




