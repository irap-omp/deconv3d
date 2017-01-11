
# Analyse a run created by read_mat.py

# run_name = 'saves/run_20161207_110028'
run_name = 'saves/run_20161207_120059'

###############################################################################

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# X Y P
expected_params = scipy.io.loadmat('input/Parametres_theoriques.mat')['Parametres_theoriques']
expected_params = np.rollaxis(expected_params, 1)
expected_a = expected_params[:, :, 0]
import math

# print "Amplitude for x=18 y=7 :", expected_a[7][18]
# print "Amplitude for x=7  y=18 :", expected_a[18][7]

chain = np.load(run_name+'_chain.npy')
# Y X P
actual_params = np.load(run_name+'_parameters.npy')

# Center
cy = int(math.floor(expected_a.shape[0] / 2.))
cx = int(math.floor(expected_a.shape[1] / 2.))
# (X, Y)
pixels = [
    (6, 6),
    (cx, cy),
    (10, 10),
    (7, 12),
    (13, 17),
    (11, 9),
]

plt.subplots_adjust(wspace=0.25, hspace=0.55, bottom=0.05, top=0.95, left=0.05, right=0.95)
for i in range(len(pixels)):
    plt.subplot(len(pixels), 1, i+1)
    pixel = pixels[i]
    x = pixel[0]
    y = pixel[1]
    chain_a = np.transpose(chain[:, y, x, :])[0]
    plt.plot(chain_a)
    plt.axhline(expected_a[y][x], color='r', label='Expected')
    plt.axhline(actual_params[y][x][0], color='g', label='Actual')
    plt.title("X={x}, Y={y}".format(x=x, y=y))
plt.legend()
plt.savefig(run_name+'_lines2.png')
plt.show()


fig = plt.figure(1, figsize=(16, 9))
plt.clf()
plt.subplots_adjust(wspace=0.15, hspace=0.15, bottom=0.05,
                    top=0.95, left=0.05, right=0.95)

def plot_image(title, img, a, b, c, colorbar_mappable=None, cmap=None, same_as=None):
    sub = fig.add_subplot(a, b, c)
    sub.set_title(title)
    vmin = None
    vmax = None
    if same_as is not None:
        vmin = same_as.colorbar.vmin
        vmax = same_as.colorbar.vmax
        print "vmin", vmin, "vmax", vmax
    im = plt.imshow(
        img, interpolation='nearest', origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    colorbar = plt.colorbar(mappable=colorbar_mappable)
    colorbar.ax.tick_params(labelsize=8)
    return im

grid_h = 2
grid_w = 3

im_a = plot_image("Expected A", expected_params[:, :, 0], grid_h, grid_w, 1)
im_c = plot_image("Expected C", expected_params[:, :, 1], grid_h, grid_w, 2)
im_w = plot_image("Expected W", expected_params[:, :, 2], grid_h, grid_w, 3)

plot_image("Actual A", actual_params[:, :, 0], grid_h, grid_w, 4, same_as=im_a)
plot_image("Actual C", actual_params[:, :, 1]+1, grid_h, grid_w, 5, same_as=im_c)
plot_image("Actual W", actual_params[:, :, 2], grid_h, grid_w, 6, same_as=im_w)

plt.savefig(run_name+'_analysis.png')
plt.show()

print("Done")

# plot.subplot2grid((rows, cols), (0, i % cols))
# plot.plot(chain_transposed[i])
# if bound:
#     plot.ylim(bmin[i], bmax[i])
# plot.title(names[i], fontsize='small')