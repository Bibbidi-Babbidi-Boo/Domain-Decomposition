import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial
from skimage.draw import polygon_perimeter, polygon
from scipy.interpolate import interp1d
import imageio
import io


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas



size = 64
map_shape = (size,size)
variance_scale = 100
numDistribs = 50
infoMap = np.zeros(map_shape, dtype = np.float32)
        
        

# ================================ HELPER FUNCTIONS ===================================
def bivariateGaussianMatrix():
    """
    Generates a single (not mixture) bivariate normal distribution, with independent axes
    Input:
        - map_shape: tuple (x, y) representing the map size
        - variance_scale: scales a the random variance for the x and y components
    Output:
        - distribution_matrix: 2D array containing single normal bivariate distribution
        - mean of the bivariateGaussain
    """
    x, y = np.mgrid[0:map_shape[0], 0: map_shape[1]]
    pos = np.dstack((x, y))
    mean2D = map_shape[0] * np.random.rand(1, 2)[0]
    cov2D = np.zeros((2, 2))
    cov2D[([0, 1], [0, 1])] = variance_scale * np.random.rand(1, 2)[0]
    rv = multivariate_normal(mean2D, cov2D)
    return rv.pdf(pos), mean2D

def createInformationMap():
    '''
    Generate a mixture of Gaussians, scaled so that the maximum value is close to 1.
    Input:
        - numDistribs: Number of distributions in the information map -> number of knots
        - map_shape: Shape of the map
        - variance_scale: Scaling value for the bivariateGaussianMatrix function
    Output:
        - infoMap: Mixture of Gaussians, plus a baseline probability for the rest of the cells
        - scalar: Scaling factor used to make maximum value = 1
        - means: Returns locations of the means of the distributions that make up the infoMap
    '''
    # Generate multiple bivariate Gaussians
    infoMaps = []
    means = []
    for _ in range(numDistribs):
        infoMap, meanDist = bivariateGaussianMatrix()
        infoMaps.append(infoMap)
        means.append(meanDist)
    infoMaps = np.asarray(infoMaps)
    means = np.asarray(means)


    # Mixture of Gaussians + epsilon probability for "empty" space
    infoMap  = np.mean(infoMaps, axis=0)

    # Scale so that the maximum value in the map is 1
    maxInfo = np.max(infoMap)
    scalar = 1/maxInfo
#     scalar = 0.5/maxInfo * np.random.rand() + 0.5/maxInfo
    infoMap = infoMap * scalar

    return infoMap, scalar, means

def plot2D(array, cmap="Greys"):
    '''
    Plot a 2D array with a colorbar
    Input:
        - array: 2D array to plot
    '''
    plt.imshow(array,cmap=cmap, interpolation="none")
    plt.colorbar()
    plt.show()


infoMap, _, _ = createInformationMap()
COLLECTIBLE_COLOR       = 128 # GREY for collectible region
NON_COLLECTIBLE_COLOR   = 0 # White for non-collectible region
LINE_COLOR              = 255   # Black for line seperating region

cuttingDiagram = NON_COLLECTIBLE_COLOR * np.ones(map_shape)
cuttingDiagram[40:50, 10:20] = COLLECTIBLE_COLOR
figs = []

for i in range(size):
    cuttingDiagram[i,i] = LINE_COLOR
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.imshow(infoMap, cmap="viridis", interpolation="none")
    ax.imshow(cuttingDiagram, cmap="Greys", interpolation="none", alpha=0.3)
    # Force a draw so we can grab the pixel buffer
    canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(canvas.renderer.buffer_rgba())
    figs.append(X)

imageio.mimwrite("testImage.gif",figs)



# plot2D(cuttingDiagram, "Greys")

# plt.imshow(cuttingDiagram, cmap="Greys", interpolation="none")
# plt.imshow(infoMap, cmap="viridis", interpolation="none", alpha=0.5)
# plt.show()







# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.imshow(cuttingDiagram, cmap="Greys", interpolation="none")
# # ax.imshow(infoMap, cmap="viridis", interpolation="none")
# # ax.imshow(cuttingDiagram, cmap="Greys", interpolation="none", alpha=0.3)
# plt.show()












# # Create a figure that pyplot does not know about.
# fig = Figure()
# # attach a non-interactive Agg canvas to the figure
# # (as a side-effect of the ``__init__``)
# canvas = FigureCanvas(fig)
# ax = fig.subplots()
# ax.imshow(infoMap, cmap="viridis", interpolation="none")
# ax.imshow(cuttingDiagram, cmap="Greys", interpolation="none", alpha=0.3)
# # Force a draw so we can grab the pixel buffer
# canvas.draw()
# # grab the pixel buffer and dump it into a numpy array
# X = np.array(canvas.renderer.buffer_rgba())

# # now display the array X as an Axes in a new figure
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, frameon=False)
# ax2.imshow(X)
# plt.show()

# imageio.imwrite("testImage.png",X)
# print("Done saving")

















