import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

points = np.array([[2, 0], [0, 1], [150, 0], [0, 150]])
vor = Voronoi(points)
voronoi_plot_2d(vor)
ax = plt.axes()
ax.set_xlim([-0.5,5])
ax.set_ylim([-0.5,5])
plt.show()
