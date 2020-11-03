from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

points = np.random.rand(6,2)
points = np.append(points, [[0,0], [0,1], [1,0], [1,1]], axis=0)
print(points)
tri = Delaunay(points)

# print(tri.simplices)
for vert in tri.simplices:
    v0 = points[vert[0]]
    v1 = points[vert[1]]
    v2 = points[vert[2]]
    # print(v0, v1, v2)
    plt.xlim(-0.01, 1.01), plt.ylim(-0.01, 1.01)
    plt.plot([v0[0],v1[0]], [v0[1],v1[1]], color='r', marker='o', markerfacecolor='b')
    plt.plot([v1[0],v2[0]], [v1[1],v2[1]], color='r', marker='o', markerfacecolor='b')
    plt.plot([v0[0],v2[0]], [v0[1],v2[1]], color='r', marker='o', markerfacecolor='b')

plt.show()
