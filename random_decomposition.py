import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.stats import multivariate_normal
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial
from skimage.draw import polygon_perimeter, polygon

class Region:
    def __init__(self, area = 0, info = 0, rr = None, cc = None, coll = False, vertices = [], index = None, label = None):
        self.area = area
        self.info = info
        self.rr = rr # Row indices for this region
        self.cc = cc # Col indices for this region
        self.coll = coll # Boolean, whether a region is collectible or not
        self.label = label
        self.num_agents = 0 # TODO

class Decompose_and_Search():
    def __init__(self):
        pass
        self.map_shape = (64, 64)
        self.num_agents = 8
        self.descriptiveDiagram = np.zeros(self.map_shape) 
        self.agentMap = np.zeros(self.map_shape)

        self.num_regions = np.random.randint(4, self.num_agents + 1)

        self.current_regions = []
        self.current_agents = []
        self.generators = None
        self.generatePoints()

    def generatePoints(self):
        points = np.random.randint(0, self.map_shape[0], (self.num_regions*2,2))
        self.generators = np.unique(points, axis=0)[:self.num_regions]

    def plot2D(self, array):
        '''
        Plot a 2D array with a colorbar
        Input:
            - array: 2D array to plot
        '''
        plt.imshow(array)
        plt.colorbar()
        plt.show()


    def reset(self):
        self.descriptiveDiagram = np.zeros(self.map_shape) 
        self.agentMap = np.zeros(self.map_shape)
        self.num_regions = np.random.randint(4, self.num_agents + 1)
        self.generatePoints()
        self.current_regions = []
        self.current_agents = []

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """Reconstruct infinite Voronoi regions in a
        2D diagram to finite regions.
        Source:
        [https://stackoverflow.com/a/20678647/1595060](https://stackoverflow.com/a/20678647/1595060)
        """
        # print("Running voronoi_finite_polygons_2d")
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        new_ridge_vertices = []
        vor_ridge_vertices = vor.ridge_vertices
        for p in vor_ridge_vertices:
            if all(i >= 0 for i in p):
                new_ridge_vertices.append(p)

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                      vor.ridge_vertices):
            all_ridges.setdefault(
                p1, []).append((p2, v1, v2))
            all_ridges.setdefault(
                p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region): # p1 is a counter (0,1, etc), region is the region "name (label)" for the p1th point
            vertices = vor.regions[region] # Returns the vertices that corresponds to the "region_th" region. Region starts at 1
            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue
            # reconstruct a non-finite region
            ridges = all_ridges[p1] # Get a list of all ridges surrounding that point [(p2, v1, v2)]
            new_region = [v for v in vertices if v >= 0] # new_region contains all the finite vertices from std vor
            for p2, v1, v2 in ridges:
                if v2 < 0: # Why is this here? Just to flip order?
                    v1, v2 = v2, v1
                if v1 >= 0:  # v1 is always the one that could be at infinity
                    # finite ridge: already in the region
                    continue
                # Compute the missing endpoint of an
                # infinite ridge
                t = vor.points[p2] - \
                    vor.points[p1]  # tangent
                t /= np.linalg.norm(t) # Normalize
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vor.points[[p1, p2]]. \
                    mean(axis=0)
                direction = np.sign(
                    np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + \
                    direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
                new_ridge_vertices.append([v2, len(new_vertices)-1])

            # Sort region counterclockwise.
            vs = np.asarray([new_vertices[v]
                             for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(
                vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[
                np.argsort(angles)]
            new_regions.append(new_region.tolist())
        return new_regions, np.asarray(new_vertices), new_ridge_vertices


    def vertIdxToVal(self, vertices, ridge_vertices):
        '''
        Transforms the array of *indices* ridge_vertices into actual locations
        Input:
            vertices: Array containing the locations of all vertices
            ridge_Vertices: Array of indices (to vertices) of the vertices that make up the ith ridge
        Output:
            ridge_vertices_vals: 3D Array (n, 2, 2) of locations of the vertices that make up the n ridges
        '''
        ridge_vertices_val = []
        for idx_pair in ridge_vertices:
            ridge_vertices_val.append((vertices[idx_pair[0]].tolist(), vertices[idx_pair[1]].tolist()))
        unique_ridge_vertices_vals = np.unique(np.asarray(ridge_vertices_val), axis=0)

        return unique_ridge_vertices_vals

    def fillErrors(self):
        rr, cc = np.where(self.descriptiveDiagram == 0)

        for i in range(len(rr)):
            try:
                # print("Replacing a 0 with left point")
                self.descriptiveDiagram[rr, cc] = self.descriptiveDiagram[rr, cc - 1]
            except:
                # print("Didn't work. Replacing 0 with right point")
                self.descriptiveDiagram[rr, cc] = self.descriptiveDiagram[rr, cc + 1]


    def createDecomposition(self):
        label = 1 # This will be used to label all the regions in the descriptive map
        vor = Voronoi(self.generators)
        new_regions, new_vertices, new_ridge_vertices = self.voronoi_finite_polygons_2d(vor, 10000)
        ridge_verts = self.vertIdxToVal(new_vertices, new_ridge_vertices)

        # Draw lines and optionally validate area
        for r in new_regions:
            vs = new_vertices[r,:]
            v_x = vs[:,0].tolist()
            v_y = vs[:,1].tolist()

            rr_fill, cc_fill = polygon(v_x, v_y,shape=self.descriptiveDiagram.shape)
            temp_area = rr_fill.shape[0]
            temp_rr, temp_cc = rr_fill, cc_fill
            temp_label = label
            temp_region = Region(area = temp_area, rr = temp_rr, cc = temp_cc, label = temp_label)

            self.descriptiveDiagram[rr_fill, cc_fill] = label
            label += 1

            n = temp_region.num_agents 
            self.current_regions.append(temp_region)
        self.fillErrors()
    
    def createAgentMap(self):
        agent_count = 0
        while agent_count < self.num_agents:
            for region in self.current_regions:
                if agent_count >= self.num_agents:
                    break
                # Find random point inside the region
                rand_idx = np.random.randint(0, len(region.rr))
                self.current_agents.append((region.rr[rand_idx], region.cc[rand_idx]))
                agent_count +=  1
                region.num_agents += 1

        for idx_pair in self.current_agents:
            self.agentMap[idx_pair] = 1
            self.descriptiveDiagram[idx_pair] = -1


def main():
    env = Decompose_and_Search()
    env.createDecomposition()
    env.createAgentMap()
    env.plot2D(env.descriptiveDiagram)

    # Comment out line 201 if you don't want to include the agents in the descriptive cutting diagram
    # To access the agentMap, use env.agentMap
    # To create multiple maps sequentially, use the reset function
    
    # To save maps, uncomment the following two lines:
    # np.save("cuttingDiagram.npy", env.descriptiveDiagram)
    # np.save("agentMap.npy", env.agentMap)

if __name__ == "__main__":
    main()


