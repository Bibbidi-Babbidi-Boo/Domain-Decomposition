import gym
from gym import error, spaces, utils
from gym.utils import seeding

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
from scipy.interpolate import interp1d



# Parameters #
# THRESHOLD               = 0.1
# MAX_STEPS               = 64

# COLLECTIBLE_COLOR       = 128 # GREY for collectible region
# NON_COLLECTIBLE_COLOR   = 255 # White for non-collectible region
# LINE_COLOR              = 0   # Black for line seperating region
# AGENTS_COLOR            = 200


class Region:
    def __init__(self, area = 0, info = 0, rr = None, cc = None, coll = False, vertices = [], index = None, label = None):
        self.area = area
        self.info = info
        self.rr = rr # Row indices for this region
        self.cc = cc # Col indices for this region
        self.coll = coll # Boolean, whether a region is collectible or not
        # self.vertices = vertices
        # self.index = index # This could be the index in the new_regions array
        self.label = label
        self.num_agents = 0 # TODO



class MAOCEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=64):
        # np.random.seed(10)
        size = 64
        self.map_shape = (size,size)
        self.num_channels = 3 #num_channels
        self.map_area = self.map_shape[0] * self.map_shape[1]

        # Define convention
        self.LINES = 1
        self.COLL = 128
        self.NOTCOLL = 255
        self.AGENTS = -255

        self.LINES_NORM = self.LINES/255
        self.COLL_NORM = self.COLL/255
        self.NOTCOLL_NORM = self.NOTCOLL/255
        self.AGENTS_NORM = self.AGENTS/255




        self.GENERATORS = 64 # Max number of generators (timesteps) allowed before ending episode
        self.num_agents = 8
        # Changes: now it's 1/n instead of 1/(n-1) (since num_agents is fixed)
        self.COLL_THRESH = np.random.uniform(1/(self.num_agents), 0.4) #1/self.num_agents #np.random.uniform(0.05, 0.4) #0.1 #np.random.uniform(0.05, 0.4)
        # TODO: Check threshold upper bound

        # Initialize generators
        self.all_generators  = [] # Used to store all generators, including the ones that cause Qhull errors, and repeated ones
        self.generators      = [] # Used to store the generators used to generate the Voronoi diagram
        self.current_regions = []

        # Initialize channels and state (stacked channels):
        self.state = np.squeeze(self.NOTCOLL * np.ones((self.map_shape[0], self.map_shape[1], self.num_channels), dtype = np.float32))
        self.lineMap            = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)
        self.areaMap            = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)
        self.cuttingDiagram     = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32) # Combination of lineMap and areaMap
        self.descriptiveDiagram = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)
        self.renderMap          = self.NOTCOLL_NORM * np.ones((self.map_shape), dtype = np.float32)

        # Initialize agents
        self.agentMap           = np.zeros(self.map_shape, dtype = np.int32)
        self.createAgentMap(self.num_agents)
        self.inaccessible = 0 # number of inaccessible regions: regions w no agents in them



        # Initialize normalized versions for observation
        self.state_norm = np.squeeze(self.NOTCOLL * np.ones((self.map_shape[0], self.map_shape[1], self.num_channels), dtype = np.float32))
        self.lineMap_norm = self.lineMap / 255
        self.areaMap_norm = self.areaMap / 255
        self.cuttingDiagram_norm = self.cuttingDiagram / 255


        # Initialize infoMap
        self.variance_scale = 100
        self.numDistribs = 50
        self.infoMap = np.zeros(self.map_shape, dtype = np.float32)
        # Add line to add info map that we know
        self.createInformationMap() # This updates the infoMap
        self.totalInfo = np.sum(self.infoMap)

        # Initialize observation and action spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, 2), dtype=np.float32) # Might change to accomodate non-square envs
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state.shape), dtype=np.float32)


        # Rendering
        self.color_map = "gray"

        # Class-wide done
        # self.done = False           # Already in reset() function
        self.max_timesteps = self.num_agents - 1 # TODO Do we need the -1?
        self.curr_timestep = 0          #CHECKED
        self.total_timesteps = 0        #CHECKED

        # Episode number
        self.ep = 0                     #CHECKED
        self.QhullErrors = 0


    # ================================ HELPER FUNCTIONS ===================================
    def bivariateGaussianMatrix(self):
        """
        Generates a single (not mixture) bivariate normal distribution, with independent axes
        Input:
            - map_shape: tuple (x, y) representing the map size
            - variance_scale: scales a the random variance for the x and y components
        Output:
            - distribution_matrix: 2D array containing single normal bivariate distribution
            - mean of the bivariateGaussain
        """
        x, y = np.mgrid[0:self.map_shape[0], 0: self.map_shape[1]]
        pos = np.dstack((x, y))
        mean2D = self.map_shape[0] * np.random.rand(1, 2)[0]
        cov2D = np.zeros((2, 2))
        cov2D[([0, 1], [0, 1])] = self.variance_scale * np.random.rand(1, 2)[0]
        rv = multivariate_normal(mean2D, cov2D)
        return rv.pdf(pos), mean2D

    def createInformationMap(self):
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
        for _ in range(self.numDistribs):
            infoMap, meanDist = self.bivariateGaussianMatrix()
            infoMaps.append(infoMap)
            means.append(meanDist)
        infoMaps = np.asarray(infoMaps)
        means = np.asarray(means)


        # Mixture of Gaussians + epsilon probability for "empty" space
        infoMap =  np.mean(infoMaps, axis=0)
        # Scale so that the maximum value in the map is 1
        # infoMapRaw = infoMap -> This is to check the actual values without scaling
        maxInfo = np.max(infoMap)
        scalar = 1/maxInfo
        self.infoMap = infoMap * scalar

        return infoMap, scalar, means

    def createAgentMap(self, num_agents):
        agent_map_complete = False
        while not agent_map_complete:
            agent_loc = tuple(np.random.randint(0, self.map_shape[0], (2)))
            # print(agent_loc)
            if self.agentMap[agent_loc] != 1:
                self.agentMap[agent_loc] = 1

            if np.sum(self.agentMap) >= num_agents:
                agent_map_complete = True
        # np.save("agentMap.npy", self.agentMap)

    def countAgents(self):
        num_regions = len(self.current_regions)
        # print("There are {} regions".format(num_regions))

        count = 0
        for region in self.current_regions:
            # print(region.num_agents)
            if region.num_agents == 0:
                count += 1
        # print("There are {} regions without agents".format(count))

        # idxs = np.where(self.descriptiveDiagram == -1)
        # if np.sum(self.agentMap[idxs]) != 0:
        #     print("There are {} agents on the line".format(np.sum(self.agentMap[idxs])))
        # Need to add the rr,cc indices for each Region object so that we can search in them
        # Iterate through the list, and using the rr,cc, count how many agents there are in each region and fill this information out in the self.current_regions list
        diff = count - self.inaccessible
        # print("diff = {} - {} = {}".format(count, self.inaccessible, diff))
        self.inaccessible = count
        return diff

    def countAgentsPerRegion(self, region):
        region.num_agents = np.sum(self.agentMap[region.rr, region.cc])


    def plot2D(self, array):
        '''
        Plot a 2D array with a colorbar
        Input:
            - array: 2D array to plot
        '''
        plt.imshow(array)
        plt.colorbar()
        plt.show()

    def midPoint_and_slope(self, p1, p2):
        '''
        Finds the midpoint between two points and the slope of the perpendicular bisector

        Input:
            - p1: Tuple of point 1: (x,y)
            - p2: Tuple of point 2: (x,y)
        Output:
            - midPoint: Tuple (x,y)
            - slope: Float
        '''
        midPoint = ((p1[0]+ p2[0])/2, (p1[1] + p2[1])/2)
        slope_og = (p2[1] - p1[1])/(p2[0] - p1[0])
        slope = - 1/slope_og
        return midPoint, slope

    def findEndpoints(self, midpoint, perpSlope):
        '''
        This function finds the endpoints for a line that has a given midpoint and a slope.

        Its intended use is in the case where we have two generator points, and we want to find the
        cutting line between the two. To do this, we find the midpoint of these points, and the slope of the
        perpendicular bisector.

        This is a naive approach, an improved version could be obtained by doing an incremental Bresenham's line algorithm
        starting at the point in both directions. In this approach, we find each of the four possible endpoints given our map
        dimensions, and we choose the two that lie within the boundary.

        Input:
            - midpoint: a tuple (x,y) representing a point in the line we want to find
            - perpSlope: a float representing the slope of the perpendicular bisector of two points
            - map_shape: a tuple (x,y) representing the size of the map
        Output:
            - endpoints: A list containing 2 integer tuples that represent the endpoints on the map: [(x_1, y_1), (x_2, y_2)]
        '''
        endpoints = []
        raw_endpoints = []

        # Check if slope is inf, -inf -> vertical cutting line
        if (perpSlope == -np.inf) or (perpSlope == np.inf):
            endpoints.append((int(midpoint[0]), 0))
            endpoints.append((int(midpoint[0]), self.map_shape[1]-1))
            raw_endpoints.append((int(midpoint[0]), 0))
            raw_endpoints.append((int(midpoint[0]), self.map_shape[1]-1))
            return endpoints, raw_endpoints

        # Check if slope is 0, -0 -> horizontal cutting line
        if perpSlope == 0:
            endpoints.append((0, int(midpoint[1])))
            endpoints.append((self.map_shape[0]-1, int(midpoint[1])))
            raw_endpoints.append((0, int(midpoint[1])))
            raw_endpoints.append((self.map_shape[0]-1, int(midpoint[1])))
            return endpoints, raw_endpoints

        # First possible endpoint: (x, 0)
        x_try = midpoint[0] - midpoint[1]/perpSlope

        if (int(round(x_try)) >= 0) and (int(round(x_try)) < self.map_shape[0]):
            if x_try >= 0:
                endpoints.append((int(round(x_try)), 0))
        raw_endpoints.append((x_try, 0))

        # Second possible endpoint: (x_map, y)
        y_try = (self.map_shape[0]-1) * perpSlope - perpSlope*midpoint[0] + midpoint[1] # mapshape-1 is to deal with indices starting at 0
        if (int(round(y_try)) >= 0) and (int(round(y_try)) < self.map_shape[1]):
            if y_try >= 0:
                endpoints.append((self.map_shape[0]-1, int(round(y_try))))
        raw_endpoints.append((self.map_shape[0]-1, y_try))

        # Third possible endpoint: (x, y_map)
        x_try = ((self.map_shape[1]-1) - midpoint[1] + perpSlope * midpoint[0]) / perpSlope
        if (int(round(x_try)) >= 0) and (int(round(x_try)) < self.map_shape[0]):
            if x_try >= 0:
                endpoints.append((int(round(x_try)), self.map_shape[1]-1))
        raw_endpoints.append((x_try, self.map_shape[1]-1))

        # Fourth possible endpoint: (0, y)
        y_try = -perpSlope * midpoint[0] + midpoint[1]
        if (int(round(y_try)) >= 0) and (int(round(y_try)) < self.map_shape[1]):
            if y_try >= 0:
                endpoints.append((0, int(round(y_try))))
        raw_endpoints.append((0, y_try))

        endpoints = list(set(endpoints)) # Accounts for corner coordinates

        if len(endpoints) != 2:
            print("ERROR: Don't have two endpoints!")

        return endpoints, raw_endpoints

    def drawLine(self, start, end):
        '''
        Implements Bresenham's line algorithm
        From http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm

        Input:
            - start: Tuple of coordinates for starting point (x,y)
            - end: Tuple of coordinates for end point (x,y)
        Output:
            - points: List of points that make up the line between the start and end points
        '''
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points


    def drawMultipleLines(self, ridgeVertices):
        '''
        Calls the drawLine method for each pair of vertices in the array ridgeVertices
        Input:
            - ridgeVertices: 3D array (n, 2, 2) where n is the number of ridges to draw,
                             the second dimension represents the two endpoints of the line,
                             and the third dimension represents the x,y coordinates for a point
        Outpu:
            - lines: A list of lists, where each of the lists is the coordinates of the points
                     that make up each line

        '''

        lines = []
        for n in range(ridgeVertices.shape[0]):
            v1 = tuple(ridgeVertices[n][0])
            v2 = tuple(ridgeVertices[n][1])

            # Make the vertices integers instead of floats
            v1 = tuple(map(int, v1))
            v2 = tuple(map(int, v2))
            lines.append(self.drawLine(v1, v2))
        return lines


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


    def calculateReward(self, diff):

        # print("----Calculating Reward----")
        # print("New regions without agents: {}".format(diff))


        N_0 = np.ceil(1/self.COLL_THRESH) # Ideal number of regions, in the case where the infoMap is uniform
        r_0 = 2 - N_0 * self.COLL_THRESH # Normalization constant.

        # Constant for each new inaccessible region
        c = N_0*r_0
        # print("(constant is: {})".format(c))
        # Reward only obtained at the end of each episode
        if self.done:
            # for reg in self.current_regions:
            #     if reg.coll: ## for every admissible region
            #         # at += reg.area  ## checking if sum of araes is total area
            #         r += math.sqrt(self.COLL_THRESH-(reg.area/4096))  ## minimize the difference between threshold and area partition, so we get good cuts which seperate into pieces of max area
            # # print("R", -r/(len(self.generators)*math.sqrt(self.COLL_THRESH)), ((math.ceil(1/self.COLL_THRESH)-len(self.generators))/len(self.generators)))
            # r = -r/(len(self.generators)*math.sqrt(self.COLL_THRESH)) + ((math.ceil(1/self.COLL_THRESH)-len(self.generators))/len(self.generators)) + 1 ## Normalize the previous reward and add to it the second reward which becomes more negative for more partitions. Add 1 to change the range of the reward from (-2, 0) to (-1, 1).

            # print("Episode is done")

            r = (2 - len(self.all_generators) * self.COLL_THRESH) / r_0 - c * diff
            # print("Last timestep reward: {} - {} = {}".format((2 - len(self.all_generators) * self.COLL_THRESH) / r_0, c * diff, r))
        else:
            r = -c*diff
            # print("Timestep reward: {}".format(r))
        return r

    def observe(self):
        observation = self.state_norm
        return observation

    def checkDone(self):
        for region in self.current_regions:
            if not region.coll:
                return False
        return True

    def renderHelper(self, done = False, showGenerators = False):

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
        ax.imshow(self.infoMap, cmap="viridis", interpolation="none", alpha=0.8)
        ax.imshow(self.renderMap, cmap="RdGy", interpolation="none", alpha=0.8)
        fig.suptitle("Threshold: {}".format(self.COLL_THRESH))
        # plt.show()
        # time.sleep(0.2)
        # plt.clf()
        # Force a draw so we can grab the pixel buffer
        canvas.draw()
        # grab the pixel buffer and dump it into a numpy array
        X = np.array(canvas.renderer.buffer_rgba())
        # self.plot2D(X)
        # time.sleep(0.2)
        # plt.clf()
        # figs.append(X)
        return X


        # fig = Figure()
        # canvas = FigureCanvas(fig)
        # ax1, ax2 = fig.subplots(1, 2, sharey=True)
        # ax1.imshow(self.infoMap, cmap="viridis", interpolation="none")
        # # ax1.imshow(self.cuttingDiagram, cmap="Greys", interpolation="none", alpha=0.3)
        # ax2.imshow(self.renderMap, cmap="RdGy")
        # # Force a draw so we can grab the pixel buffer
        # canvas.draw()
        # # grab the pixel buffer and dump it into a numpy array
        # X = np.array(canvas.renderer.buffer_rgba())
        # self.plot2D(X)
        # time.sleep(0.2)
        # plt.clf()
        # # figs.append(X)
        # return X


    def normActionToMapAction(self, norm_action):
        '''
        Takes a normalized action [-1,1] and maps it into the size of the map [0, map_shape-1] and casts it as an integer
        norm_action is a 2-dim array of shape (1, 2)
        '''
        map_action = np.ones(norm_action.shape, dtype=np.int32)
        for i in range(norm_action.shape[1]):
            map_action[0,i] = round((self.map_shape[0]-1)/2 * (norm_action[0,i] + 1))
        return map_action

    def mapActionToNormAction(self, map_action):
        '''
        Takes a map action [0, map_shape-1] and maps it into normalized actions [-1, 1]
        map_action is a 2-dim array of shape (1, 2)
        '''
        norm_action = np.ones(map_action.shape, dtype=np.float32)
        for i in range(map_action.shape[1]):
            norm_action[0,i] = 2/(self.map_shape[0]-1) * map_action[0,i] - 1
        return norm_action

    def resetMaps(self):
        '''
        Update lineMap, areaMap, cuttingDiagram, and state. infoMap does not get updated because it is static in this version
        '''
        self.state = np.squeeze(self.NOTCOLL * np.ones((self.map_shape[0], self.map_shape[1], self.num_channels), dtype = np.int32))
        self.lineMap = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)
        self.areaMap = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)

        self.cuttingDiagram = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32) # Combination of lineMap and areaMap
        self.descriptiveDiagram = self.NOTCOLL * np.ones((self.map_shape), dtype = np.int32)
        # self.agentMap = np.zeros(self.map_shape, dtype = np.uint8)
        # print("---Resetting renderMap---")
        self.renderMap          = self.NOTCOLL_NORM * np.ones((self.map_shape), dtype = np.float32)


        self.state_norm = np.squeeze(self.NOTCOLL * np.ones((self.map_shape[0], self.map_shape[1], self.num_channels), dtype = np.float32))
        self.lineMap_norm = self.lineMap / 255
        self.areaMap_norm = self.areaMap / 255
        self.cuttingDiagram_norm = self.cuttingDiagram / 255

    def updateMaps(self):
        '''
        Uses current generators to update the lineMap, areaMap, cuttingDiagram, and state
        '''


        # Convert list of generators to array of sites
        sites = np.vstack(np.asarray(self.generators))
        self.current_regions = []

        label = 1
        lineDesc = -1


        # Define functions for two cases: 2 generators, or more than 2 generators
        def twoGenerators():
            midpoint, slope = self.midPoint_and_slope(tuple(sites[0]), tuple(sites[1]))
            endpoints, raw_endpoints = self.findEndpoints(midpoint, slope)
            endp1 = endpoints[0]
            endp2 = endpoints[1]

            temp_color1 = 4
            temp_color2 = 5
            area1 = 0
            area2 = 0

            # points = self.drawLine(endp1[::-1],endp2[::-1])
            points = self.drawLine(endp1,endp2)

            label1 = 1
            label2 = 2


            # Vertical cutting line
            if slope == np.inf or slope == -np.inf:
                for r in range(self.map_shape[1]):
                    for c in range(self.map_shape[0]):
                        if c < int(midpoint[0]):
                            self.cuttingDiagram[r,c] = temp_color1
                            self.descriptiveDiagram[r, c] = label1
                            area1 += 1
                        elif c > int(midpoint[0]):
                            self.cuttingDiagram[r,c] = temp_color2
                            self.descriptiveDiagram[r, c] = label2
                            area2 += 1

            # For any other line, use actual y value to determine area
            else: # Any other non-vertical cutting line
                for r in range(self.map_shape[0]):
                    real_y = 1/slope * (r - midpoint[1]) + midpoint[0]
                    for c in range(self.map_shape[1]):
                        if c > real_y:
                            self.cuttingDiagram[c,r] = temp_color1
                            self.descriptiveDiagram[c, r] = label1
                            area1 += 1
                        else:
                            self.cuttingDiagram[c,r] = temp_color2
                            self.descriptiveDiagram[c, r] = label2
                            area2 += 1

            # Create Region instance and add: area, info, rr, cc, label, num_agents
            temp_area = area1
            temp_rr, temp_cc = np.where(self.cuttingDiagram == temp_color1)
            temp_info = np.sum(self.infoMap[temp_rr, temp_cc])
            temp_label = label1
            temp_region = Region( area = temp_area, info = temp_info, rr = temp_rr, cc = temp_cc, label = temp_label)
            temp_region.num_agents = np.sum(self.agentMap[temp_region.rr, temp_region.cc])


            # Add regions to region list
            # temp_region = Region(area1) # TODO
            # Assign actual colors to the areas (validate areas)
            region1 = np.where(self.cuttingDiagram == temp_color1) # TODO return rr,cc for Region, get info, get num_agents


            info_norm = temp_region.info/self.totalInfo
            area_norm = temp_region.area/self.map_area
            n         = temp_region.num_agents
            if area_norm*info_norm/n <= (self.COLL_THRESH * self.COLL_THRESH):
                self.cuttingDiagram[region1] = self.COLL
                self.areaMap[region1] = self.COLL
                # print("---Updating renderMap: 2 Generators, region 1 collectible---")
                self.renderMap[region1] = self.COLL_NORM
                temp_region.coll = True
            else:
                self.cuttingDiagram[region1] = self.NOTCOLL
                self.areaMap[region1] = self.NOTCOLL
                # print("---Updating renderMap: 2 Generators, region 1 not collectible---")
                self.renderMap[region1] = self.NOTCOLL_NORM
                temp_region.coll = False
            self.current_regions.append(temp_region)

            # Create Region instance and add: area, info, rr, cc, label, num_agents
            temp_area = area2
            temp_rr, temp_cc = np.where(self.cuttingDiagram == temp_color2)
            temp_info = np.sum(self.infoMap[temp_rr, temp_cc])
            temp_label = label2
            temp_region = Region( area = temp_area, info = temp_info, rr = temp_rr, cc = temp_cc, label = temp_label)
            temp_region.num_agents = np.sum(self.agentMap[temp_region.rr, temp_region.cc])

            # temp_region = Region(area2)
            region2 = np.where(self.cuttingDiagram == temp_color2)


            info_norm = temp_region.info/self.totalInfo
            area_norm = temp_region.area/self.map_area
            n         = temp_region.num_agents
            if area_norm*info_norm/n <= (self.COLL_THRESH * self.COLL_THRESH):
                self.cuttingDiagram[region2] = self.COLL
                self.areaMap[region2] = self.COLL
                self.renderMap[region2] = self.COLL_NORM
                temp_region.coll = True
            else:
                self.cuttingDiagram[region2] = self.NOTCOLL
                self.areaMap[region2] = self.NOTCOLL
                self.renderMap[region2] = self.NOTCOLL_NORM
                temp_region.coll = False
            self.current_regions.append(temp_region)


            # Plot lines
            for i in points:
                self.cuttingDiagram[i] = self.LINES
                self.lineMap[i] = self.LINES
                # self.descriptiveDiagram[i] = lineDesc
                self.renderMap[i] = self.LINES_NORM


    #         return canvas

        def moreThanTwoGenerators():
            # print("Inside more than two generators")
            label = 1 # This will be used to label all the regions in the descriptive map
            lineDesc = -1 # This will be used to label the lines in the descriptive map
            vor = Voronoi(sites)
            new_regions, new_vertices, new_ridge_vertices = self.voronoi_finite_polygons_2d(vor, 10000)
            ridge_verts = self.vertIdxToVal(new_vertices, new_ridge_vertices)

            # Draw lines and optionally validate area
            for r in new_regions:
                vs = new_vertices[r,:]
                v_x = vs[:,0].tolist()
                v_y = vs[:,1].tolist()

                rr_fill, cc_fill = polygon(v_x, v_y,shape=self.cuttingDiagram.shape)
                temp_area = rr_fill.shape[0]
                temp_rr, temp_cc = rr_fill, cc_fill
                temp_info = np.sum(self.infoMap[rr_fill, cc_fill])
                temp_label = label
                temp_region = Region( area = temp_area, info = temp_info, rr = temp_rr, cc = temp_cc, label = temp_label)
                temp_region.num_agents = np.sum(self.agentMap[temp_region.rr, temp_region.cc])
                self.descriptiveDiagram[rr_fill, cc_fill] = label
                label += 1
                # temp_region = Region(rr_fill.shape[0])
                # print("Region size: " + str(rr_fill.shape[0]))

                info_norm = temp_region.info/self.totalInfo
                area_norm = temp_region.area/self.map_area
                n         = temp_region.num_agents
                if area_norm*info_norm/n <= (self.COLL_THRESH * self.COLL_THRESH):
                    # print("region {} is collectible".format(r))
                    self.cuttingDiagram[rr_fill, cc_fill] = self.COLL
                    self.areaMap[rr_fill, cc_fill] = self.COLL
                    self.renderMap[rr_fill, cc_fill] = self.COLL_NORM
                    temp_region.coll = True
                else:
                    # print("region {} is not collectible".format(r))
                    self.cuttingDiagram[rr_fill, cc_fill] = self.NOTCOLL
                    self.areaMap[rr_fill, cc_fill] = self.NOTCOLL
                    self.renderMap[rr_fill, cc_fill] = self.NOTCOLL_NORM
                    temp_region.coll = False

                self.current_regions.append(temp_region)
                rr, cc = polygon_perimeter(v_x, v_y,shape=self.cuttingDiagram.shape, clip=False)
                self.cuttingDiagram[rr, cc] = self.LINES
                self.lineMap[rr, cc] = self.LINES
                # self.descriptiveDiagram[rr, cc] = lineDesc
                self.renderMap[rr, cc] = self.LINES_NORM
    #         return canvas

        # Do appropriate acction based on number of generators

        # No sites
        if sites.shape[0] == 0:
            # return self.cuttingDiagram # Return empty canvas
            pass

        # One site
        elif sites.shape[0] == 1:
            # print("One generator")
            # # Plot generators
            # for s in sites.tolist():
            #     self.cuttingDiagram[tuple(s)] = GENS
            self.cuttingDiagram[:] = self.NOTCOLL
            self.areaMap[:] = self.NOTCOLL
            self.lineMap[:] = self.NOTCOLL
            self.descriptiveDiagram = label
            self.renderMap[:] = self.NOTCOLL_NORM


            # TODO: Add single region to self.current_regions()
            temp_area = self.map_area
            temp_rr, temp_cc = np.where(self.cuttingDiagram == self.NOTCOLL)
            temp_info = self.totalInfo
            temp_label = label
            temp_region = Region( area = temp_area, info = temp_info, rr = temp_rr, cc = temp_cc, label = temp_label)
            temp_region.num_agents = np.sum(self.agentMap[temp_region.rr, temp_region.cc])

            self.current_regions.append(temp_region)


        # Two sites
        elif sites.shape[0] == 2:
            twoGenerators()

        # More than two sites
        else:
            moreThanTwoGenerators()

        # Count number of agents
        new_inaccessible = self.countAgents() # Figure out how many new inaccessible regions there are to penalize in reward function
        # Plot agents in cuttingDiagram
        idxs = np.where(self.agentMap == 1)
        # self.cuttingDiagram[idxs] = self.AGENTS
        # print("Agents have a value of {}".format(self.AGENTS_NORM))
        self.renderMap[idxs] = self.AGENTS_NORM



        # Update normalized versions
        self.lineMap_norm = self.lineMap / 255
        self.areaMap_norm = self.areaMap / 255
        self.cuttingDiagram_norm = self.cuttingDiagram / 255

        # This changes based on what maps we want to return as the observation
        self.state = np.stack((self.cuttingDiagram, self.infoMap, self.agentMap), axis=-1)
        self.state_norm = np.stack((self.cuttingDiagram_norm, self.infoMap, self.agentMap), axis=-1)

        return new_inaccessible






    # ================================ GYM FUNCTIONS ===================================

    def step(self, action):
        '''
        action should be a 2-element tuple corresponding to the normalized coordinates of the new Voronoi generator.
        That is, input is [-1, 1], and we need to map it to [0, map_shape[0]]
        '''
        # self.cuttingDiagram = np.array([[self.NOTCOLL]*self.map_shape[0]]*self.map_shape[1], dtype=np.uint8) # Clear out the diagram each time, but build with incremental points
        self.resetMaps()
        # print("Timestep: {}".format(self.curr_timestep))

        action = self.normActionToMapAction(action)

        self.generators.append(action)
        self.all_generators.append(action)

        # Convert the list of generators to array
        gen_array = np.vstack(np.asarray(self.generators))
        unique_generators = np.unique(gen_array, axis=0)
        # Make sure there are no repeated points, and if there are, use only the unique points, set the reward to 0 and return
        repeatedFlag = False
        originalSize = gen_array.shape[0]
        uniqueSize = unique_generators.shape[0]


        if originalSize != uniqueSize:
            repeatedFlag = True
            self.generators.pop()

        # If three points are colinear, pop the last generator and and set the reward to 0
        colinearFlag = False
        # Execute action
        try:
            # self.generatePlot(np.vstack(np.asarray(self.generators)))
            diff = self.updateMaps()
        except Exception as exception:
            print("============Exception=============")
            self.QhullErrors += 1
            colinearFlag = True
            print("Exception raised: {}".format(exception))
            self.generators.pop()
            # self.generatePlot(np.vstack(np.asarray(self.generators)))
            diff = self.updateMaps()
        observation = self.observe()

        # done
        # toCollect = np.where(self.cuttingDiagram == self.NOTCOLL)
        # if (toCollect[0].shape[0] == 0) or (self.curr_timestep >= self.max_timesteps-1):
        if self.checkDone() or (self.curr_timestep >= self.max_timesteps-1):
            self.done = True
        else:
            self.done = False

        # Reward
        reward = 0
        # R_t = base cost + #acceptable_regions/(2 * #total regions) + penalty for placing a generator inside a green region
        if (not repeatedFlag) and (not colinearFlag) or self.done:
            reward = self.calculateReward(diff)

        info = {"generators":self.generators, "episode":self.ep, "threshold":self.COLL_THRESH, 'reward': reward, 'done': self.done, "QhullErrors": self.QhullErrors, "ep_timesteps": self.curr_timestep, "total_timesteps": self.total_timesteps}
        #info = {"Points":self.generators, "episode":self.ep, "threshold":self.COLL_THRESH, 'reward': reward, 'done': self.done}

        self.curr_timestep += 1
        self.total_timesteps += 1

        if self.done == True:
            self.ep += 1

        return observation, reward, self.done, info

    def reset(self):
        # print("+++++++RESET+++++++")
        self.done = False
        self.curr_timestep = 0
        self.COLL_THRESH = np.random.uniform(1/(self.num_agents), 0.4) #0.1 #np.random.uniform(0.05, 0.4) # THRESHOLD #np.random.uniform(0.05, 0.4)
        self.all_generators = []
        self.generators = []
        self.current_regions = []
        self.QhullErrors = 0
        self.inaccessible = 0

        random_point = np.random.randint(0, self.map_shape[0], (1,2), dtype=np.int8)
        random_point_norm = self.mapActionToNormAction(random_point)
        self.generators.append(random_point) # add random point
        self.all_generators.append(random_point)

        # Update infoMap
        self.createInformationMap()
        self.totalInfo = np.sum(self.infoMap)

        self.agentMap = np.zeros(self.map_shape, dtype = np.int32)
        self.createAgentMap(self.num_agents)

        # print("Episode threshold: {}".format(self.COLL_THRESH))


        # self.cuttingDiagram = np.array([[self.NOTCOLL]*self.map_shape[0]]*self.map_shape[1], dtype=np.uint8)
        self.resetMaps()

        #Add first point to cutting diagram
        # self.generatePlot(np.vstack(np.asarray(self.generators)))
        self.updateMaps()
        observation = self.observe()
        # print("====================================")
        # random_point = np.asarray(random_point, dtype=np.float32)
        # for i in range(2):
        #     random_point[0,i] = ((random_point[0,i]/63.)*2.)-1      # Normalize the first generator

        return observation, random_point_norm

    def render(self, mode='human'):
        return self.renderHelper(done = False, showGenerators = True) # Generators are shown, but are not provided as observation to the agent

    def close(self):
        self.renderHelper(done = True, showGenerators = False)
