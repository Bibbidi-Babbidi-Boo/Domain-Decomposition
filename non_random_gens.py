import gym
import numpy as np
import gym_MAOC
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial
from skimage.draw import polygon_perimeter, polygon
from math import *
import matplotlib.pyplot as plt

env = gym.make('MAOC-v0')

def find_largest_area(sites, observation):
    ## If there are currently only two regions, for each point in the environment check to which one it is closer to
    if sites.shape[0] == 2:
        count_s0 = 0
        count_s1 = 0
        reg_s0 = []
        reg_s1 = []
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                if sqrt((i-sites[0][0])**2 + (j-sites[0][1])**2) > sqrt((i-sites[1][0])**2 + (j-sites[1][1])**2):
                    count_s0+=1
                    reg_s0.append([i, j])
                elif sqrt((i-sites[0][0])**2 + (j-sites[0][1])**2) > sqrt((i-sites[1][0])**2 + (j-sites[1][1])**2):
                    count_s1+=1
                    reg_s1.append([i, j])
        if count_s0>count_s1:
            count = count_s0
            reg = reg_s0
            pos = 0
        else:
            count = count_s1
            reg = reg_s1
            pos = 1
        ## Return the posn of region, its size and the points in the region
        return pos, count, np.array(reg)
    ## If more than two regions, plot the voronoi diagram
    if sites.shape[0]>2:
        vor = Voronoi(sites)
        new_regions, new_vertices, new_ridge_vertices = env.voronoi_finite_polygons_2d(vor, 10000)
        ridge_verts = env.vertIdxToVal(new_vertices, new_ridge_vertices)
        c  = -1
        curr_max_area = 0
        curr_max_pos = -1
        ## For every new region, find the number of points within that region
        for r in new_regions:
            c+=1
            vs = new_vertices[r,:]
            v_x = vs[:,0].tolist()
            v_y = vs[:,1].tolist()
            rr_fill, cc_fill = polygon(v_x, v_y,shape=(64, 64))
            if rr_fill.shape[0]>curr_max_area:
                curr_max_area = rr_fill.shape[0]
                curr_max_pos = c
                count = rr_fill.shape[0]
                rr_fill_best = rr_fill.tolist()
                cc_fill_best = cc_fill.tolist()
        reg = np.array((rr_fill_best, cc_fill_best))
        reg = np.transpose(reg)
        ## Return the posn, size and points within laregest region
        return curr_max_pos, count, reg


## Finding the reflection of the point abou x=y and x+y=63
def ref(x1, y1):
    x2, y2 = y1, x1
    x3, y3 = 63-y1, 63-x1
    return [x2, y2], [x3, y3]

## Given previous geenerator finding the new good generator
def get_good_gens(prev_loc, observation):
    ## For the second generator to be added just take the reflection of first as voronoi cnat be created
    if prev_loc.shape[0]==1:
        p1, p2 = ref(prev_loc[len(prev_loc)-1][0], prev_loc[len(prev_loc)-1][1])
        if (p1[0]-prev_loc[len(prev_loc)-1][0])**2 + (p1[1]-prev_loc[len(prev_loc)-1][1])**2 > (p2[0]-prev_loc[len(prev_loc)-1][0])**2 + (p2[1]-prev_loc[len(prev_loc)-1][1])**2:
            action = np.array([p1])
        else:
            action = np.array([p2])
        action = env.mapActionToNormAction(action)
        return action
    else:
        ## Find the largest region where the point is to be added
        pos, count, reg = find_largest_area(prev_loc, observation)
        min_val = 4096
        locs = prev_loc.tolist()
        cons = []
        ## Set constraints that the point chosen must lie within the largest region (closesr to the generator of that region than the others)
        for j in range(len(locs)):
            if j != pos:
                cons.append({'type':'ineq', 'fun': lambda point: (point[0]-locs[j][0])**2 + (point[1]-locs[j][1])**2 - (point[0]-locs[pos][0])**2 - (point[1]-locs[pos][1])**2})
        ## Sample 20 random points with the desired region
        for i in range(20):
            point = reg[np.random.randint(0, len(reg)-1)]
            try:
                ## Check if with that point as starting, we are able to solve the optimization problem, where we want to minimize the abs difference between the area of each region and the threshold
                next_loc_temp = minimize(rel_area, point, args = (prev_loc), constraints=cons, bounds=((0, 63), (0, 63)), method='SLSQP')
                val = rel_area(next_loc_temp.x, prev_loc)
                ## Get the best point, given our optimization finished
                if val<min_val and next_loc_temp.success==True:
                    min_val = val
                    next_loc = next_loc_temp
            except:
                pass
        ## Return best generator
        next_loc = np.array([next_loc.x])
        action = env.mapActionToNormAction(next_loc)
        return action

## Function to return the sumb of abs difference between the area of each region formed after adding the new point and the threshold
def rel_area(p1, prev_loc):
    sites = np.concatenate((prev_loc, np.array([p1])))
    vor = Voronoi(sites)
    new_regions, new_vertices, new_ridge_vertices = env.voronoi_finite_polygons_2d(vor, 10000)
    c = 0
    for r in new_regions:
        vs = new_vertices[r,:]
        v_x = vs[:,0].tolist()
        v_y = vs[:,1].tolist()
        rr_fill, cc_fill = polygon(v_x, v_y,shape=(64, 64))
        c += abs((rr_fill.shape[0]/4096)-env.COLL_THRESH)
    return c

###############################################################################

 ##         TO SOLO RUN THE SCRIPT UNCOMMENT        ##


# state, prev_loc = env.reset()
# observation = np.array([[1]*64]*64, dtype=np.uint8)
# total = 0
# count2 = 0
# num_episodes = 10
# t = 0
#
# while count2 < num_episodes:
#     t += 1
#     # env.render()
#     if t==1:
#         action = get_good_gens(prev_loc, observation)
#         print(prev_loc, observation)
#     else:
#         gens = np.array(info['generators']).astype('int32')
#         gens = np.reshape(gens, (len(gens), 2))
#         action = get_good_gens(gens, observation)
#         print(gens, observation)
#         # find_largest_area(gens, observation)
#     observation, reward, done, info = env.step(action)
#     if done:
#         print("==========DONE==========")
#         # env.render()
#         print("Episode finished after {} timesteps".format(t+1))
#         print("Last reward: {}".format(reward))
#         total+=len(gens)+1
#         t = 0
#         count2 += 1
#         observation, prev_loc = env.reset()
#         print("Total", total, count2)
# env.close()

###############################################################################
