import gym
import gym_MAOC
import tensorflow as tf
import numpy as np
import os
import multiprocessing
import threading
from random import choice
import time
from time import sleep
from time import time
import scipy
from scipy.interpolate import interp1d
import scipy.signal as signal
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.spatial
from skimage.draw import polygon_perimeter, polygon
import imageio
from math import *
from ACNet import *

#################### PARAMETERS #################################

#### SYSTEM ####
NUM_THREADS     = 12 #multiprocessing.cpu_count() #8

#### NETWORK ####
GRID_SIZE       = 64
a_size          = 2
NUM_CHANNELS    = 3
BATCH_SIZE      = 1

#### TRAINING ####
LR_Q                   = 1e-5
max_episode_length     = 300
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .99 # discount rate for advantage estimation and reward discounting
GLOBAL_NET_SCOPE       = 'global'

#### CONDITIONS ####
SUMMARY_WINDOW = 10
RESET_TRAINER  = False
TRAINING       = True
ADAPT_LR       = False
OUTPUT_GIFS    = True

#### DIRECTORIES TO SAVE DATA ####
unique_name  = '_ma_info'
model_path   = 'model' + str(unique_name)
train_path   = 'train' + str(unique_name)
gifs_path    = 'gifs' + str(unique_name)
logfile_path = 'info' + str(unique_name) # Stores info, which includes the generators used in each case for debugging purposes

#### LOAD MODEL ####
load_model = False
MODEL = ''#'/home/himanshu/MAOceanCleanup/model'
MODEL_NUMBER = None #40300


### RUN HEURISTIC ###
RUN_HEURISTIC = False


#### Shared arrays for tensorboard ####
episode_rewards        = [ [] for _ in range(NUM_THREADS) ]
episode_lengths        = [ [] for _ in range(NUM_THREADS) ]
episode_mean_values    = [ [] for _ in range(NUM_THREADS) ]
rollouts               = [ None for _ in range(NUM_THREADS)]
printQ                 = False # (for headless)

############## FUNCTIONS ################
def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")


def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker():
    def __init__(self,name,GRID_SIZE,a_size,trainer,model_path, logfile_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.logfile_path = logfile_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        # self.episode_rewards = []
        # self.episode_lengths = []
        # self.episode_mean_values = []
        # self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
        self.nextGIF = episode_count
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ACNet(scope = self.name, GRID_SIZE = GRID_SIZE, a_size = 2, trainer = trainer, TRAINING = True, GLOBAL_NET_SCOPE = GLOBAL_NET_SCOPE)
        self.update_local_ops = update_target_graph(GLOBAL_NET_SCOPE,self.name)

        self.env = gym.make('MAOC-v0')

    def train(self,rollout,sess,gamma,bootstrap_value):

        rollout = np.array(rollout)
        # print(rollout[][0].shape)
        observations = rollout[:,0]
        next_loc = rollout[:,[1,2]]
        rewards = rollout[:,3]
        next_observations = rollout[:,4]
        values = rollout[:,6]
        prev_loc = rollout[:,[7,8]]
        # print(observations[0])
        # print(np.vstack(observations).shape)
        observations = np.vstack(observations)    # checked

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)             # checked

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:observations,
            self.local_AC.advantages:advantages,
            self.local_AC.prev_loc:prev_loc,
            self.local_AC.sampled_next_locs:next_loc,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n

    def find_largest_area(self, sites, observation):
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
            # print("VOR:", sites)
            new_regions, new_vertices, new_ridge_vertices = self.env.voronoi_finite_polygons_2d(vor, 10000)
            ridge_verts = self.env.vertIdxToVal(new_vertices, new_ridge_vertices)
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


    ## Finding the reflection of the point about x=y and x+y=63
    def ref(self, x1, y1):
        x2, y2 = y1, x1
        x3, y3 = 63-y1, 63-x1
        return [x2, y2], [x3, y3]


    ## Given previous geenerator finding the new good generator
    def get_good_gens(self, prev_loc, observation):
        ## For the second generator to be added just take the reflection of first as voronoi cnat be created
        if prev_loc.shape[0]==1:
            p1, p2 = self.ref(prev_loc[len(prev_loc)-1][0], prev_loc[len(prev_loc)-1][1])
            if (p1[0]-prev_loc[len(prev_loc)-1][0])**2 + (p1[1]-prev_loc[len(prev_loc)-1][1])**2 > (p2[0]-prev_loc[len(prev_loc)-1][0])**2 + (p2[1]-prev_loc[len(prev_loc)-1][1])**2:
                action = np.array([p1])
            else:
                action = np.array([p2])
            action = self.env.mapActionToNormAction(action)
            return action
        else:
            ## Find the largest region where the point is to be added
            pos, count, reg = self.find_largest_area(prev_loc, observation)
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
                    next_loc_temp = minimize(self.rel_area, point, args = (prev_loc), constraints=cons, bounds=((0, 63), (0, 63)), method='SLSQP')
                    val = self.rel_area(next_loc_temp.x, prev_loc)
                    ## Get the best point, given our optimization finished
                    if val<min_val and next_loc_temp.success==True:
                        min_val = val
                        next_loc = next_loc_temp
                except:
                    pass
            ## Return best generator
            next_loc = np.array([next_loc.x])
            action = self.env.mapActionToNormAction(next_loc)
            return action


    ## Function to return the sumb of abs difference between the area of each region formed after adding the new point and the threshold
    def rel_area(self, p1, prev_loc):
        sites = np.concatenate((prev_loc, np.array([p1])))
        vor = Voronoi(sites)
        new_regions, new_vertices, new_ridge_vertices = self.env.voronoi_finite_polygons_2d(vor, 10000)
        c = 0
        for r in new_regions:
            vs = new_vertices[r,:]
            v_x = vs[:,0].tolist()
            v_y = vs[:,1].tolist()
            rr_fill, cc_fill = polygon(v_x, v_y,shape=(64, 64))
            c += abs((rr_fill.shape[0]/4096)-self.env.COLL_THRESH)
        return c


    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_rewards, episode_lengths, episode_mean_values, episode_count
        # episode_count = sess.run(self.global_episodes)  # is this correct?
        total_steps = 0
        print ("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():

                sess.run(self.update_local_ops)
                all_gens = []
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                done = False

                state, prev_loc = self.env.reset()
                # print(prev_loc)
                all_gens = [prev_loc]
                # episode_frames.append(state*255)
                episode_frames.append(self.env.render())
                
                rnn_state = self.local_AC.state_init
                # print(rnn_state[0].shape)
                self.batch_rnn_state = rnn_state
                # while self.env.is_episode_finished() == False:

                saveGIF = False
                if OUTPUT_GIFS and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF = episode_count + 64
                    GIF_episode = int(episode_count)
                    # episode_frames = [ self.env.cuttingDiagram ]
                    episode_frames = [ self.env.render() ]

                while True:
                    # if episode_count%5 == 0:
                    # self.env.render()
                    # state = state.reshape(1, state.shape[0], state.shape[1], 1)
                    state = np.reshape(state, [BATCH_SIZE, GRID_SIZE, GRID_SIZE, -1]) # BATCH_SIZE should always be 1 because of RNN, last arg (-1) infers size to accomodate for 1 or multiple channels

                    # print(self.env.generators)
                    #Take an action using probabilities from policy network output.
                    next_loc, value, rnn_state = sess.run([self.local_AC.next_loc,self.local_AC.value,self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs:state,
                        self.local_AC.prev_loc: prev_loc,
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})

                    # Uncomment to run with heuristic:
                    if RUN_HEURISTIC:
                        if np.random.rand() < 1. / (0.1*episode_count + 0.1):
                            in1 = np.reshape(all_gens, (len(all_gens), 2))
                            in2 = np.reshape(state, (64, 64))
                            next_loc = self.get_good_gens(in1, in2)


                    next_state, reward, done, info = self.env.step(next_loc)

                    all_gens = np.array(info['generators']).astype('int32')
                    all_gens = np.reshape(all_gens, (len(all_gens), 2))


                    episode_buffer.append([state,next_loc[0][0],next_loc[0][1],reward,next_state,done,value[0,0],prev_loc[0][0],prev_loc[0][1]])
                    episode_values.append(value[0,0])

                    episode_reward += reward
                    total_steps += 1
                    episode_step_count += 1


                    prev_loc = next_loc
                    state = next_state

                    # all_gens.append(self.env.normActionToMapAction(prev_loc))

                    if saveGIF:
                        # episode_frames.append(self.env.cuttingDiagram)
                        episode_frames.append(self.env.render())


                    if done:
                        # Write the generators used for this episode
                        with open(logfile_path+'/infoFile.txt', 'a') as file:
                            file.write("==========\n")
                            file.write("Name: {}\n".format(self.name))
                            file.write("Episode: {}\n".format(info["episode"]))
                            file.write("Number of timesteps: {}\n".format(info["ep_timesteps"]))
                            file.write("Reward: {}\n".format(info["reward"]))
                            file.write("Other: \n")
                            file.write("Threshold: {}, Qhull errors: {}, total timesteps: {}\n".format(info["threshold"], info["QhullErrors"], info["total_timesteps"]))
                            # file.write("Generators: \n")
                            # file.write(str(info["generators"]) + "\n")
                            file.close()
                        break
                        # self.episode_rewards.append(episode_reward)
                        # self.episode_lengths.append(episode_step_count)
                        # self.episode_mean_values.append(np.mean(episode_values))

                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        # v1 = sess.run(self.local_AC.value,
                        #     feed_dict={self.local_AC.inputs:[s],
                        #     self.local_AC.state_in[0]:rnn_state[0],
                        #     self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        # v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        # episode_buffer = []
                        # sess.run(self.update_local_ops)

                episode_lengths[self.number].append(episode_step_count)
                episode_rewards[self.number].append(episode_reward)
                episode_mean_values[self.number].append(np.nanmean(episode_values))
                # Update the network using the episode buffer at the end of the episode.
                # if len(episode_buffer) != 0:
                v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                # if episode_count % 5 == 0 and episode_count != 0:
                    # if self.name == 'worker_0' and episode_count % 25 == 0:
                    #     time_per_step = 0.05
                    #     images = np.array(episode_frames)
                    #     make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                    #         duration=len(images)*time_per_step,true_image=True,salience=False)
                # if episode_count % 250 == 0 and self.name == 'worker_0':
                #   saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                #   print ("Saved Model")


                if episode_count%SUMMARY_WINDOW == 0:
                    if episode_count % 100 == 0:        # Why do we only store worker 0?
                        print ('Saving Model', end='\n')
                        saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                        print ('Saved Model', end='\n')

                    SL = SUMMARY_WINDOW * num_workers
                    mean_reward = np.nanmean(episode_rewards[self.number][-SL:])
                    mean_length = np.nanmean(episode_lengths[self.number][-SL:])
                    mean_value = np.nanmean(episode_mean_values[self.number][-SL:])

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                    summary.value.add(tag='Perf/Length', simple_value=mean_length)
                    summary.value.add(tag='Perf/Value', simple_value=mean_value)

                    summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                    summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                    summary.value.add(tag='Losses/Entropy', simple_value=e_l)
                    summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                    summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                    global_summary.add_summary(summary, int(episode_count))

                    global_summary.flush()

                    if printQ:
                        print('{} Tensorboard updated ({})'.format(episode_count, self.number), end='\r')

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images.astype(np.uint8), '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,episode_reward))
                    else:
                        make_gif(images.astype(np.uint8), '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)

                # if self.name == 'worker_0':
                #   sess.run(self.increment)
                episode_count += 1
                print("Episode Done: ", episode_count)


################# MAIN #########################
tf.reset_default_graph()
print("Hello World")

# Create directories

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(logfile_path):
    os.makedirs(logfile_path)

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')




# with tf.device("/gpu:0"):
with tf.device("/cpu:0"):
    master_network = ACNet(scope = GLOBAL_NET_SCOPE, GRID_SIZE = GRID_SIZE, a_size = 2, trainer = None, TRAINING = False, GLOBAL_NET_SCOPE = GLOBAL_NET_SCOPE) # Generate global network

    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr=tf.divide(tf.constant(LR_Q),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
    else:
        lr=tf.constant(LR_Q)

    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
    #trainer = tf.train.experimental.enable_mixed_precision_graph_rewrite(trainer)

    # trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    if TRAINING:
        num_workers = NUM_THREADS # Set workers # = # of available CPU threads
    else:
        num_workers = 1

    # num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    # print("Number of workers: ", num_workers)
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i, GRID_SIZE, a_size, trainer, model_path, logfile_path, global_episodes))
    # saver = tf.train.Saver(max_to_keep=5)
    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=1)


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            # if not TRAINING:
            with open(MODEL +'/checkpoint', 'w') as file:
                file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                file.close()
            ckpt = tf.train.get_checkpoint_state(MODEL)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            episode_count=MODEL_NUMBER # int(p)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ",episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
                trainer = tf.train.experimental.enable_mixed_precision_graph_rewrite(trainer)


        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            # sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])
