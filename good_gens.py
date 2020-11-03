import gym
import numpy as np
import gym_MAOC
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


env = gym.make('MAOC-v0')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
t = 0
observation = env.reset()
num_episodes = 100
count = 0

## Finding the reflection of the point abou x=y and x+y=3
def ref(x1, y1):
    x2, y2 = y1, x1
    x3, y3 = 63-y1, 63-x1
    return [x2, y2], [x3, y3]

## Given previous geenerator finding the new goo generator
def get_good_gens(prev_loc, observation):
    print(prev_loc)
    # exit()
    flag = 0
    ## If we have odd number of points reflect the last point given that it does not fall in a collectible region
    if len(prev_loc)%2 != 0:
        print("IN")
        p1, p2 = ref(prev_loc[len(prev_loc)-1][0], prev_loc[len(prev_loc)-1][1])
        print("PP", p1, p2)
        if (p1[0]-prev_loc[len(prev_loc)-1][0])**2 + (p1[1]-prev_loc[len(prev_loc)-1][1])**2 > (p2[0]-prev_loc[len(prev_loc)-1][0])**2 + (p2[1]-prev_loc[len(prev_loc)-1][1])**2:
            if observation[p1[0]][p1[1]] == 1:
                action = np.array([p1])
            else:
                flag+=1
        else:
            if observation[p1[0]][p1[1]] == 1:
                action = np.array([p2])
            else:
                flag+=1
        if flag != 1:
            print("action", action)
            action = env.mapActionToNormAction(action)
    ## Else sample a new point which does not fall in the collectible regions
    if flag==1 or len(prev_loc)%2 == 0:
        print("OUT")
        while True:
            action = env.action_space.sample()
            var = env.normActionToMapAction(action)
            if observation[var[0][0]][var[0][1]] == 1:
                break
    return action

state, prev_loc = env.reset()
observation = np.array([[1]*64]*64, dtype=np.uint8)
total = 0

while count < num_episodes:
    t += 1
    env.render()
    if t==1:
        action = get_good_gens(prev_loc, observation)
    else:
        gens = np.array(info['generators']).astype('int32')
        gens = np.reshape(gens, (len(gens), 2))
        action = get_good_gens(gens, observation)
    print("GOT", action)
    observation, reward, done, info = env.step(action)
    print(info)
    if done:
        print("==========DONE==========")
        env.render()
        print("Episode finished after {} timesteps".format(t+1))
        print("Last reward: {}".format(reward))
        total+=t
        t = 0
        count += 1
        observation, prev_loc = env.reset()
        print("Total", total, count)
env.close()
