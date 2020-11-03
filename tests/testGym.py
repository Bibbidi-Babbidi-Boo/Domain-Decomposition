import gym
import numpy as np
import gym_MAOC

env = gym.make('MAOC-v0')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
t = 0
observation, first_point = env.reset()
num_episodes = 100
count = 0


while count < num_episodes:
    t += 1
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Observation at location 0,0:")
    print(info)
    if done:
        print("==========DONE==========")
        env.render()
        print("Episode finished after {} timesteps".format(t+1))
        print("Last reward: {}".format(reward))
        t = 0
        count += 1
        observation, first_point = env.reset()
env.close()
