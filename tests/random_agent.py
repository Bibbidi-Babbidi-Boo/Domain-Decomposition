import gym
import numpy as np 
import gym_MAOC
# import imageio


env = gym.make('MAOC-v0')

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")


# genList = [np.array([[53, 56]], dtype=np.int32), np.array([[ 0, 63]], dtype=np.int32), np.array([[ 0, 53]], dtype=np.int32), 
#            np.array([[ 0, 25]], dtype=np.int32), np.array([[10, 63]], dtype=np.int32), np.array([[33, 63]], dtype=np.int32)]


# env.generators.pop()
# frames = []
# frames.append(env.render())

num_episodes = 1000

num_successful = 0

for i in range(num_episodes):
    print("==========")
    print("Episode: {}".format(i))
    done = False
    observation, first_point = env.reset()
    env.render()
    R = 0
    while not done:
        print("========")
        # normGen = env.mapActionToNormAction(gen)

        observation, reward, done, info = env.step(env.action_space.sample())
        print("Reward: {}".format(reward))
        print("Done: {}".format(done))
        R += reward
        print("Info: ")
        print(info)
        env.render()
    if env.curr_timestep < env.max_timesteps:
        num_successful += 1
    print("Episode return = {}".format(R))
    # frames.append(env.render())

print(num_successful)
