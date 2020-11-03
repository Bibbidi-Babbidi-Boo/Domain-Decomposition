import gym
import numpy as np 
import gym_MAOC
import imageio


env = gym.make('MAOC-v0')

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")
def checkDone(env):
    notCollCount = 0
    for region in env.current_regions:
        print("Region {} is coll? {}".format(region.label, region.coll))
        if not region.coll:
            notCollCount += 1
    print("Non-collectible regions: {}".format(notCollCount))


genList = [np.array([[30, 52]], dtype=np.int32), np.array([[ 26, 54]], dtype=np.int32), np.array([[ 28, 39]], dtype=np.int32), 
           np.array([[ 41, 4]], dtype=np.int32), np.array([[34, 13]], dtype=np.int32), np.array([[41, 55]], dtype=np.int32)]



observation, first_point = env.reset()
agentMap = np.load("agentMap.npy")
env.agentMap = agentMap
env.generators.pop()
frames = []
frames.append(env.render())
for gen in genList:
    print("=== Generator: {} ===".format(gen))
    normGen = env.mapActionToNormAction(gen)
    observation, reward, done, info = env.step(normGen)
    # print("Reward: {}".format(reward))
    print("Done: {}".format(done))
    # print("Info: ")
    # print(info)
    # env.render()
    env.plot2D(env.renderMap)
    checkDone(env)
    # print("Number of non-collectible locations: {}".format(np.where(env.renderMap == env.NOTCOLL_NORM)[0].shape))
    # frames.append(env.render())

# labeledDiagram = env.descriptiveDiagram
# env.plot2D(labeledDiagram)


# np.save('labeledDiagramNoLines.npy', labeledDiagram)
# images = np.array(frames)
# make_gif(images.astype(np.uint8), 'testAgents.gif')

# observation = env.reset()
# genList = [np.array([[10, 24]]), np.array([[10, 48]]), np.array([[10, 56]])]
# r = env.testReward(genList)
# env.render()


# env.plot2D(env.agentMap)