# testing trained model
import tensorflow as tf
import gym
import gym_MAOC
from ACNet import *

model_path = 'model'
env = gym.make('MAOC-v0')
MODEL_NUMBER = 1500

sess = tf.Session()
master_network = ACNet(scope = 'global', GRID_SIZE = 64, a_size = 2, trainer = None, TRAINING = False, GLOBAL_NET_SCOPE = 'global')
saver = tf.train.Saver(max_to_keep=1)

print("Loading model")
with open(model_path+'/checkpoint', 'w') as file:
	file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
	file.close()
	ckpt = tf.train.get_checkpoint_state(model_path)
	p=ckpt.model_checkpoint_path
	p=p[p.find('-')+1:]
	p=p[:p.find('.')]
	episode_count=int(p)
	saver.restore(sess,ckpt.model_checkpoint_path)


for i in range(5):
	state, next_loc = env.reset()
	next_loc = np.asarray(next_loc, dtype=np.float32)
	for i in range(2):
		next_loc[0,i] = ((next_loc[0,i]/63.)*2.)-1
	rnn_state = master_network.state_init

	episode_reward = 0
	total_steps = 0
	done = False
	while True:
		env.render()
		state = state.reshape(1, state.shape[0], state.shape[1], 1)

		action, value, rnn_state = sess.run([master_network.action,master_network.value,master_network.state_out], 
						feed_dict={master_network.inputs:state,
						master_network.prev_loc: next_loc,
						master_network.state_in[0]:rnn_state[0],
						master_network.state_in[1]:rnn_state[1]})

		action = np.expand_dims(np.asarray(action), axis=0)

		next_state, reward, done, info = env.step(action)
		episode_reward += reward
		state = next_state                    
		total_steps += 1
		next_loc = action
		if done or total_steps == 32:
			break

	print('Episode length', total_steps, 'Episode reward', episode_reward)

