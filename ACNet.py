import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D ,Flatten
import numpy as np
from tensorflow.nn.rnn_cell import BasicLSTMCell

#parameters for training
GRAD_CLIP              = 32.
KEEP_PROB1             = 1 # was 0.5
KEEP_PROB2             = 1 # was 0.7

RNN_SIZE               = 512
# GOAL_SIZE              = 2
loc_layer_size         = 2

# glimpse_size1 = 11
# glimpse_size2 = 22
# glimpse_size3 = 32

'''
CHANGES
- changed num_channels = 1
'''
num_channels  = 3
# fov_size      = 3
# loc_std       = 0.8


#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

'''
G(x) =  	1	  		  {-(1/2)*[(x-u)/sigma]^2}
	 ------------------- e
	 sigma*(2*pi)^(1/2)
'''
def gaussian_pdf(mean, loc_std, sample):
	Z = 1.0 / (loc_std * tf.sqrt(2.0 * np.pi))
	a = - tf.square(sample - mean) / (2.0 * tf.square(loc_std))
	return Z * tf.exp(a)

class ACNet:
	def __init__(self, scope, GRID_SIZE, a_size, trainer,TRAINING, GLOBAL_NET_SCOPE):
		with tf.variable_scope(str(scope)+'/qvalues'):
			#The input size may require more work to fit the interface.
			self.inputs = tf.placeholder(shape=[None,GRID_SIZE,GRID_SIZE, num_channels], dtype=tf.float32)		# input state
			# self.goal_pos = tf.placeholder(shape=[None,2],dtype=tf.float32)
			self.prev_loc = tf.placeholder(shape=[None,2], dtype=tf.float32)

#            self.policy, self.next_loc, self.value, self.state_out, self.state_in, self.state_init, self.valids, self.blocking, self.mypos, self.goalpos, self.next_loc_mean = self._build_net(self.inputs, self.inputs_primal, self.prev_loc, RNN_SIZE, TRAINING,a_size)

			'''
			CHANGES
			- removed target_blocking, blocking layers, blocking_loss
			- removed imitation gradients and losss
			- removed valid_loss
			- removed train_valid
			- commented out policy loss (since, discrete)
			- next_loc_loss is now new policy loss
			- responsible_next_loc is NOW policy

			'''
			self.value,  self.next_loc_mean, self.loc_std, self.next_loc, self.state_out, self.state_in, self.state_init = self._build_net(self.inputs, self.prev_loc, RNN_SIZE, TRAINING, a_size)  # self.goal_pos


		if TRAINING:
			self.target_v               = tf.placeholder(tf.float32, [None], 'Vtarget')
			self.advantages             = tf.placeholder(shape=[None], dtype=tf.float32)

			self.sampled_next_locs      = tf.placeholder(tf.float32, [None,2])									# sampled action is stored here
			self.policy   = gaussian_pdf(self.next_loc_mean, self.loc_std, self.sampled_next_locs)		 # Distribution == Policy


			# Loss Functions
			self.value_loss     = 0.5*tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))

			# H(x) = Sum[p(x)*log(p(x))]
			self.entropy        = - 0.01 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-10,1.0)))

			self.policy_loss  = - 0.2 * tf.reduce_sum( tf.log(tf.clip_by_value(self.policy[:,0],1e-15,1.0)) * self.advantages + tf.log(tf.clip_by_value(self.policy[:,1],1e-15,1.0)) * self.advantages)

			#For Normal RL Part
			self.loss           = self.value_loss + self.policy_loss - self.entropy		# removed self.blocking_loss, valid_loss, discrete_policy _loss         #+ 0.5*self.mypos_loss + 0.5*self.goalpos_loss

			#For Imitation Learning Part
			# self.bc_loss          = 0.5 * tf.reduce_mean(tf.contrib.keras.backend.categorical_crossentropy(self.optimal_actions_onehot,self.policy))
                        # self.next_loc_loss_il = 0.2 * tf.reduce_sum(tf.sqrt(tf.square(self.next_loc_mean[:-1,:] - self.il_nextloc)))

			# self.imitation_loss   = self.bc_loss    #+ self.next_loc_loss_il

			# Get gradients from local network using local losses and
			# normalize the gradients using clipping
			local_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvalues')
			self.gradients     = tf.gradients(self.loss, local_vars)
			self.var_norms     = tf.global_norm(local_vars)
			grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

			# Apply local gradients to global network
			global_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
			self.apply_grads   = trainer.apply_gradients(zip(grads, global_vars))

			#now the gradients for imitation loss
			# self.i_gradients     = tf.gradients(self.imitation_loss, local_vars)
			# self.i_var_norms     = tf.global_norm(local_vars)
			# i_grads, self.i_grad_norms = tf.clip_by_global_norm(self.i_gradients, GRAD_CLIP)

			# Apply local gradients to global network
			# self.apply_imitation_grads   = trainer.apply_gradients(zip(i_grads, global_vars))
		print("Hello World... From  "+str(scope))     # :)

	def _build_net(self, inputs, prev_loc, RNN_SIZE, TRAINING, a_size): # goal_pos

		'''
		CHANGES
		- Added one more block consisting of 3 conv layers and 1 max pool layer
		- kernel size was changed (3,3) -> (8,8), strides from 1 to 4, to get 1 x 1 in last layer
		- removed policy layers
		'''
		w_init   = tf.contrib.layers.variance_scaling_initializer()

		# glimpse1 = tf.image.extract_glimpse(inputs, [glimpse_size1,glimpse_size1], self.prev_loc, centered=True, normalized=True)

		# glimpse2 = tf.image.extract_glimpse(inputs, [glimpse_size2,glimpse_size2], self.prev_loc, centered=True, normalized=True)
		# glimpse2 = tf.image.resize(glimpse2,        [glimpse_size1,glimpse_size1])

		# glimpse3 = tf.image.extract_glimpse(inputs, [glimpse_size3,glimpse_size3], self.prev_loc, centered=True, normalized=True)
		# glimpse3 = tf.image.resize(glimpse3,        [glimpse_size1,glimpse_size1])

		# self.glimpses = tf.concat([glimpse1,glimpse2,glimpse3],axis=-1)

		# Block 1
		conv1a   =  Conv2D(padding="same", filters=RNN_SIZE//8, kernel_size=[8, 8],   strides=4, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(self.inputs)
		conv1b   =  Conv2D(padding="same", filters=RNN_SIZE//8, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv1a)
		conv1c   =  Conv2D(padding="same", filters=RNN_SIZE//8, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv1b)
		pool1    =  MaxPool2D(pool_size=[2,2])(conv1c)

		# Block 2
		conv2a   =  Conv2D(padding="same", filters=RNN_SIZE//4, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(pool1)
		conv2b   =  Conv2D(padding="same", filters=RNN_SIZE//4, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv2a)
		conv2c   =  Conv2D(padding="same", filters=RNN_SIZE//4, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv2b)
		pool2    =  MaxPool2D(pool_size=[2,2])(conv2c)

		# Block 3
		conv3a   =  Conv2D(padding="same", filters=RNN_SIZE//2, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(pool2)
		conv3b   =  Conv2D(padding="same", filters=RNN_SIZE//2, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv3a)
		conv3c   =  Conv2D(padding="same", filters=RNN_SIZE//2, kernel_size=[3, 3],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=tf.nn.relu)(conv3b)
		pool3    =  MaxPool2D(pool_size=[2,2])(conv3c)

		# final convolutional layer
		#removed GOAL_SIZE
		conv4    =  Conv2D(padding="valid", filters=RNN_SIZE-loc_layer_size, kernel_size=[2, 2],   strides=1, data_format='channels_last', kernel_initializer=w_init,activation=None)(pool3)

		# FC layers
		flat1a   =  Flatten(data_format='channels_last')(conv4)
		#removed GOAL_SIZE
		flat1b   =  Dense(units=RNN_SIZE-loc_layer_size)(flat1a)

		# FC layers for goal_pos input
		# goal_layer1 = Dense(units=GOAL_SIZE)(goal_pos)
		# goal_layer2 = Dense(units=GOAL_SIZE)(goal_layer1)

		# FC layers to find next location
		loc_layer1  = Dense(units=loc_layer_size)(prev_loc)
		loc_layer2  = Dense(units=loc_layer_size)(loc_layer1)

		# Concatenationation of above layers, followed by FC layer
		concat   = tf.concat([flat1b, loc_layer2],1) # goal_layer2
		h1       = Dense(units=RNN_SIZE)(concat)
		h2       = Dense(units=RNN_SIZE)(h1)
		self.h3  = tf.nn.relu(h2+concat)

		#Recurrent network for temporal dependencies
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE,state_is_tuple=True)
		c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
		h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
		state_init = [c_init, h_init]
		c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
		h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
		state_in = (c_in, h_in)
		rnn_in = tf.expand_dims(self.h3, [0])
		step_size = tf.shape(inputs)[:1]
		state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
		lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
		lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
		time_major=False)
		lstm_c, lstm_h = lstm_state
		state_out = (lstm_c[:1, :], lstm_h[:1, :])
		self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])

		'''
		CHANGES
		- removed blocking layer
		- edited out stop_gradient lines (Dont need them)
		'''
		# Value FC
		value         = Dense(units=1, kernel_initializer=normalized_columns_initializer(1.0), bias_initializer=None, activation=None)(inputs=self.rnn_out)

		# rnn_out_frozen = tf.stop_gradient(self.rnn_out)
		next_loc_mean  = Dense(units=2, kernel_initializer=normalized_columns_initializer(1.0), bias_initializer=None, activation=tf.math.tanh)(inputs=self.rnn_out)	# was rnn_out_frozen
		loc_std        = Dense(units=1, kernel_initializer=normalized_columns_initializer(1.0), activation=tf.nn.softplus)(inputs = self.rnn_out)

		# Policy FC
		next_loc       = tf.clip_by_value(next_loc_mean + tf.random_normal([1,2], 0, loc_std), -1, 1)
		# next_loc       = tf.stop_gradient(next_loc)

		return value, next_loc_mean, loc_std, next_loc, state_out, state_in, state_init
