from __future__ import division
import numpy as np
import tensorflow as tf
import os
import shutil


def mix_activation(x):
	"""modified hyperbolic tangent activation"""
	return 0.7*tf.tanh(x) + 0.3*x

# WGAN-GP model
class Gan():
	"""WGAN-GP model"""

	def __init__(self, data_input, data, lam=10, num_hidden=512, batch_size=200, num_epochs=100, lr_rateg=1e-4, lr_rated=1e-4, lr_decay=1.0, to_restore=False, output_path='WGAN-GP', net_type='FC'):
		self.data_input = data_input
		self.data = data # training data, each row is an instance
		self.num_hidden = num_hidden
		self.lam = lam # the parameter for gradient penalty
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.lr_rateg = lr_rateg
		self.lr_rated = lr_rated
		self.lr_decay = lr_decay
		self.to_restore = to_restore
		self.output_path = output_path
		self.net_type = net_type
		self.disc_loss_history = []
		self.epoch = 0


	def generator(self, noise_input):
		"""Generator"""
		"""
		z is noise, and generator transform z to the target distribution
		"""
		num_hidden = self.num_hidden
		output_size = self.data.shape[1]
		if self.net_type == 'FC':
			with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
				h1 = tf.nn.relu(tf.layers.dense(noise_input, num_hidden))
				h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
				h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
				h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
				output = tf.layers.dense(h4, output_size)
		elif self.net_type == 'conv':
			with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
				h1 = tf.nn.relu(tf.layers.dense(noise_input, 3*3*64))
				fold = tf.reshape(h1, [-1, 3, 3, 64])
				h2 = tf.nn.relu(tf.layers.conv2d_transpose(fold, filters=128, kernel_size=2, strides=1, padding='valid'))
				b2 = tf.layers.batch_normalization(h2, training = True)
				h3 = tf.nn.relu(tf.layers.conv2d_transpose(b2, filters=128, kernel_size=2, strides=1, padding='valid'))
				b3 = tf.layers.batch_normalization(h3, training = True)
				h4 = tf.nn.relu(tf.layers.conv2d_transpose(b3, filters=64, kernel_size=2, strides=1, padding='valid'))
				b4 = tf.layers.batch_normalization(h4, training = True)
				h5 = tf.nn.relu(tf.layers.conv2d_transpose(b4, filters=64, kernel_size=2, strides=1, padding='valid'))
				b5 = tf.layers.batch_normalization(h5, training = True)
				h6 = tf.nn.relu(tf.layers.conv2d_transpose(b5, filters=32, kernel_size=2, strides=1, padding='valid'))
				flatten = tf.reshape(h6, [-1, 8*8*32])
				h7 = tf.nn.relu(tf.layers.dense(flatten, num_hidden))
				h8 = tf.nn.relu(tf.layers.dense(h7, num_hidden))
				output = tf.layers.dense(h8, output_size)
		else:
			raise ValueError("net_type is not supported")
		return output

	def discriminator(self, x):
		"""Discriminator"""
		"""
		x is fake or true data 
		"""
		if self.net_type == 'FC':
			with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
				num_hidden = self.num_hidden
				h1 = tf.nn.relu(tf.layers.dense(x, num_hidden))
				h2 = tf.nn.relu(tf.layers.dense(h1, num_hidden))
				h3 = tf.nn.relu(tf.layers.dense(h2, num_hidden))
				h4 = tf.nn.relu(tf.layers.dense(h3, num_hidden))
				output = tf.layers.dense(h4, 1)
		elif self.net_type == 'conv':
			with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
				num_hidden = self.num_hidden
				x = tf.reshape(x, [-1, 17, 17, 1])
				h1 = mix_activation(tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same'))
				b1 = tf.layers.batch_normalization(h1, training=True)
				h2 = mix_activation(tf.layers.conv2d(b1, filters=128, kernel_size=3, strides=1, padding='same'))
				b2 = tf.layers.batch_normalization(h2, training=True)
				h3 = mix_activation(tf.layers.conv2d(b2, filters=64, kernel_size=3, strides=1, padding='same'))
				b3 = tf.layers.batch_normalization(h3, training=True)
				h4 = mix_activation(tf.layers.conv2d(b3, filters=64, kernel_size=3, strides=2, padding='same'))
				b4 = tf.layers.batch_normalization(h4, training=True)
				h5 = mix_activation(tf.layers.conv2d(b4, filters=32, kernel_size=3, strides=2, padding='same'))
				b5 = tf.layers.batch_normalization(h5, training=True)
				h6 = mix_activation(tf.layers.conv2d(b5, filters=32, kernel_size=3, strides=2, padding='same'))
				flatten = tf.reshape(h6, [-1, 3*3*32])
				h7 = mix_activation(tf.layers.dense(flatten, num_hidden))
				h8 = mix_activation(tf.layers.dense(h7, num_hidden))
				output = tf.layers.dense(h8, 1)
		else:
			raise ValueError("net_type is not supported")
		return output

	def train(self):
		"""train step"""
		z = tf.placeholder(tf.float32, shape=[None, self.data_input.shape[1]])
		real_data = tf.placeholder(tf.float32, shape=[None, self.data.shape[1]])
		fake_data = self.generator(z)

		disc_real = self.discriminator(real_data)
		disc_fake = self.discriminator(fake_data)

		# WGAN-loss
		disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) 
		gene_loss = -tf.reduce_mean(disc_fake) + self.lam * tf.reduce_mean((real_data-fake_data)**2)

		# gradient penalty
		alpha = tf.random_uniform([self.batch_size, 1], minval=0., maxval=1.)
		interpolates = alpha * real_data + (1 - alpha) * fake_data
		disc_interpolates = self.discriminator(interpolates)
		gradients = tf.gradients(disc_interpolates, [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1)**2)

		# WGAN-GP loss
		disc_loss += self.lam * gradient_penalty

		# extract training variables
		gene_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

		# set optimizer
		disc_train_op = tf.train.AdamOptimizer(self.lr_rated).minimize(disc_loss, var_list=disc_vars)
		gene_train_op = tf.train.AdamOptimizer(self.lr_rateg).minimize(gene_loss, var_list=gene_vars)

		# make dir to save model
		saver = tf.train.Saver()
		if self.to_restore:
			chkpt_fname = tf.train.latest_checkpoint(self.output_path)
			with tf.Session() as sess:
				saver.restore(sess, chkpt_fname)
		else:
			if os.path.exists(self.output_path):
				shutil.rmtree(self.output_path)
			os.mkdir(self.output_path)

		# Training loop
		critic_iters = 5
		num_data = self.data.shape[0]
		iter_per_epoch = max(num_data//self.batch_size, 1)
		num_iters = self.num_epochs * iter_per_epoch # total iterations

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for idx_iter in range(num_iters):
				batch_mask = np.random.choice(num_data, self.batch_size) # for batch gradient
				batch_data = self.data[batch_mask]
				batch_data_input = self.data_input[batch_mask]
				if idx_iter > 0:
					_ = sess.run(gene_train_op,
								feed_dict={z: batch_data_input, real_data: batch_data})
				for idx_critic in range(critic_iters):
					disc_loss_cur, _ = sess.run([disc_loss, disc_train_op],
												feed_dict={z: batch_data_input, real_data: batch_data})
					self.disc_loss_history.append(disc_loss_cur)
				print('iteration: {}, disc_loss: {:.4}'. format(idx_iter+1, self.disc_loss_history[-1]))

				saver.save(sess, os.path.join(self.output_path, "model"), global_step=idx_iter)
				epoch_end = (idx_iter + 1) % iter_per_epoch == 0
				if epoch_end:
					self.epoch += 1
					self.lr_rated *= self.lr_decay

			opt_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
			weightmat_ranks = []
			singular_values = []
			for k in range(len(opt_weights)):
				opt_weights[k] = opt_weights[k].eval()
				if k%2 == 0:
					mat_rank = np.linalg.matrix_rank(opt_weights[k])
					_, s, _ = np.linalg.svd(opt_weights[k])
					weightmat_ranks.append(mat_rank)
					singular_values.append(s)
			np.save('singular_values.npy', singular_values)
			np.save('weightmat_ranks.npy', weightmat_ranks)

	def generate_sample(self, batch_times=1):
		"""generate samples using trained model"""
		"""generate batch_times * batch_size samples using Generator"""
		data_dim = self.data.shape[1]
		#generate_samples = np.zeros((batch_times*self.batch_size, data_dim))
		z_test = tf.random_uniform([batch_times*self.batch_size, self.data_input.shape[1]], minval=0.01, maxval=1.0)

		chkpt_fname_final = tf.train.latest_checkpoint(self.output_path)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, chkpt_fname_final)
			generate_data = self.generator(z_test)
			generate_samples = sess.run(generate_data)
		return generate_samples
