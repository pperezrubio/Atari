# Copyright (c) 2015 lautimothy, ev0
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import sys
import numpy as np
import scipy as sp
import theano as thn
import theano.tensor as tn
import scipy.stats as stats
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import convolve2d as conv2d
from scipy.signal import correlate2d as corr2d
from mlp import *
from util import *
import theano.tensor.signal.downsample as downsample


def maxpool(data, factor):
	"""
	Perform average pooling of a given factor on data.
	The image size must be divisible by the factor.

	Args:
	-----
		data: A k x N x m x n array of feature maps.
		factor: A tuple specifying the pooling factor.

	Returns:
	--------
		An k x N x (m/p1) x (n/p2) array of feature maps.
	"""
	x = tn.dtensor4('x')
	f = thn.function([x], downsample.max_pool_2d(x, factor))
	return f(data)


def maxupsample(error, data, factor):

	x = tn.dtensor4('x')
	f = thn.function([x], downsample.max_pool_2d_same_size(x, factor))
	activations = f(data)/data
	return np.kron(error, np.ones(factor)) * activations


class ConvLayer():
	"""
	A Convolutional layer.
	"""

	def __init__(self, k, ksize, pfctr, n):
		"""
		Initialize convolutional layer.

		Args:
		-----
			k: No. feature maps in layer.
			ksize: Tuple repr. the size of a kernel.
			msize: Tuple repr. the size of a feature map.
			pfctr: Pooling factor.
		"""
		self.pfctr = pfctr
		self.no_in = n
		fout = k * np.prod(ksize) / np.prod(pfctr)
		self.kernels = (6.0 / (self.no_in + fout)) * np.random.randn(k, ksize[0], ksize[1])
		#self.bias = 0.01 * np.random.randn(k, 1, msize[0], msize[1])
		self.bias = 0.01 * np.random.randn(k, 1, 1, 1)


	def bprop(self, dE):
		"""
		Compute error gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dE: A k1 x N x m2 x n2 array of errors from prev layers.

		Returns:
		-------
			A k2 x N x m1 x n1 array of errors.
		"""
		k, N, m, n = dE.shape
		dEds = maxupsample(dE, np.tanh(self.maps + self.bias), self.pfctr) * sech2(self.maps + self.bias)

		self.dEdw = np.zeros((self.kernels.shape))
		x = np.sum(np.sum(self.x, axis=0), axis=0) #Sum all planes, then sum all instances.
		for i in xrange(k):
			self.dEdw[i] = np.rot90(corr2d(x, np.sum(dEds[i], axis=0), "valid"), 2)

		self.dEdb = np.sum(np.sum(np.sum(dEds, axis=1), axis=1), axis=1).reshape(self.bias.shape)

		k1, N, m1, n1 = self.x.shape
		self.dEdx = np.zeros((k1, N, m1, n1))
		for l in xrange(N):
			for i in xrange(k1):
				for j in xrange(k):
					self.dEdx[i, l] += corr2d(dEds[j, l], self.kernels[j], "full")

		return self.dEdx


	def update(self, lr):
		"""
		Update the weights in this layer.

		Args:
		-----
			lr: Learning rate.
		"""
		k1, N, m1, n1 = self.x.shape
		self.kernels = self.kernels - (lr * self.dEdw/N)
		self.bias = self.bias - (lr * self.dEdb/N)


	def feedf(self, data):
		"""
		Perform a forward pass on the input data.

		Args:
		-----
			data: An k1 x N x m1 x n1 array of input plains.

		Returns:
		-------
			A k2 x N x m2 x n2 array of output plains.
		"""
		l, N, m, n = data.shape
		k, f1, f2 = self.kernels.shape
		self.maps = np.zeros((k, N, m - f1 + 1, n - f2 + 1))
		self.x = data

		for i in xrange(k):
			for e in xrange(N):
				for j in xrange(l):
					self.maps[i, e] += conv2d(data[j, e], self.kernels[i], "valid") 

		return maxpool(np.tanh(self.maps + self.bias), self.pfctr)


class Cnn():
	"""
	Convolutional neural network class.
	"""

	def __init__(self, layers):
		"""
		Initialize network.

		Args:
		-----
			layers: Dict of fully connected and convolutional layers arranged heirarchically.
		"""
		self.layers = deepcopy(layers["fully-connected"]) + deepcopy(layers["convolutional"])
		self.div_x_shape = None
		self.div_ind = len(layers["fully-connected"])


	def train(self, train_data, train_label, valid_data, valid_label, test_data, test_label, params):
		"""
		Train the convolutional neural net on the training and validation set.

		Args:
		-----
			train_data 	:	no_imgs x img_length x img_width array of images.
			valid_data 	:	no_imgs x img_length x img_width array of images.
			train_label :	k x no_imgs binary array of image class labels.
			valid_label :	k x no_imgs binary array of image class labels.
			params 		:	A dictionary of training parameters.
		"""
		N, m, n = train_data.shape

		for i in xrange(N):

			pred = self.predict(train_data[i].reshape(1, m, n))
			label = np.zeros(pred.shape)
			label[:, train_label[i]] = 1

			self.backprop(pred - label) #Note: dw = w - lr * dEdw
			self.update(params)

			valid_clfn = self.classify(self.predict(valid_data))
			valid_ce = mce(valid_clfn, valid_label)

			print '\rIteration:' + "{:10.2f}".format(i) + ' Valid MCE:' + "{:10.2f}".format(valid_ce)
			if i != 0 and i % 100 == 0:
  				print '\n'

  		test_clfn = self.classify(self.predict(test_data))
  		test_ce = mce(test_clfn, test_label)
  		print '\rTest MCE:' + "{:10.2f}".format(test_ce)

  		return 0

	def backprop(self, dE):
		"""
		Propagate the error gradients through the network.

		Args:
		-----
			dE: A no_imgs x k_classes array of error gradients.
		"""
		error = dE.T
		for layer in self.layers[0 : self.div_ind]:
			error = layer.bprop(error)

		#Reshape output from fully-connected layer.
		k, N, m, n = self.div_x_shape #Check N == N.
		temp = np.zeros((k, N, m, n))
		for i in xrange(N):
			temp[:, i, :, :] = error[:, i].reshape(k, m, n)

		error = temp
		for layer in self.layers[self.div_ind:]:
			error = layer.bprop(error)


	def predict(self, imgs):
		"""
		Return the probability distribution over the class labels for
		the given images.

		Args:
		-----
			data: A no_imgs x img_length x img_width array.

		Returns:
		-------
			A no_imgs x k_classes array.
		"""
		N, m, n = imgs.shape
		pdata = imgs.reshape(1, N, m, n)
		for i in xrange(len(self.layers) - 1, self.div_ind - 1, -1):
			pdata = self.layers[i].feedf(pdata)

		#Reshape output of convolutional layer.
		k, N, m, n = pdata.shape
		temp = np.zeros((k * m * n, N))
		self.div_x_shape = k, N, m, n

		for i in xrange(N):
			temp[:, i] = pdata[:, i, :, :].ravel()

		pdata = temp
		for i in xrange(self.div_ind - 1, -1, -1):
			pdata = self.layers[i].feedf(pdata)

		return pdata.T


	def update(self, params):
		"""
		Update the network weights.

		Args:
		-----
			params: Training parameters.
		"""
		for layer in self.layers:
			layer.update(params['learn_rate'])


	def classify(self, prediction):
		"""
		Peform binary classification on the class probabilities.

		Args:
		-----
			prediction: An N x k array of class probabilities.

		Returns:
		--------
			An N x k array of binary class assignments.
		"""
		N, k = prediction.shape
		clasfn = np.zeros((N, 1))

		for row in xrange(N):
			ind = np.where(prediction[row] == np.amax(prediction[row]))[0][0]
			#prediction[row, :] = np.zeros(k)
			clasfn[row, 0] = ind

		return clasfn


def testMnist(filename):
	"""
	Test cnn using all the mnist digits.

	Args:
	-----
		filename: Name of file containing mnist digits.
	"""
	print "Loading data..."
	data = np.load(filename)
	train_data = data['train_data'][29000:31000]
	valid_data = data['valid_data'][0:1000]
	test_data = data['test_data']
	train_label = data['train_label'][29000:31000].reshape(2000, 1)
	valid_label = data['valid_label'][0:1000].reshape(1000, 1)
	test_label = data['test_label'].reshape(10000, 1)

	print "Initializing network..."
	layers = {
		"fully-connected": [PerceptronLayer(10, 100, "softmax"), PerceptronLayer(100, 256, "tanh")],
		# Ensure size of output maps in preceeding layer
		# is equals to the size of input maps in next layer.
		"convolutional": [ConvLayer(16, (5,5), (2,2), 6), ConvLayer(6, (5,5), (2,2), 1)]
		#"convolutional": [ConvLayer(16, (5,5), (8,8), (2,2), 6), ConvLayer(6, (5,5), (24,24), (2,2), 1)]
	}
	cnn = Cnn(layers)
	print "Training network..."
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, {'learn_rate': 0.1})


if __name__ == '__main__':

	testMnist('data/cnnMnist2.npz')
