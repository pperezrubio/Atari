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

import numpy as np
from copy import deepcopy
from util import *


def sigmoid(data):
	"""
	Run The sigmoid activation function over the input data.

	Args:
	----
		data : A k x N array.

	Returns:
	-------
		A k x N array.
	"""
	return 1 / (1 + np.exp(-data))


def softmax(data):
	"""
	Run the softmax activation function over the input data.

	Args:
	----
		data : A k x N array.

	Returns:
	-------
		A k x N array.
	"""
	k, N = data.shape
	e = np.exp(data)
	return e/np.sum(e, axis=0).reshape(1, N)


def sech2(data):
	"""
	Find the hyperbolic secant function over the input data.

	Args:
	-----
		data : A k x N array.

	Returns:
	--------
		A k x N array.
	"""
	return np.square(1 / np.cosh(data))


def cross_entropy(preds, labels):
    """
    Compute the cross entropy over the predictions.

    Args:
    -----
        preds : An N x k array of class predicitions.
        labels : An N x k array of class labels.
    
    Returns:
    --------
        The cross entropy.
    """
    N, k = preds.shape

    if k == 1: #sigmoid
        return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
    
    return -np.sum(np.sum(labels * np.log(preds), axis=1).reshape(N, 1))


def mce(preds, labels):
    """
    Compute the mean classification error over the predictions.

    Args:
    ----
        preds : An N x k array of class predictions.
        labels: An N x k array of class labels.
    
    Returns:
    --------
        The mean classification error.
    """
    N, l = labels.shape
    return np.sum(preds != labels)/float(N)


class PerceptronLayer():
	"""
	A perceptron layer.
	"""

	def __init__(self, no_outputs, no_inputs, outputType="tanh"):
		"""
		Initialize fully connected layer.

		Args:
		-----
			no_outputs: No. output classes.
			no_inputs: No. input features.
			outputType: Type of output ('sum', 'sigmoid', 'tanh' or 'softmax')
		"""
		self.o_type = outputType
		if outputType == "sigmoid" or outputType == "tanh":
			self.w = (6.0/(no_outputs + no_inputs)) * np.random.randn(no_outputs, no_inputs)
		else:
			self.w = 0.01 * np.random.randn(no_outputs, no_inputs)
			
		self.b = 0.01 * np.random.randn(no_outputs, 1)
		#if no_outputs <= 2:
			#self.o_type = "sigmoid"

		self.rms = [np.zeros(self.w.shape), np.zeros(self.b.shape), 0]


	def bprop(self, dEds):
		"""
		Compute gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dEdx: A no_output x N array of errors from prev layers.

		Return:
		-------
			A no_inputs x N array of input errors.
		"""
		if self.o_type == 'sigmoid':
			dEdo = dEds
			z = sigmoid(np.dot(self.w, self.x) + self.b)
			dods = z * (1 - z)
			dEds = dEdo * dods
		elif self.o_type == 'tanh':
			dEdo = dEds
			dods = sech2(np.dot(self.w, self.x) + self.b)
			dEds = dEdo * dods 

		self.dEdw = np.dot(dEds, self.x.T)
		self.dEdb = np.sum(dEds, axis=1).reshape(self.b.shape)
		self.dEdx = np.dot(dEds.T, self.w).T
		
		return self.dEdx


	def update(self, lr, useRMSprop):
		"""
		Update the weights in this layer.
		"""
		m, N = self.x.shape
		#print "Before RMS. Shape: ", self.dEdw.shape
		#print "\n"
		#printMatrix(self.dEdw)
		#print "\n"
		self.dEdw = self.dEdw / N
		self.dEdb = self.dEdb / N		

		if useRMSprop:
			n = self.rms[2]
			rmsw = ((n * self.rms[0]) + np.square(self.dEdw)) / (n + 1)
			rmsb = ((n * self.rms[1]) + np.square(self.dEdb)) / (n + 1)
			self.dEdw = self.dEdw / np.sqrt(rmsw)
			self.dEdb = self.dEdb / np.sqrt(rmsb)

			#Handle illegal values
			self.dEdw[np.where(np.isnan(self.dEdw))] = 0
			self.dEdb[np.where(np.isnan(self.dEdb))] = 0
			self.dEdw[np.where(np.isinf(self.dEdw))] = 0
			self.dEdb[np.where(np.isinf(self.dEdb))] = 0

			#Update averages
			self.rms[2] = self.rms[2] + 1
			self.rms[0] = rmsw
			self.rms[1] = rmsb

		#print "RMS", n
		#printMatrix(rmsw)

		#print "After RMS. Shape: ", self.dEdw.shape
		#print "\n"
		#printMatrix(self.dEdw)
		#print "\n"

		#print "Weight. Shape: ", self.w.shape
		#print "\n"
		#printMatrix(self.w)
		#print "\n"

		#print "Weight update. Shape: ", self.dEdw.shape
		#print "\n"
		#printMatrix(lr * self.dEdw)
		#print "\n"

		#Update weights
		self.w = self.w - (lr * self.dEdw)
		self.b = self.b - (lr * self.dEdb)


	def feedf(self, data):
		"""
		Perform a forward pass on the input data.

		Args:
		-----
			data: An no_inputs x N array.

		Return:
		-------
			A no_outputs x N array.
		"""
		self.x = data
		s = np.dot(self.w, self.x) + self.b

		if self.o_type == "sum":
			return s
		elif self.o_type == "tanh":
			return np.tanh(s)
		elif self.o_type == "softmax": 
			return softmax(s)

		return sigmoid(s)


class Mlp():
	"""
	A multilayer perceptron.
	"""

	def __init__(self, layers):
		"""
		Initialize the mlp.

		Args:
		-----
			layers: List of mlp layers arranged heirarchically.
		"""
		self.layers = deepcopy(layers)


	def train(self, train_data, train_target, valid_data, valid_target, test_data, test_target, hyperparameters):
		"""
		Train the mlp on the training set and validation set using the provided
		hyperparameters.

		Args:
		----
			train_data 	:	no_instance x no_features matrix.
			train_target :	no_instance x k_class matrix.
			valid_data 	:	no_instance x no_features matrix.
			valid_target :	no_instance x k_class matrix.array shuffle numpy
			hyperparameters :	A dictionary of training parameters.
		"""
		N, m1 = train_data.shape
		N, m2 = train_target.shape

		# Train the network with batch 'cos
		# online too erratic & mini-batch too much work.
		epochs = hyperparameters['epochs']

		for epoch in xrange(epochs):

			self.backprop(self.predict(train_data) - train_target)
			self.update(hyperparameters)

			#Measure network's performance.
			train_class = self.classify(self.predict(train_data))
			valid_class = self.classify(self.predict(valid_data))
			ce_train = mce(train_class, train_target)
			ce_valid = mce(valid_class, valid_target)
			print '\rEpoch' + "{:10.2f}".format(epoch) + ' Train MCE:' + "{:10.2f}".format(ce_train) + ' Validation MCE:' + "{:10.2f}".format(ce_valid)
			if epoch != 0 and epoch % 100 == 0:
  				print '\n'

  		test_class = self.classify(self.predict(test_data))
		ce_test = mce(test_class, test_target)
  		print '\r Test MCE:' + "{:10.2f}".format(ce_test)

  		return 0


  	def backprop(self, dEds):
  		"""
  		Propagate the error gradients through the network.

  		Args:
  		-----
  			dEds: Error gradient w.r.t WX.
  		"""
  		error = dEds.T
  		for i in xrange(len(self.layers)):
  			error = self.layers[i].bprop(error)


  	def update(self, parameters):
  		"""
  		Update the network weights using the training
  		parameters.

  		Args:
  		-----
  			parameters: Training parameters.
  		"""
  		for layer in self.layers:
  			if 'use_RMSprop' in parameters.keys():
  				layer.update(parameters['learn_rate'], parameters['use_RMSprop'])
  			else:	
  				layer.update(parameters['learn_rate'], False)


  	def predict(self, data):
  		"""
  		Return the posterior probability distribution of data
  		over k classes.

  		Args:
  		-----
  			data: An N x k array of input data.

  		Returns:
  		-------
  			An N x k array of posterior distributions.
  		"""
  		x = data.T
  		for i in xrange(len(self.layers) - 1, 0, -1):
  			x = self.layers[i].feedf(x)

  		return self.layers[0].feedf(x).T


  	def classify(self, prediction):
		"""
		Peform classification using the class predictions of the classifier.
		"""
		m, n = prediction.shape

		if n == 1:
			for row in xrange(m):
				if prediction[row, 0] < 0.5:
					prediction[row, 0] = 0
				else:
					prediction[row, 0] = 1 
		else:
			for row in xrange(m):
				i = 0
				for col in xrange(n):
					if prediction[row, col] > prediction[row, i]:
						i = col
				prediction[row, :] = np.zeros(n)
				prediction[row, i] = 1

		return prediction


	def saveModel(self):
		"""
		Save the neural network model.
		"""
		#TODO: Implement.
		pass


	def loadModel(self, model):
		"""
		Load a model for this neural network.
		"""
		#TODO: Implement this later.
		pass


def testmlp(filename):
  	"""
  	Test mlp with mnist 2 and 3 digits.

  	Args:
  	-----
    	filename: Name of the file for 2 and 3.
  	"""
	data = np.load(filename)
	input_train = np.hstack((data['train2'], data['train3']))
	input_valid = np.hstack((data['valid2'], data['valid3']))
	input_test = np.hstack((data['test2'], data['test3']))
	target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
	target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
	target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))

	mlp = Mlp([PerceptronLayer(1, 10), PerceptronLayer(10, 256)])
	mlp.train(input_train.T, target_train.T, input_valid.T, target_valid.T, input_test.T, target_test.T, {'learn_rate': 0.02, 'epochs': 1600})


if __name__ == '__main__':

	testmlp('digits.npz')
