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
from mlp import *
from cnn import *
from util import *
from copy import deepcopy as cpy


class Qlearn():
	"""
	Q learning module.
	"""

	def direct_train(self, s, s_prime, a, r, gamma, term, hyperparameters):
		"""
		Train a variant of Q networks using the direct algorithm for
		Q learning.
		"""
		#Predict Q values
		qs = self.predict(s)
		qs_prime = 0

		if not term: #Predict qs_prime with prev weights.
			theta = cpy(self.layers)
			self.layers = self.layers_old
			qs_prime = self.predict(s_prime)
			self.layers = theta

		#Set prev weights to the current ones.
		self.layers_old = cpy(self.layers)

		#Change current weights according to update equation
		dE = np.zeros(qs.shape)
		dE[0, a] = qs[0, a] - r - gamma * np.max(qs_prime)
		self.backprop(dE)
		self.update(hyperparameters)


class Qnn(Mlp, Qlearn):
	"""
	A Q neural network.
	"""

	def __init__(self, layers):
		"""
		Initialize the Qnn.

		Args:
		-----
			layers: List of mlp layers arranged heirarchically.
		"""
		Mlp.__init__(self, layers)

		self.layers_old = []
		for layer in self.layers:
			self.layers_old.append(PerceptronLayer(layer.w.shape[0], layer.w.shape[1], layer.o_type))


	def train(self, s, s_prime, a, r, gamma, term, hyperparameters):
		"""
		Train the dqn.

		Args:
		----
			s 	: A no_imgs x img_length x img_width array repr a state.
			s_prime : A no_imgs x img_length x img_width array repr the next state.
			a : Action taken at state s.
			r : A double value repr reward gotten by entering next_state.
			gamma : A double value repr the discount factor.
			term : A boolean value: True if next_state is terminal, False otherwise.
			hyperparameters : A dictionary of training parameters.
		"""
		
		self.direct_train(s, s_prime, a, r, gamma, term, hyperparameters)


class Dqn(Cnn, Qlearn):
	"""
	A deep Q neural network.
	"""

	def __init__(self, layers):
		"""
		Initialize the Qnn.

		Args:
		-----
			layers: List of layers arranged heirarchically.
		"""

		self.layers_old = []

		Cnn.__init__(self, layers)

		for layer in layers['fully-connected']:
			self.layers_old.append(PerceptronLayer(layer.w.shape[0], layer.w.shape[1], layer.o_type))
		for layer in layers['convolutional']:
			self.layers_old.append(ConvLayer(layer.kernels.shape[0], layer.kernels.shape[1:], layer.pfctr, layer.no_in))

	
	def train(self, s, s_prime, a, r, gamma, term, hyperparameters):
		"""
		Train the dqn.

		Args:
		----
			s 	: A no_imgs x img_length x img_width array repr a state.
			s_prime : A no_imgs x img_length x img_width array repr the next state.
			a : Action taken at state s.
			r : A double value repr reward gotten by entering next_state.
			gamma : A double value repr the discount factor.
			term : A boolean value: True if next_state is terminal, False otherwise.
			hyperparameters : A dictionary of training parameters.
		"""

		self.direct_train(s, s_prime, a, r, gamma, term, hyperparameters)