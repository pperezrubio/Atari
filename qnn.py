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
from copy import deepcopy as cpy
from mlp import *
from util import *


class Qnn(Mlp):
	"""
	A Q neural network.
	"""

	def __init__(self, layers, func=np.max):
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

		self.func = func

	def train(self, s, s_prime, a, r, gamma, term, hyperparameters):
		"""
		Train the Qnn.

		Args:
		----
			s 	: A 1 x no_feats array repr a state.
			s_prime : A 1 x no_feats array repr a state.
			r : A double value repr reward gotten by entering next_state.
			gamma : A discount factor.
			term : A boolean value: True if next_state is terminal, False otherwise.
			hyperparameters : A dictionary of training parameters.
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
		dE[0, a] = qs[0, a] - r - gamma * self.func(qs_prime) #TODO: Pass term as binary.
		self.backprop(dE)
		self.update(hyperparameters)
