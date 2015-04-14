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
from util import *
from copy import deepcopy


class Qlearn():
	"""
	Q learning module.
	"""

	def direct_train(self, s, s_prime, a, r, gamma, cont, hyperparams, minmax=np.max):
		"""
		Train a variant of Q networks using the direct algorithm for
		Q learning.

		Args:
		----
			s 	: A no_states x no_feats array repr different states s.
			s_prime : The no_states x no_feats array repr the next state from states s.
			a 	: A no_states vector of actions taken at states s.
			r 	: A no_states vector of rewards gotten by taking an action at states s.
			gamma : The discount factor for future rewards.
			cont : A no_states boolean vector indicating if the next states from states s is not terminal.
			minmax : Use max or min Q values for a next state.
			hyperparams : A dictionary of training parameters.
		"""
		
		qs = self.predict(s)
		qs_prime = np.zeros(qs.shape)

		#Predict qs_prime with prev weights.
		theta = deepcopy(self.layers)
		self.layers = self.layers_old
		qs_prime = self.predict(s_prime)
		self.layers = theta

		#Set prev weights to the current ones.
		self.layers_old = deepcopy(self.layers)

		#Change current weights according to update equation
		dE = np.zeros(qs.shape)
		dE[np.arange(dE.shape[0]), a] = qs[np.arange(qs.shape[0]), a] - r - (gamma * cont * minmax(qs_prime, axis=1))
		self.backprop(dE)
		self.update(hyperparams)


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


	def train(self, s, s_prime, a, r, gamma, cont, hyperparameters, minmax=np.max):
		"""
		Train the qnn using stochastic gradient.

		Args:
		----
			s 	: A no_states x no_feats array repr different states s.
			s_prime : The no_states x no_feats array repr the next state from states s.
			a 	: A no_states vector of actions taken at states s.
			r 	: A no_states vector of rewards gotten by taking an action at states s.
			gamma : The discount factor for future rewards.
			cont : A no_states boolean vector indicating if the next states from states s is not terminal.
			minmax : Use max or min Q values for a next state.
			hyperparams : A dictionary of training parameters.
		"""
		
		self.direct_train(s, s_prime, a, r, gamma, cont, hyperparameters, minmax)
