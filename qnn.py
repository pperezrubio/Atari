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


class Qnn(Mlp):
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
		if layers[0].o_type == "sum":

			Mlp.__init__(self)

			self._theta = []
			for layer in self.layers:
				self._theta.append({'w': 0.01 * np.random.randn(layer.w.shape[0], layer.w.shape[1]),
									'b': 0.01 * np.random.randn(layer.b.shape[0], layer.b.shape[1])})
		else:
			raise Exception


	def train(self, state, next_state, reward, discount, terminate, hyperparameters):
		"""
		Train the Qnn on the given states using the provided hyperparameters.

		Args:
		----
			states 	: A no_states x no_frames x no_feats array of states.
			rewards : A no_states x 1 array of rewards.
			hyperparameters : A dictionary of training parameters.
		"""
		#Predict Q values
		qs = self.predict(state)

		#Store current network weights
		theta = []
		for i in xrange(len(self.layers)):
			theta.append({'w': self.layers[i].w, 'b': self.layers[i].b})

		if not terminate:
			for i in xrange(len(self.layers)):
				self.layers[i].w, self.layers[i].b = self._theta[i]['w'], self._theta[i]['b']

			qs_prime = self.predict(next_state)

			for i in xrange(len(self.layers)):
				self.layers[i].w, self.layers[i].b = theta[i]['w'], theta[i]['b']
		else:
			qs_prime = 0

		#Set prev weights to the current ones.
		self._theta = theta

		#Change current weights according to update equation
		target = reward + (discount * np.max(qs_prime))
		self.backprop(target - qs)
		self.update(hyperparameters)

