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

import random
import numpy as np
from mlp import *
from qnn import *

class Grid():
	"""
	Grid world class.
	"""

	def __init__(self):
		"""
		Initialize the grid world class.
		"""
		self.grid = [' ', ' ', ' ', ' ', ' ', 'T', ' ', ' ', 'T', 'T', ' ', ' ', ' ', ' ', ' ', ' ']
		self.reward = [0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, -0.5, 0, -0.5]
		self.dict_of_moves = { #Up, Down, Left, Right
			0:[4, 0, 1, 0],
			1:[5, 1, 2, 0],
			2:[6, 2, 3, 1],
			3:[7, 3, 3, 2],
			4:[8, 0, 5, 4],
			5:[5, 5, 5, 5],
			6:[10, 2, 7, 5],
			7:[11, 3, 7, 6],
			8:[8, 8, 8, 8],
			9:[9, 9, 9, 9],
			10:[14, 6, 11, 9],
			11:[15, 7, 11, 10],
			12:[12, 8, 13, 12],
			13:[13, 9, 14, 12],
			14:[14, 10, 15, 13],
			15:[15, 11, 15, 14]
		}
		self.agent = 0
		self.grid[self.agent] = 'X'


	def __str__(self):
		"""
		String representation of the grid world class.
		"""
		s = '----------------\n'
		s = s + ' ' + str(self.grid[15]) + ' | ' + str(self.grid[14]) + ' | ' + str(self.grid[13]) + ' | ' + str(self.grid[12]) + '\n'
		s = s + '----------------\n'
		s = s + ' ' + str(self.grid[11]) + ' | ' + str(self.grid[10]) + ' | ' + str(self.grid[9]) + ' | ' + str(self.grid[8]) + '\n'
		s = s + '----------------\n'
		s = s + ' ' + str(self.grid[7]) + ' | ' + str(self.grid[6]) + ' | ' + str(self.grid[5]) + ' | ' + str(self.grid[4]) + '\n'
		s = s + '----------------\n'
		s = s + ' ' + str(self.grid[3]) + ' | ' + str(self.grid[2]) + ' | ' + str(self.grid[1]) + ' | ' + str(self.grid[0]) + '\n'
		s = s + '----------------\n'

		return s


	def play(self, move):
		"""
		Make a move by the following player on the specified tile.
		"""
		if move == -1:
			return -1

		if self.agent == self.dict_of_moves[self.agent][move]:
			return -1

		self.grid[self.agent] = ' '
		self.agent = self.dict_of_moves[self.agent][move]
		self.grid[self.agent] = 'X'

		return 0


	def getReward(self, old_pos, new_pos):
		"""
		Evaluate the current game state and provide a reward.
		"""
		#if old_pos == new_pos:
		#	return 0

		return self.reward[new_pos]


	def getState(self):
		state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		state[self.agent] = 1
		return state, self.agent


	def isTerminal(self):
		if self.agent in [5, 9, 8]:
			return True

		return False


	def reset(self):
		self.grid = [' ', ' ', ' ', ' ', ' ', 'T', ' ', ' ', 'T', 'T', ' ', ' ', ' ', ' ', ' ', ' ']
		self.agent = 0
		self.grid[self.agent] = 'X'


if __name__ == '__main__':

	game = Grid()

	qnn = Qnn([PerceptronLayer(4, 16, "sum")])
	REM = {}
	gamma = 0.55
	epsilon = 0.9
	params = {'learn_rate': 0.2}
	episode = 1
	no_episodes = 1000
	play = True
	nn_win_count = 0

	while episode < no_episodes:
		
		print "Episode:", episode
		print game

		while not game.isTerminal():
			s, i = game.getState()
			s = np.array(s).reshape(1, len(s)) #Get current state

			a = -1
			while game.play(a) == -1:
				#Use epsilon greedy strategy
				x = random.uniform(0, 1)
				temp = episode * (epsilon - 0.1)/no_episodes

				if x <= (epsilon - temp):
					a = random.randint(0, 3)
				else:
					qval = qnn.predict(s)
					a = (np.where(qval == np.max(qval))[1])[0]
					
			#Store Experience
			s_prime, i_prime = game.getState()
			s_prime = np.array(s_prime).reshape(1, len(s_prime))
			REM[tuple(s.flat), a] = (s, s_prime, a, game.getReward(i, i_prime), gamma, game.isTerminal())

			#Sample random experience
			e = REM.values()[random.randint(0, len(REM.values()) - 1)]
			qnn.train(e[0], e[1], e[2], e[3], e[4], e[5], params)

			if a == 0:
				move = 'up'
			elif a == 1:
				move = 'down'
			elif a == 2:
				move = 'left'
			else:
				move = 'right'

			print "Neural net moves " + move + "..."
			print game

		game.reset()
		episode = episode + 1
