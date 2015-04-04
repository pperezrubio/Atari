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

class TicTacToe():
	"""
	Tic Tac Toe class.
	"""

	def __init__(self):
		"""
		Initialize the tic tac toe board.

		Args:
		-----
			board_size: Size of the board.
		"""
		self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]


	def __str__(self):
		"""
		String representation of the game.
		"""
		board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']


		for i in xrange(len(self.board)):
			if self.board[i] == 0.5:
				board[i] = 'O'
			elif self.board[i] == 1:
				board[i] = 'X'

		s = ' ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + '\n'
		s = s + '-----------\n'
		s = s + ' ' + board[3] + ' | ' + board[4] + ' | ' + board[5] + '\n'
		s = s + '-----------\n'
		s = s + ' ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + '\n'

		return s


	def play(self, tile, player):
		"""
		Make a move by the following player on the specified tile.
		"""
		if tile in range(len(self.board)) and self.board[tile] == 0 and player == 'X':
			self.board[tile] = 1
		elif tile in range(len(self.board)) and self.board[tile] == 0 and player == 'O':
			self.board[tile] = 0.5
		else:
			return -1

		return 0



	def getReward(self, player):
		"""
		Evaluate the current game state and provide a reward.
		"""

		p1, p2 = 0.5, 1
		if player == 'X':
			p1, p2 = 1, 0.5

		if self.board[6] == self.board[3] == self.board[0] == p1:
			return 1
		elif self.board[6] == self.board[3] == self.board[0] == p2:
			return -1
		if self.board[7] == self.board[4] == self.board[1] == p1:
			return 1
		elif self.board[7] == self.board[4] == self.board[1] == p2:
			return -1

		if self.board[8] == self.board[5] == self.board[2] == p1:
			return 1
		elif self.board[8] == self.board[5] == self.board[2] == p2:
			return -1

		#Evaluate diagonal positions
		if self.board[6] == self.board[4] == self.board[2] == p1:
			return 1
		elif self.board[6] == self.board[4] == self.board[2] == p2:
			return -1

		if self.board[8] == self.board[4] == self.board[0] == p1:
			return 1
		elif self.board[8] == self.board[4] == self.board[0] == p2:
			return -1

		if self.board[6] == self.board[7] == self.board[8] == p1:
			return 1
		elif self.board[6] == self.board[7] == self.board[8] == p2:
			return -1

		#Evaluate horizontal board positions
		if self.board[6] == self.board[7] == self.board[8] == p1:
			return 1
		elif self.board[6] == self.board[7] == self.board[8] == p2:
			return -1
		if self.board[3] == self.board[4] == self.board[5] == p1:
			return 1
		elif self.board[3] == self.board[4] == self.board[5] == p2:
			return -1

		if self.board[0] == self.board[1] == self.board[2] == p1:
			return 1
		elif self.board[0] == self.board[1] == self.board[2] == p2:
			return -1

		return 0


	def getState(self):
		return self.board


	def isTerminal(self):
		if self.getReward('X') == 1 or self.getReward('O') == 1:
			return True

		if 0 not in self.board:
			return True

		return False


	def reset(self):
		self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]



if __name__ == '__main__':

	game = TicTacToe()

	qnn = Qnn([PerceptronLayer(9, 30, "sum"), PerceptronLayer(30, 9)])
	REM = []
	gamma = 0.55
	epsilon = 0.4
	params = {'learn_rate': 0.02}
	episode = 1
	play = True

	while play:
		
		print "Episode:", episode
		print game

		while not game.isTerminal():
			#For now, assume human is always player 1
			m = -1
			while game.play(m, 'X') == -1:
				#m = int(raw_input('Enter a position to play: '))
				m = random.randint(0, 8)
			print "Player 1 plays..\n"
			print game

			if not game.isTerminal():
				s = game.getState()
				s = np.array(s).reshape(1, len(s)) #Get current state

				a = -1
				while game.play(a, 'O') == -1:
					#Use epsilon greedy strategy
					x = random.uniform(0, 1)
					if x <= epsilon:
						a = random.randint(0, 8)
					else:
						qval = qnn.predict(s)
						a = (np.where(qval == np.max(qval))[1])[0]
					
				#Store Experience
				s_prime = game.getState()
				s_prime = np.array(s_prime).reshape(1, len(s_prime))
				if game.isTerminal == True:
					term = True
				else:
					term = False
				REM.append((s, s_prime, a, game.getReward('O'), gamma, term))

				#Sample random experience
				e = REM[random.randint(0, len(REM) - 1)]
				qnn.train(e[0], e[1], e[3], e[4], e[5], params)

				print "Neural net plays...\n"
				print game

		if game.getReward('X') == 1:
			print "Player 1 wins.\n"
		elif game.getReward('O') == 1:
			print "Player 2 wins.\n"
		else:
			print "Draw.\n"

		game.reset()
		episode = episode + 1
