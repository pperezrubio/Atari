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
		board = ['', '', '', '', '', '', '', '', '']


		for i in xrange(len(self.board)):
			if self.board[i] == 0.5:
				board[i] = 'O'
			elif self.board[i] == 1:
				board[i] = 'X'

		s = ' ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + '\n'
		s = s + '--------\n'
		s = s + ' ' + board[3] + ' | ' + board[4] + ' | ' + board[5] + '\n'
		s = s + '--------\n'
		s = s + ' ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + '\n'

		return s


	def play(self, player):
		"""
		Make a move by the following player on the specified tile.
		"""
		if self.board[tile] != 0 and player == 'X':
			self.board[tile] = 1
		elif self.board[tile] != 0 and player == 'O':
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

		if 0 in self.board:
			return True

		return False


	def getHumanMove(self):
		pass



if __name__ == '__main__':

	game = TicTacToe()
	#qnn = Qnn([PerceptronLayer(9, 300, "sum"), PerceptronLayer(300, 9)])

	while True:
		episode = 1
		print "Episode:", episode
		print game

		while not game.isTerminal():
			#For now, assume human is always player 1
			print "Player 1 to play."
			game.play(game.getHumanMove(), 1)
			print game
			if not game.isTerminal():
				print "Player 2 to play."
				game.play(game.getHumanMove(), 0)
				print game

		if game.getReward('X') == 1:
			print "Player 1 wins."
		elif game.getReward('O') == 1:
			print "Player 2 wins."
		else:
			print "Draw."

		episode = episode + 1
