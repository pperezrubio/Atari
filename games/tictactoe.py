# Tic Tac Toe
import random

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


	def __repr__(self):
		"""
		String representation of the game.
		"""

		board = ['', '', '', '', '', '', '', '', '']

		for i in xrange(len(self.board)):
			if self.board[i] == 0.5:
				board[i] = 'O'
			elif self.board[i] == 1:
				board[i] = 'X'

      	print('   |   |')
      	print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])
      	print('   |   |')
      	print('-----------')
      	print('   |   |')
      	print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
      	print('   |   |')
      	print('-----------')
      	print('   |   |')
      	print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
      	print('   |   |')


    def play(tile, player):
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


    def getReward():
    	"""
    	Evaluate the current game state and provide a reward if necessary.
    	"""
    	pass

    def getState():
    	"""
    	Evaluate the current state of the board.
    	"""
    	pass

