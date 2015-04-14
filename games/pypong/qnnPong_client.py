import re
import numpy as np
import random
import time
import sys, os
import itertools
import cPickle as pickle

sys.path.append('../')

from mlp import *
import qnn
import qnn_client

# Settings for different gaming environments ####

class globalparam():

    def __init__(self):

        self.GAME='pygamepong'

        self.GAMEIN = 'pongfifo_in'
        self.GAMEOUT = 'pongfifo_out'

        self.MOVES = [0,1]

        # reset action for all
        self.RESET = 45

        # Server greet regex
        self.GREETREGEX = '([0-9]+)x([0-9]+)'

        # Client engage message
        self.ENGAGEMSG = 'ENGAGE'

        # Client move regex
        self.MOVEREGEX = '%d\n'

        # Maximum number of moves in one episode
        self.MAXMOVES = 10000

        # qnn input
        self.STATE_FEATURES = 120
        #STATE_FEATURES = 14

        # Maximising q function
        self.MAXIMISE = False


##################################################


class Pongclient(qnn_client.gameclient): 

    ### game specific functions ###

    def validmoves( self, s):
        '''Game specific. return available action for given state'''
        return self.gp.MOVES


    def parse(self, response):
        ''' Game specific.

        Input:
            response: list of elements ended by ':'
        '''
        s, r, t = response
        
        s = self.parse_state(s)
        r = float(r)
        t = int(t)

        # reward reassignment
        if r == 1: r = 0
        elif r == -1: r = 1
        
        return s, r, t


    def parse_state(self, s):
        '''Game specific '''
        w, h = self.header
        state = s.strip('()').split(',')
        
        # normalization
        #state = [float(c)/([w,h][i % 2]) for i,c in enumerate(state)]
        # without normalization  (for reconstruction)
        state = [float(c) for c in state]        

        return tuple(state)



    def reconstruct(self, s):
        ''' Game specific
        
        Input:
            1D tuple

        Output:
            2D matrix (1 x no.features)
        '''

        #return np.array([s])

        tsx, tsy, ssx, ssy, scx, scy, esx, esy, ecx, ecy, bsx, bsy, bcx, bcy = s

        frame = np.zeros((tsy,tsx))
        
        # self paddle
        for i in xrange(int(ssy)):
            for j in xrange(int(ssx)):
                frame[i+scy, j+scx] = 1

        # enemy paddle
        for i in xrange(int(esy)):
            for j in xrange(int(esx)):
                frame[i+ecy, j+ecx] = 1

        # ball
        for i in xrange(int(bsy)):
            for j in xrange(int(bsx)):
                x = j+bcx
                y = i+bcy
                if x >= 0 and x < tsx and y>=0 and y< tsy:
                    frame[y,x] = 1
      
        return frame.reshape((1,tsx*tsy))


if __name__ == '__main__':

    gp = globalparam()

    nnp = {
             'learn_rate':0.2,
             'layers': 2,
             'hid':[300,300],
             'out':'sigmoid'
          }

    # BEGIN
    q1 = Pongclient(gparm = gp,  nnparam=nnp)

    #q1.loadnn()
    #q1.evaluate()

    print nnp

    time1 = time.time()
    q1.train(epoch=5000)
    time2 = time.time()

    print 'TIME:', time2-time1
