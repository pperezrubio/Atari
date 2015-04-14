import re
import numpy as np
from scipy import ndimage
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

        self.GAME='alepong'

        # pipe names
        self.GAMEIN = 'ale_fifo_in'
        self.GAMEOUT = 'ale_fifo_out'

        # Server greet regex
        # www-hhh
        self.GREETREGEX = "([0-9]{3})-([0-9]{3})"

        # Client engage message
        # r,s,k,e
        self.ENGAGEMSG = '1,0,0,1'

        # Client move regex
        self.MOVEREGEX = '%d,0\n'


        # moves list
        self.MOVES = [3,4]

        # reset action for all
        self.RESET = 45


        # ale specific
        # screen factorise
        self.FACT = 8
        # screen width
        self.WID = 160 / self.FACT
        # index to crop
        self.CROP_S = 10880
        self.CROP_E = 62080
        

        # qnn input
        self.STATE_FEATURES = self.WID**2
        # Maximising q function
        self.MAXIMISE = False

        # Maximum number of moves in one episode
        self.MAXMOVES = 10000


##################################################


class ALEclient(qnn_client.gameclient):

    ### game specific functions ###

    def validmoves( self, s):
        '''Game specific. 
           Return available action for given state
           Also translate moves to start from 0
        '''

        m = range(len(self.gp.MOVES))
        return m


    def parse(self, response):
        ''' Game specific.

        Input:
            response: list of elements ended by ':'
        '''
        s, e = response
        t, r = e.split(',')

        s = self.parse_state(s)
        r = float(r)
        t = int(t)
 
        if r == 1:
            r = 0
        elif r == -1:
            r = 1
        
        return s, r, t


    def parse_state(self, s):
        '''Game specific '''
        w, h = self.header

        # crop
        s = s[self.gp.CROP_S:self.gp.CROP_E]

        l = len(s)
        i = 0
        state = np.zeros((1,l/2))

        # grayscale: select only hue of pixel color
        while i < l:
            state[0,i/2] = int(s[i+1],base=16)/15.0
            i += 2

        # down sample
        state = block_mean(state.reshape((160,160)), self.gp.FACT)

        return tuple(state.reshape((1, self.gp.STATE_FEATURES))[0])


    def reconstruct(self, s):
        ''' Game specific
        
        Input:
            1D tuple

        Output:
            2D matrix (1 x no.features)
        '''
        #for i in range(20):
        #    for j in range(20):
        #        print '{:.3f}'.format(s[i*20+j]),
        #    print

        #raw_input()

        return np.array([s])

        #return frame


def block_mean(ar, fact):
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


if __name__ == '__main__':

    gp = globalparam()

    nnp = {
             'learn_rate':0.2,
             'layers': 2,
             'hid':[600,600],
             'out':'sigmoid'
            }

    # BEGIN
    q1 = ALEclient(gparm = gp, nnparam=nnp)

    #q1.loadnn()
    #q1.evaluate()

    print nnp

    time1 = time.time()
    q1.train(epoch=5000)
    time2 = time.time()

    print 'TIME:', time2-time1
