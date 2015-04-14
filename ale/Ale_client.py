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


ale_param = {

        # pipe names
        'pipein': 'ale_fifo_in',
        'pipeout': 'ale_fifo_out',

        # Server greet regex
        # www-hhh
        'greetregex': "([0-9]{3})-([0-9]{3})",

        # Client engage message
        # r,s,k,e
        'engagemsg':= '1,0,0,1',

        # Client move regex
        'moveregex': '%d,0\n',

        # reset action for all
        'reset':45

            }

game_param = {

        # Game specific
        'game': 'alepong',

        # moves list
        'moves': [3,4],

        # ale specific
        # down sample factor
        'factor': 8,

        # index to crop
        'crop_start': 10880,
        'crop_end': 62080,
        
        'crop_wid': 160,
        'crop_hei': 160,

        # qnn input
        'state_features': 400,

        # Maximum number of moves in one episode
        'maxfames': 10000

             }


agent_param = {

        'stack_no': 4,
        'minbatch_no': 32,
    
        'use_rmse_prop': True,

        # Maximising q function
        'maximise' = False

        'learn_rate': 0.2,
        'epilson': 0.9,
        'gamma': 0.8,


        'hidden_layers': 2,
        'hidden_units':[600,600],
        'out':'sigmoid'
             
              }


##################################################


class ALEclient(qnn_client.gameclient):

    ### game specific functions ###

    def validmoves( self, s):
        '''Game specific. 
           Return available action for given state
           Also translate moves to start from 0
        '''

        m = range(len(self.gp['moves']))
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

        state = self.crop_grayscale(s)

        # down sample
        state = block_mean(state, self.gp['factor'])

        return tuple(state.flatten()[0])

    def crop_grayscale(self, s):

        # crop
        s = s[self.gp['crop_start']:self.gp['crop_end']]

        l = len(s)
        i = 0
        state = np.zeros((1,l/2))

        # grayscale: select only hue of pixel color
        while i < l:
            state[0,i/2] = int(s[i+1],base=16)/15.0
            i += 2

        return state.reshape(  (self.gp['crop_hei'],self.gp['crop_wid']) )
        

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


    # BEGIN
    q1 = ALEclient(enparm= ale_parm, gparm = game_param, nnparam=agent_param)

    #q1.loadnn()
    #q1.evaluate()

    print nnp

    time1 = time.time()
    q1.train(epoch=5000)
    time2 = time.time()

    print 'TIME:', time2-time1
