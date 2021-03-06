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

import re
import numpy as np
import random
import time
import sys, os
import itertools
import cPickle as pickle
import matplotlib

sys.path.append('../')

import qnn
import util
from mlp import *
from game_client import Gameclient


ale_param = {

    # pipe names
    'pipein': 'ale_fifo_in',
    'pipeout': 'ale_fifo_out',

    # Server greet regex
    # www-hhh
    'greetregex': "([0-9]{3})-([0-9]{3})",

    # Client engage message
    # r,s,k,e
    'engagemsg': '1,0,0,1',

    # Client move regex
    'moveregex': '%d,18\n',

    # reset action for all
    'reset':45,

    'display_state': True
}

game_param = {

    'pong': {

        # Game specific
        'name': 'Ale Pong',

        # moves list
        'moves': [3, 4],

        # down sample factor
        'factor': (8,8),

        # index to crop
        'crop_start': 10880,
        'crop_end': 62080,
        
        'crop_wid': 160,
        'crop_hei': 160,

        # reward
        'pos_rwd_max': 1,
        'neg_rwd_max': 1,

        # Maximum number of moves in one episode
        'maxframes': 10000
    },

    'spaceinvaders': {

        # Game specific
        'name': 'Ale Space Invaders',

        # moves list
        'moves': [0,1,3,4,11,12],

        # down sample factor
        'factor': (8,8),

        # index to crop
        'crop_start': 8640, #2240
        'crop_end': 59840,
        
        'crop_wid': 160,
        'crop_hei': 160,

        # reward
        'pos_rwd_max': 200,
        'neg_rwd_max': 1,

        # Maximum number of moves in one episode
        'maxframes': 10000
    },

    'breakout': {

        # Game specific
        'name': 'Ale Breakout',

        # moves list
        'moves': [0,1,3,4],

        # down sample factor
        'factor': (8,8),

        # index to crop
        'crop_start': 9600, #15040,
        'crop_end': 60800, #66240,
    
        'crop_wid': 160,
        'crop_hei': 160,

        # reward
        'pos_rwd_max': 20,
        'neg_rwd_max': 1,

        # Maximum number of moves in one episode
        'maxframes': 10000
    },

    'seaquest': {

        # Game specific
        'name': 'Ale Seaquest',

        # moves list
        'moves': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],

        # down sample factor
        'factor': (8,8),

        # index to crop
        'crop_start': 8320,
        'crop_end': 59520,
    
        'crop_wid': 160,
        'crop_hei': 160,

        # reward
        'pos_rwd_max': 20,
        'neg_rwd_max': 1,

        # Maximum number of moves in one episode
        'maxframes': 10000
    }
}

agent_param = {

    'state_frames': 2,
    'no_epochs': 1000,
    'batch_size': 32,

    'use_RMSprop': True,
    'maximise': False, # Maximising q function

    'replay_rounds': 10,
    'learn_rate': 0.2,
    'min_epilson': 0.1, #Epsilon decay starts at 1.0.
    'gamma': 0.8,

    'hidden_layers': 2,
    'hidden_units':[1000, 1000],
    'out':'sigmoid'

}



class ALEclient(Gameclient):
    """
    Ale client class.
    """

    def parse(self, response):
        """
        Parse an ALE response to a set of 1d frame, reward 
        and a terminal signal.

        Input:
        ------
            response: list of elements ended by ':'

        Returns:
        --------
            A greyscaled vector repr. frame, reward, terminal signal.
        """
        s, e = response
        t, r = e.split(',')
        
        return self.crop_grayscale(s), self.clip_reward(float(r)), int(t)


    def crop_grayscale(self, s):
        """
        Crop and grey scale a 1d frame from its string
        representation s.

        Args:
        -----
            s: String representation of a frame.

        Returns:
        --------
            Cropped and greyscaled vector repr. frame.
        """

        s = s[self.game_params['crop_start']:self.game_params['crop_end']]

        l = len(s)
        i = 0
        frame = np.zeros((l/2,))

        # grayscale: select only hue of pixel color
        while i < l:
            frame[i/2] = int(s[i+1],base=16)/15.0
            i += 2

        return frame


if __name__ == '__main__':

    game = sys.argv[1].rstrip('.bin')
    ale = ALEclient(aleparams=ale_param, gameparams=game_param[game], agentparams=agent_param)

    time1 = time.time()
    ale.train()
    time2 = time.time()

    print 'Time to train:', time2 - time1
