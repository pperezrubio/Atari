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

sys.path.append('../')

from mlp import *
import qnn
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
    'maxframes': 10000
}


agent_param = {

    'state_frames': 4,
    'no_epochs': 1000,
    'minbatch_no': 32,

    'use_rmse_prop': True,

    # Maximising q function
    'maximise' = False

    'learn_rate': 0.2,
    'min_epilson': 0.1, #Epsilon decay starts at 0.9.
    'gamma': 0.8,

    'hidden_layers': 2,
    'hidden_units':[600, 600],
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
            A greyscaled 1d frame, reward, terminal signal.
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
            Cropped and greyscaled 1d frame.
        """
        s = s[self.gp['crop_start']:self.gp['crop_end']]

        l = len(s)
        i = 0
        state = np.zeros((1,l/2))

        # grayscale: select only hue of pixel color
        while i < l:
            state[0,i/2] = int(s[i+1],base=16)/15.0
            i += 2

        return state


if __name__ == '__main__':

    ale = ALEclient(aleparams=ale_param, gameparams=game_param, agentparams=agent_param)

    time1 = time.time()
    ale.train()
    time2 = time.time()

    print 'Time to train:', time2 - time1
