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
import theano.tensor.signal.downsample as downsample
import theano as thn
import theano.tensor as tn
import math
from copy import deepcopy
import matplotlib.pyplot as plt

import qnn
import util
from mlp import *

#TODO: Add reporting for q-value per frame.

ale_available_moves = {
    0  : 'noop',
    1  : 'fire',
    2  : 'up',
    3  : 'right',
    4  : 'left',
    5  : 'down',
    6  : 'up-right',
    7  : 'up-left',
    8  : 'down-right',
    9  : 'down-left',
    10 : 'up-fire',
    11 : 'right-fire',
    12 : 'left-fire',
    13 : 'down-fire',
    14 : 'up-right-fire',
    15 : 'up-left-fire',
    16 : 'down-right-fire',
    17 : 'down-left-fire'
}


class Gameclient():
    """
    Game client module.
    """

    def __init__(self, aleparams, gameparams, agentparams):
        """ 
        Initialize game client.
        """
        
        self.ale_params = aleparams
        self.game_params = gameparams
        self.agent_params = agentparams

        self.fin = open(self.ale_params['pipeout'])
        self.fout = open(self.ale_params['pipein'] ,'w')
        
        self.header = self.handshake()

        # Calculating inputs to qnn
        gameparams['state_features'] = (gameparams['crop_hei']/gameparams['factor'][0]) * (gameparams['crop_wid']/gameparams['factor'][1]) * agentparams['state_frames']

        #Construct agent
        if agentparams['hidden_layers'] == 0:
            layers = [PerceptronLayer(len(gameparams['moves']), gameparams['state_features'], agentparams['out'])]
        else:
            layers = [PerceptronLayer(len(gameparams['moves']), agentparams['hidden_units'][0], agentparams['out'])]
            for l in xrange(agentparams['hidden_layers'] - 1):
                layers.append(PerceptronLayer(agentparams['hidden_units'][l], agentparams['hidden_units'][l + 1]))
            layers.append( PerceptronLayer(agentparams['hidden_units'][-1], gameparams['state_features']))
 
        if agentparams['maximise']:
            self.minimax = np.max
        else:
            self.minimax = np.min

        self.qnn = qnn.Qnn(layers)

        self.ERM = {} #Experience Replay Memory

        self.evaluation_metric = {

            'epoch': [],

            'avg_qvals_per_epoch' : [],

            'total_reward_per_round':[],

            'avg_rewards_per_epoch' : []
        }

        plt.ion() #Interactive plots.


    def handshake(self):
        # initial handshake
        while 1:
            str_in = self.fin.readline().strip()

            # server greeting phrase
            reg = re.match( self.ale_params['greetregex'] ,str_in)
            if reg:
                self.fout.write( self.ale_params['engagemsg'] + '\n')
                self.fout.flush()
                break
                
        rtn = [int(i) for i in reg.groups()]        
                
        return rtn


    def train(self):
        """
        Train agent on the ALE environment.
        """

        # get random states
        print 'Getting set of states to hold out...'
        #rand_states = self.evaluate_agent(testcount=10, select_rand=0)
        self.reset_metrics()

        print 'Training agent on ALE ' + self.game_params['name'] + '...'
        for epoch in xrange(self.agent_params['no_epochs']):

            self.evaluation_metric['epoch'].append(epoch)

            # restart game
            str_in = self.fin.readline()
            self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) # send in reset signal
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]
            frame, reward, term = self.parse(response)
            state = [frame]
            phi_s = self.preprocess_state(state)

            # if first state is terminal already, next epoch
            if term == 1:
                self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                self.fout.flush()
                continue

            for i in xrange(self.game_params['maxframes'] - 1):

                print 'Epoch: ' + str(epoch) + ' , Move: ' + str(i) + '.'
                
                # send action to ale
                action = self.get_agent_action(phi_s, epoch)
                mapped_a = self.map_agent_moves(action)
                print 'Selected action: ' + ale_available_moves[mapped_a[0]] + '.'
                self.fout.write(self.ale_params['moveregex'] %  mapped_a[0]) 
                self.fout.flush()

                # get next frame
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                frame, reward, term = self.parse(response)
                
                # append observed frame to sequence of frames to make next state
                next_state = state[-self.agent_params['state_frames']+1:] + [frame]

                phi_sprime = self.preprocess_state(next_state)
                cont = True
                if term == 1: cont = False

                # store transition experience
                self.ERM[(tuple(phi_s.ravel()), action[0], tuple(phi_sprime.ravel()))] = (reward, cont)

                # perform experience replay on mini-batch
                self.experience_replay()

                phi_s, state = phi_sprime, next_state

                if term or i == self.game_params['maxframes']-1:
                    # Terminal state
                    self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                    self.fout.flush()
                    break

            # Further train the agent on its experiences.
            self.experience_replay(self.agent_params['replay_rounds'])

            # Evaluate agent's performance
            self.evaluation_metric['epoch'].append(epoch)
            avg_qval = self.evaluate_avg_qvals(rand_states)
            self.evaluate_agent(testcount = 1000)


    def preprocess_state(self, state): #TODO: Display to cross check.
        """
        Preprocess a sequence of frames that make up a state.

        Args:
        -----
            state: A sequence of frames.

        Returns:
        --------
            Preprocessed state    
        """
        N, m, n = self.agent_params['state_frames'], self.game_params['crop_hei'], self.game_params['crop_wid']
        factor = self.game_params['factor']
        maxed = np.zeros((N, m, n), dtype='float64')

        # max pool and downsample
        maxed[0] = state[0].reshape(m, n)
        for i in xrange(1, len(state)):
            maxed[i] = np.max(np.asarray(state[i - 1: i]), axis=0).reshape(m, n)

        x = tn.dtensor3('x')
        f = thn.function([x], downsample.max_pool_2d(x, factor))
        downsampled = f(maxed)

        if self.ale_params['display_state']:
            s = downsampled[-1].reshape(m / factor[0], n / factor[1])
            plt.figure(1)
            plt.clf()
            plt.imshow(s, 'gray')
            plt.pause(0.01)
        
        return downsampled.reshape(1, np.prod(downsampled.shape[0:])) #Stack


    def get_agent_action(self, states, episode, useEpsilon=None):
        """
        Select max actions for the given state based on an epsilon greedy strategy.

        Args:
        ------
            states: A no_states x no_feats. array of states.
            game_moves: List of moves for current game.
            min_epsilon: Minimum epsilon.
            episode: Current episode.
            no_episodes: Total number of episodes.

        Return:
        -------
            A no_states row vector of actions. 
        """
        
        epsilon = useEpsilon
        if epsilon is None:
            # Epsilon decay. Starts at 1.0.
            epsilon = max(math.exp(-5 * float(episode)/self.agent_params['no_epochs']), self.agent_params['min_epilson'])

        #Explore
        if random.uniform(0, 1) <= epsilon:
            print 'Executing random action.'
            return np.asarray([random.choice(range(len(self.game_params['moves']))) for no_states in xrange(states.shape[0])])

        #Else exploit
        print 'Executing policy based action.'
        qvals = self.qnn.predict(states)
        #print qvals #Sub with print matrix later on.
        if self.agent_params['maximise']:
            nn_moves = np.argmax(qvals, axis=1)
        else:
            nn_moves = np.argmin(qvals, axis=1) #TODO: reflected when training agent!!!

        return nn_moves


    def map_agent_moves(self, nn_moves):
        """
        Map agent moves to valid game moves.
        
        Args:
        -----
            nn_moves: Index of moves returned by agent.

        Return:
        -------
            Valid moves for this game.
        """

        return [self.game_params['moves'][ind] for ind in nn_moves]


    def experience_replay(self, rounds=1):
        """
        Train the agent on a mini-batch of pooled experiences.

        Args:
        -----
            rounds: Replay rounds.
        """
        batch_size = self.agent_params['batch_size']

        for i in xrange(rounds):
            print 'Performing round ' + str(i) + ' of experience replay on mini-batch size of ' + str(batch_size) + '.'
            states, nxt_states, actions, rewards, conts = [], [], [], [], []
            keys = [random.choice(self.ERM.keys()) for i in xrange(batch_size)]

            for key in keys:
                phi_s, action, phi_sprime = key
                reward, cont = self.ERM[key]

                states.append(phi_s)
                nxt_states.append(phi_sprime)
                actions.append(action)
                rewards.append(reward)
                conts.append(cont)

            self.qnn.train(np.asarray(states), np.asarray(nxt_states), np.asarray(actions), np.asarray(rewards), self.agent_params['gamma'], np.asarray(conts), self.agent_params)


    def clip_reward(self, r):
        """
        Normalise reward according to positive and negative limits.

        Args:
        -----
            r: Float repr. reward.

        Return:
        -------
            Clipped reward.
        """
        if r > 0:
            r = r / self.game_params['pos_rwd_max']
        elif r < 0:
            r = r / self.game_params['neg_rwd_max']

        return r


    def reset_metrics(self):

        self.evaluation_metric = {

            'epoch': [],

            'avg_qvals_per_epoch' : [],

            'total_reward_per_round':[],

            'avg_rewards_per_epoch' : []
        }


    def evaluate_avg_qvals(self, states, minmax=np.max):
        """
        Return the agent's average Q-values over a set of states.

        Args:
        -----
            states: A set of states.

        Returns:
        --------
            Average Q-values over states.
        """ 
        print 'Evaluating average Q value for held out states...'
        avg_qval = np.mean(minmax(self.qnn.predict(states), axis=1))
        self.evaluation_metric['avg_qvals_per_epoch'].append(avg_qval)
        plt.figure(2)
        plt.title('Average Q on ' + self.game_params["name"])
        plt.xlabel('Training Epochs')
        plt.ylabel('Average Action Value (Q)')
        plt.plot(self.evaluation_metric['epoch'], self.evaluation_metric['avg_qvals_per_epoch'], 'b')
        plt.draw()


    def evaluate_agent(self, testcount = 500, select_rand=0):
        """Evaluate performance of agent using optimal policy (with epsilon=0.05). Also use
           to pick random states for evaluation of q values to gauge network performance.

        Args:
        -----
            testcount: no. of episode to run
            select_rand: boolean. whether or not to collect random states
        """
        rounds = 0
        totalmoves = 0
        totalscore = 0
        score = 0
        self.evaluation_metric['total_reward_per_round'] = []

        if select_rand:
            Epsil = 0
            states = []
        else: Epsil = 0.05

        for i in range(testcount):

            j = 0
            score = 0

            # restart game
            str_in = self.fin.readline()
            self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
            self.fout.flush()

            frames = [np.zeros((self.game_params['crop_hei']*self.game_params['crop_wid']))  for i in range(self.agent_params['state_frames'])]

            while 1:
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]

                f,r,t = self.parse(response)
                score += r

                # Terminal state
                if t==1 or j == self.game_params['maxframes']:
                    if j > 0:
                        rounds += 1
                        totalscore += score
                        self.evaluation_metric['total_reward_per_round'].append(score)
                    
                    self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                    self.fout.flush()
                    break

                # shift frame window
                frames.append(f)
                if len(frames) > self.agent_params['state_frames']: frames.pop(0)

                phi_sprime = self.preprocess_state(frames)

                a = self.get_agent_action(phi_sprime, 0, useEpsilon=Epsil)

                if select_rand: states.append(phi_sprime)
             
                mapped_a = self.map_agent_moves(a)
                self.fout.write(self.ale_params['moveregex'] % mapped_a[0]) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        self.evaluation_metric['avg_rewards_per_epoch'].append(np.mean(self.evaluation_metric['total_reward_per_round']))
        plt.figure(3)
        plt.title('Average Reward on ' + self.game_params["name"])
        plt.xlabel('Training Epochs')
        plt.ylabel('Average Reward per Episode.')
        plt.plot(self.evaluation_metric['epoch'], self.evaluation_metric['avg_rewards_per_epoch'], 'b')
        plt.draw()

        if select_rand:
            states = states[:10000]
            print 'Selected random states:', str(len(states))
            return states
            #return [random.choice(states) for i in range(100)]
              


    def saveexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (self.game_params['game'], self.header)

        pickle.dump(self.ERM, open(filename.replace(' ',''), 'wb'))


    def savenn(self, filename= None):
        if filename == None:
            l = [len(self.game_params['moves'])] + self.agent_params['hidden_units'] + [self.game_params['state_features']]
            filename = 'store/qnn_%s_%s.p' % (self.game_params['game'],str(l).replace(' ','') )

        pickle.dump(self.qnn, open(filename, 'wb'))


    def loadexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (self.game_params['game'], self.header)

        self.ERM = pickle.load( open(filename.replace(' ',''),'rb') )


    def loadnn(self, filename=None ):
        if filename == None:
            l = [len(self.game_params['moves'])] + self.agent_params['hidden_units'] + [self.game_params['state_features']]
            filename = 'store/qnn_%s_%s.p' % (self.game_params['game'],str(l).replace(' ',''))

        self.qnn =  pickle.load( open(filename,'rb'))


