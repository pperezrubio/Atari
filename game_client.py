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

from mlp import *
import qnn


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

        self.fin = open( self.ale_params['pipeout'] )
        self.fout = open( self.ale_params['pipein'] ,'w')
        
        self.header = self.handshake()


        if agentparams['hidden_layers'] == 0:
            layers = [ PerceptronLayer(len(gameparams['moves']), gameparams['state_features'], agentparams['out']) ]
        else:
            layers = [PerceptronLayer(len(gameparams['moves']), agentparams['hidden_units'][0], agentparams['out'])]
            for l in xrange(agentparams['hidden_layers']-1):
                layers.append( PerceptronLayer(agentparams['hidden_units'][l], agentparams['hidden_units'][l+1]) )
            layers.append( PerceptronLayer(agentparams['hidden_units'][-1], gameparams['state_features']) )
 
        if agentparams['maximise']: self.minimax = np.max
        else: self.minimax = np.min

        self.qnn = qnn.Qnn( layers )
        self.ERM = {}


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
        Trains the agent on the ALE environment.
        """

        rand_states = []

        for epoch in xrange(self.agent_params['no_epochs']):

            #Evaluate agent performance and get metrics.
            if ((epoch + 1) % 100) == 0:
                print 'Epoch: ', epoch + 1
                print 'No experiences: ', len(self.ERM)
                self.evaluate_agent()
                self.evaluate_avgqv(rand_states)
                self.saveexp()
                self.savenn()

            #Build random states.
            if epoch > 0 and len(rand_states) < 500:
                for i in xrange(5):
                    key = [random.choice(self.ERM.keys())]
                    if rand_states == []:
                        rand_states = self.ERM[key][2]
                    else:
                        rand_states = np.vstack((rand_states, self.ERM[key][2]))

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]

            f, r, term = self.parse(response)
            s = self.create_state([f])

            # if first state is terminal already, next epoch
            if term == 1:
                self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                self.fout.flush()
                continue

            # action for first frame
            a = self.get_agent_action(s, 0, useEp=True)

            k_frames = [f]
            recent_reward = r

            for i in xrange(self.game_params['maxframes'] - 1):
                
                # send in action
                mapped_a = self.map_agent_moves(a)
                self.fout.write(self.ale_params['moveregex'] %  mapped_a[0]) 
                self.fout.flush()

                # get next frame
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                f, r, term = self.parse(response)
                
                # Append frame to k_frame
                k_frames.append(f)
                if r != 0: recent_reward = r 

                # Calculate action if frame is first in stack
                if len(k_frames) == 1:
                    f_ = self.create_state(k_frames)
                    a = self.get_agent_action(f_, epoch, useEp=True)
            
                # if reached state_frames, stack frames into state and put into ERM
                if (i+1) % self.agent_params['state_frames'] == 0 or term: 

                    s_ = self.create_state(k_frames)

                    # add (state,action) pair to experience
                    self.ERM[( tuple(s.flat), tuple(a), tuple(s_.flat))] = (recent_reward, term, s, s_)

                    # train one                
                    self.replay()

                    s = s_

                    # action
                    a = self.get_agent_action(s, epoch, useEp=True)

                    k_frames = []
                    recent_reward = 0


                if term or i == self.game_params['maxframes']-1:
                    # Terminal state
                    self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                    self.fout.flush()
                    break
            
            # train nn on exp
            for i in range( min(len(self.ERM), self.agent_params['replay_rounds']) ):
                self.replay()

        # Final evaluation
        self.evaluate(testcount = 1000)


    def get_agent_action(self, states, episode, useEp):
        """
        Select max actions for the given state based on an epsilon greedy strategy.

        Input:
            states: A no_states x no_feats. array of states.
            game_moves: List of moves for current game.
            min_epsilon: Minimum epsilon.
            episode: Current episode.
            no_episodes: Total number of episodes.

        Return:
            A no_states row vector of actions. 
        """

        # Epsilon decay. Starts at 0.9.
        epsilon = max(math.exp(-2 * float(episode)/self.agent_params['no_epochs'])-0.1, self.agent_params['min_epilson'])

        #Explore
        if useEp and random.uniform(0, 1) <= epsilon:
            return np.asarray([random.choice(range(len(self.game_params['moves']))) for no_states in xrange(states.shape[0])])

        #Exploit
        qvals = self.qnn.predict(states)
        if self.agent_params['maximise']:
            nn_moves = np.argmax(qvals, axis=1)
        else:
            nn_moves = np.argmin(qvals, axis=1)

        return nn_moves


    def map_agent_moves(self, nn_moves):
        """ Map agent moves to valid game moves
        
        Args:
        -----
            nn_moves: index of moves returned by agent
        """

        return [self.game_params['moves'][ind] for ind in nn_moves]


    def create_state(self, frames):
        """
        Create a game state from a set of frames.

        Args:
        -----
            frames: 1d game frames to stack.
            height: 

        Returns:
        --------
            A state.
        """

        stacks = np.zeros((len(frames), self.game_params['crop_hei'], self.game_params['crop_wid']), dtype='float64')
        for i in range(len(frames)):
            stacks[i] = frames[i].reshape(self.game_params['crop_hei'], self.game_params['crop_wid'])

        #Get max and downsample
        x = tn.dmatrix('x')
        f = thn.function([x], downsample.max_pool_2d(x, self.game_params['factor'] ))
        state = f(np.max(stacks,axis=0))
        
        return state.reshape(1, np.prod(state.shape[0:]))


    def replay(self):
        ''' Train qnn based on experience'''

        ind = random.choice(self.ERM.keys())
        r_, t_, sf, s_f = self.ERM[ind]
        s,a,s_ = ind

        pass

        #self.qnn.train(sf, s_f, a, r_, self.gamma, t_, self.agent_params, self.minimax)


    def evaluate_agent(self, testcount = 500):
        rounds = 0
        totalmoves = 0
        totalscore = 0
        score = 0

        for i in range(testcount):

            j = 0
            score = 0

            # restart game
            str_in = self.fin.readline()
            self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
            self.fout.flush()

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
                    
                    self.fout.write(self.ale_params['moveregex'] % self.ale_params['reset']) 
                    self.fout.flush()
                    break

                k_frames.append(f)

                if j % self.agent_params['state_frames'] == 0:
                    s = self.re(k_frames)
                    a = self.get_agent_action(s, useEp=0)
                    k_frames = []

                mapped_a = self.map_agentmoves(a)
                self.fout.write(self.ale_params['moveregex'] % mapped_a[0]) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds
              


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


    def evaluate_avgqv(self, states):
        tar = self.qnn.predict(states)
        avgqvs =  np.mean(self.func(tar,1))

        print 'AVG OPTIMAL Q-VALUE:', avgqvs
 
    

    def clip_reward(self, r):
        '''Normalise reward according to positive and negative limits.

        Args:
        ----
            r: reward
 
        '''

        if r > 0:
            r = r / self.game_params['pos_rwd_max']
        elif r < 0:
            r = r / self.game_params['neg_rwd_max']

        return r



    ### game specific functions ###


    def parse(self, response):
        ''' Game specific.

        Input:
            response: list of elements ended by ':'
        '''
        s, r, t = response
        s = self.parse_state(s)
        
        return s,float(r),int(t)


    def parse_state(self, s):
        '''Game specific '''
        w, h = self.header

        state = s

        return state


    def reconstruct(self, s):
        ''' Game specific
        
        Input:
            1D tuple

        Output:
            2D matrix (1 x no.features)
        '''
        return np.array([s])

