import re
import numpy as np
import random
import time
import sys, os
import itertools
import cPickle as pickle
import theano.tensor.signal.downsample as downsample

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

        self.fin = open( self.gp.GAMEOUT )
        self.fout = open( self.gp.GAMEIN ,'w')
        
        self.header = self.handshake()

        # lambda
        self.gamma = 0.8
        # max exploit probability
        self.exploit = 0.7
        self.param = nnparam


        if nnparam['layers'] == 0:
            layers = [ PerceptronLayer(len(gparm.MOVES), gparm.STATE_FEATURES, nnparam['out']) ]
        else:
            layers = [PerceptronLayer(len(gparm.MOVES), nnparam['hid'][0], nnparam['out'])]
            for l in xrange(nnparam['layers']-1):
                layers.append( PerceptronLayer(nnparam['hid'][l], nnparam['hid'][l+1]) )
            layers.append( PerceptronLayer(nnparam['hid'][-1], gparm.STATE_FEATURES) )
 
        if gparm.MAXIMISE: self.func = np.max
        else: self.func = np.min

        self.qnn = qnn.Qnn( layers )
        self.ERM = {}


    def handshake(self):
        # initial handshake

        while 1:
            str_in = self.fin.readline().strip()

            # server greeting phrase
            reg = re.match( self.gp.GREETREGEX ,str_in)
            if reg:
                self.fout.write( self.gp.ENGAGEMSG + '\n')
                self.fout.flush()
                break
                
        rtn = [int(i) for i in reg.groups()]        
                
        return rtn


    def train(self):
        """
        Trains the agent on the ALE environment.
        """

        rand_states = None

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
                    if rand_states is None:
                        rand_states = self.ERM[key][2]
                    else:
                        rand_states = np.vstack((rand_states, self.ERM[key][2]))

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]

            f, r, t = self.parse(response)
            k_frames = [self.reconstruct(f)]
            k = 1

            # if first state is terminal already, next epoch
            if term == 1:
                self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
                self.fout.flush()
                continue

            for i in xrange(self.game_params['max_frames'] - 1):
                
                if k < self.agent_params['state_frames']:


                # action
                a = self.get_agent_action(sf, progress=float(ep)/epoch ,opt=0)

                # send in action
                self.fout.write(self.gp.MOVEREGEX % (self.gp.MOVES[a],)) 
                self.fout.flush()

                # get next state
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                s_, r_, t_ = self.parse(response)

                # add (state,action) pair to experience
                sf = self.reconstruct(s)
                s_f = self.reconstruct(s_)
                self.exp[( s, a, s_ )] = (r_, t_, sf, s_f)

                # train one                
                self.replay()

                if t_ == 1 or j == self.gp.MAXMOVES-1:
                    # Terminal state
                    self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
                    self.fout.flush()
                    break
                else:
                    # current state is next state
                    s, r = s_, r_
                    sf = s_f


            
            # train nn on exp
            for i in range( min(len(self.exp),replay_rate) ):
                self.replay()

        # Final evaluation
        self.evaluate(testcount = 1000)


    def get_agent_action(self, states, game_moves, min_epsilon, episode, no_episodes, useEp):
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
        epsilon = max(math.exp(-2 * float(episode)/no_episodes), min_epsilon)

        #Explore
        if useEp and random.uniform(0, 1) <= epsilon:
            return np.asarray([random.choice(game_moves) for no_states in xrange(states.shape[0])])

        #Exploit
        qvals = self.qnn.predict(states)
        nn_moves = np.argmax(qvals, axis=1)
        return np.asarray([game_moves[ind] for ind in nn_moves])


    def create_state(self, frames, height, width, down_factor):
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

        stacks = np.zeros((len(frames), height, width), dtype='float64')
        for frame in frames:
            stacks[i] = frames[i].reshape(height, width)

        #Get max and downsample
        x = tn.dmatrix('x')
        f = thn.function([x], downsample.max_pool_2d(x, factor))
        state = f(np.max(stacks))

        return state.reshape(1, np.prod(state.shape[0:]))


    def replay(self):
        ''' Train qnn based on experience'''

        ind = random.choice(self.exp.keys())
        r_, t_, sf, s_f = self.exp[ind]
        s,a,s_ = ind

        #self.qv_set((sf, s_f, a, r_, t_))
        s, s_prime, a, r, term = sa
        self.qnn.train(s, s_prime, a, r, self.gamma, term, self.param, self.func)


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
            self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
            self.fout.flush()

            while 1:
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]

                s,r,t = self.parse(response)
                score += r

                # Terminal state
                if t==1 or j == self.gp.MAXMOVES:
                    if j > 0:
                        rounds += 1
                        totalscore += score
                    
                    self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
                    self.fout.flush()
                    break

                sf = self.reconstruct(s)
                a = self.get_agent_action(sf,opt=1)
                self.fout.write(self.gp.MOVEREGEX % (self.gp.MOVES[a],)) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds
              


    def saveexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (self.gp.GAME, self.header)

        pickle.dump(self.exp, open(filename.replace(' ',''), 'wb'))


    def savenn(self, filename= None):
        if filename == None:
            l = [len(self.gp.MOVES)] + self.param['hid'] + [self.gp.STATE_FEATURES]
            filename = 'store/qnn_%s_%s.p' % (self.gp.GAME,str(l).replace(' ','') )

        pickle.dump(self.qnn, open(filename, 'wb'))


    def loadexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (self.gp.GAME, self.header)

        self.exp = pickle.load( open(filename.replace(' ',''),'rb') )


    def loadnn(self, filename=None ):
        if filename == None:
            l = [len(self.gp.MOVES)] + self.param['hid'] + [self.gp.STATE_FEATURES]
            filename = 'store/qnn_%s_%s.p' % (self.gp.GAME,str(l).replace(' ',''))

        self.qnn =  pickle.load( open(filename,'rb'))


    def evaluate_avgqv(self, states):
        tar = self.qnn.predict(states)
        avgqvs =  np.mean(self.func(tar,1))

        print 'AVG OPTIMAL Q-VALUE:', avgqvs
 
    

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

