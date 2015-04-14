import re
import numpy as np
import random
import time
import sys, os
import itertools
import cPickle as pickle

from mlp import *
import qnn


class gameclient():

    def __init__(self, gparm, nnparam):
        """ Initialise qlearning module
        """
        
        self.gp = gparm

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
        self.exp = {}


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
        

    def qv(self,s):
        ''' return the q values of all actions of state sorted'''
        inputs = s
        tar = self.qnn.predict(inputs).T

        aqs = [ [a, tar[a]] for a in self.validmoves(s)]

        return sorted(aqs, key=lambda x:x[1], reverse= self.gp.MAXIMISE)


    def qv_set(self,sa):
        ''' set the q value of state s action a'''
        s, s_prime, a, r, term = sa
        self.qnn.train(s, s_prime, a, r, self.gamma, term, self.param, self.func)

   
    def pi(self,s, progress = 0, opt=0, rand=0):
        """ Select an action given a state based on present policy (may be optimal or not)
        Input:
            s: current state, should not be terminal state
        Return:
            a: action based on present policy 
        """
        # get action list for given state
        if rand: return random.choice(self.validmoves(s))

        aqs = self.qv(s)

        if random.random() > (self.exploit*progress + opt):
            # exploration (select action randomly)
            a,q = aqs[int(random.random()*len(aqs))]
        else:
            # pick action with max/min q
            a,q = aqs[0]
        
        return a


    def maxq(self, s, r, t):
        """ return max q value of given state s

        Input:
            s: given state
            r: reward of given state
            t: whether state is terminal
        
        Returns:
            q: if s is terminal state, return reward
               else return reward + max q
        """
        if t: return r

        a,q = self.qv(s)[0]
        return r + self.lam*q


    def train(self, epoch=1000, replay_rate=2000):
        """ Trains the Q matrix on the maze
        
        Inputs:
            epoch: number of cycles
        """
        #self.evaluate(10)

        tstates = []

        for ep in range(epoch):
            if ((ep+1) % 100) == 0 :
                print '[qlearn] epoch:', ep+1
                print 'EXP size:', len(self.exp)
                self.evaluate()
                self.evaluate_avgqv(tstates)
                self.saveexp()
                self.savenn()

            if ep > 0 and len(tstates) < 500:
                for i in xrange(5):
                    ind = random.choice(self.exp.keys())
                    if tstates != []: tstates = np.vstack((tstates,self.exp[ind][2]))
                    else: tstates = exp[ind][2]  # index 2 is reconstructed state

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]
            s, r, t = self.parse(response)
            sf = self.reconstruct(s)

            # if first state is terminal already, next epoch
            if t == 1:
                self.fout.write(self.gp.MOVEREGEX % self.gp.RESET) 
                self.fout.flush()
                continue

            for j in range( self.gp.MAXMOVES ):
            
                # action
                a = self.pi(sf, progress=float(ep)/epoch ,opt=0)

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


    def replay(self):
        ''' Train qnn based on experience'''

        ind = random.choice(self.exp.keys())
        r_, t_, sf, s_f = self.exp[ind]
        s,a,s_ = ind

        self.qv_set((sf, s_f, a, r_, t_))
              


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


    def evaluate (self, testcount = 500):
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
                a = self.pi(sf,opt=1)
                self.fout.write(self.gp.MOVEREGEX % (self.gp.MOVES[a],)) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds
 
    

    ### game specific functions ###

    def validmoves(self, s):
        '''Game specific. return available action for given state. '''

        m = range(len(self.gp.MOVES))
        return m


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

