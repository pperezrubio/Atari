import re
import numpy as np
import random
import time
import sys, os
import itertools
import cPickle as pickle

from mlp import *
import qnn

# Settings for different gaming environments ####


# GAME: pong
GAME='pygamepong'

# pipe names
GAMEIN = 'pongfifo_in'
GAMEOUT = 'pongfifo_out'

# moves list
# ALE
#MOVES = [3,4]
# maze
#MOVES = [0,1,2,3]
# pong
MOVES = [0,1]
# reset action for all
RESET = 45

# ALE: www-hhh
#GREETREGEX = "([0-9]{3})-([0-9]{3})"
# maze: hxw
GREETREGEX = '([0-9]+)x([0-9]+)'

# ALE: r,s,k,e
#ENGAGEMSG = '1,0,0,1'
# maze
ENGAGEMSG = 'ENGAGE'


# ALE
MOVEREGEX = '%d,0\n'
# maze
MOVEREGEX = '%d\n'


MAXMOVES = 10000

#maze
STATE_FEATURE = 2
#pong
#STATE_FEATURE = 2400
STATE_FEATURE = 14

# Maximising q function
MAXIMISE = False

##################################################


# Generate moves table to discover problem


class gameclient():

    def __init__(self, nnparam={}):
        """ Initialise qlearning module
        """
        
        self.fin = open( GAMEOUT )
        self.fout = open( GAMEIN ,'w')
        
        self.header = self.handshake()

        # lambda
        self.gamma = 0.8
        # max exploit probability
        self.exploit = 0.9
        self.param = nnparam

        l1 = PerceptronLayer(len(MOVES), 30)
        #l2 = PerceptronLayer(30,30)
        l2 = PerceptronLayer(30, STATE_FEATURE)
 
        #self.qnn = qnn.Qnn(param = nnparam)
        self.qnn = qnn.Qnn([l1,l2])



    def handshake(self):
        # initial handshake

        while 1:
            str_in = self.fin.readline().strip()

            # server greeting phrase
            reg = re.match( GREETREGEX ,str_in)
            if reg:
                self.fout.write( ENGAGEMSG + '\n')
                self.fout.flush()
                break
                
        rtn = [int(i) for i in reg.groups()]        
                
        return rtn
        

    def qv(self,s):
        ''' return the q values of all actions of state sorted'''
        # update all q-values for state s 
        #inputs = np.array([s])
        inputs = s
        tar = self.qnn.predict(inputs).T
        aqs = [ [a,tar[a]] for a in self.movesa(s)]

        return sorted(aqs, key=lambda x:x[1], reverse= MAXIMISE)
        


    def qv_set(self,sa):
        ''' set the q value of state s action a'''
        s, s_prime, a, r, term = sa
        #s = np.array([s])
        #s_prime = np.array([s_prime])

        self.qnn.train(s, s_prime, a, r, self.gamma, term, self.param)

    
    def movesa( self, s):
        '''Game specific. return available action for given state'''
        return MOVES


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

        state = s.strip('()').split(',')        
        state = [float(c)/([w,h][i % 2]) for i,c in enumerate(state)]
        #state = [float(c) for c in state]        

        return tuple(state)


    def pi(self,s, progress = 0, opt=0):
        """ Select an action given a state based on present policy (may be optimal or not)
        Input:
            s: current state, should not be terminal state
        Return:
            a: action based on present policy 
        """
        
        # get action list for given state
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

        aqs = self.qv(s)
        if t: return r

        a,q = aqs[0]
        return r + self.lam*q


    def train(self, epoch=1000):
        """ Trains the Q matrix on the maze
        
        Inputs:
            epoch: number of cycles
        """
        self.evaluate(1000)

        exp = {}

        for ep in range(epoch):
            if ((ep+1) % 100) == 0 :
                print '[qlearn] epoch:', ep+1
                print 'EXP size:', len(exp)
                self.evaluate()
                self.saveexp(exp)
                self.savenn()

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(MOVEREGEX % RESET) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]
            s, r, t = self.parse(response)
            sf = reconstruct(s)

            # if first state is terminal already, next epoch
            if t == 1:
                self.fout.write(MOVEREGEX % RESET) 
                self.fout.flush()
                continue

            for j in range( MAXMOVES ):
            
                # action
                a = self.pi(sf, progress=float(ep)/epoch ,opt=0)

                # send in action
                self.fout.write(MOVEREGEX % (a,)) 
                self.fout.flush()

                # get next state
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                s_, r_, t_ = self.parse(response)

                # add (state,action) pair to experience
                sf = reconstruct(s)
                s_f = reconstruct(s_)
                exp[( s, a, s_ )] = (r_, t_, sf, s_f)

                # train one
                self.replay(exp)
 

                # Terminal state
                if t_ == 1 or j == MAXMOVES-1:
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    break
                else:
                    s, r = s_, r_
                    sf = s_f

            # train nn on exp
            for i in range(len(exp)):
               self.replay(exp)

        # Final evaluation
        self.evaluate(testcount = 1000)



    def replay(self, exp):

        ind = random.choice(exp.keys())
        r_, t_, sf, s_f = exp[ind]
        s,a,s_ = ind

        self.qv_set((sf, s_f, a, r_, t_))
              


    def saveexp(self, exp, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (GAME, self.header)
        pickle.dump(exp, open(filename, 'wb'))

    def savenn(self, filename='store/qnn_%s.p' % GAME ):
        pickle.dump(self.qnn, open(filename, 'wb'))

    def loadexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (GAME, self.header)
        return pickle.load( open(filename,'rb') )

    def loadnn(self, filename='store/qnn_%s.p' % GAME ):
        self.qnn =  pickle.load( open(filename,'rb'))


    def evaluate (self, testcount = 100):
        '''Game Specific'''

        rounds = 0
        totalmoves = 0
        totalscore = 0
        score = 0

        for i in range(testcount):

            j = 0
            score = 0

            # restart game
            str_in = self.fin.readline()
            self.fout.write(MOVEREGEX % RESET) 
            self.fout.flush()

            while 1:
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]

                s,r,t = self.parse(response)
                score += r

                # Terminal state
                if t==1 or j == MAXMOVES:
                    if j > 0:
                        rounds += 1
                        #print score
                        totalscore += score
                    
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    break

                sf = reconstruct(s)
                a = self.pi(sf,opt=1)
                self.fout.write(MOVEREGEX % (a,)) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds


def reconstruct( s ):
    ''' Game specific
    '''

    return np.array([s])

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

    param = {
             'learn_rate':0.2,
            }

    # BEGIN
    q1 = gameclient( nnparam=param)

    q1.loadnn()
    q1.evaluate()

    print param

    time1 = time.time()
    #q1.train(epoch=5000)
    time2 = time.time()

    print 'TIME:', time2-time1
