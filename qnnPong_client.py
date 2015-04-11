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

#GAME='alepong'
GAME='pygamepong'
#GAME='maze'

# pipe names
GAMEIN = 'pongfifo_in'
GAMEOUT = 'pongfifo_out'

# moves list
if GAME == 'alepong':
    MOVES = [3,4]
elif GAME == 'pygamepong':
    MOVES = [0,1]
elif GAME == 'maze':
    MOVES = [0,1]

# reset action for all
RESET = 45


# Server greet regex
if GAME == 'alepong': # www-hhh
    GREETREGEX = "([0-9]{3})-([0-9]{3})"
elif GAME == 'maze' or GAME == 'pygamepong':  # hxw
    GREETREGEX = '([0-9]+)x([0-9]+)'

# Client engage message
if GAME == 'alepong':  # r,s,k,e
    ENGAGEMSG = '1,0,0,1'
elif GAME == 'maze' or GAME == 'pygamepong':
    ENGAGEMSG = 'ENGAGE'

# Client move regex
if GAME == 'alepong':
    MOVEREGEX = '%d,0\n'
elif GAME == 'maze' or GAME == 'pygamepong':
    MOVEREGEX = '%d\n'


# Maximum number of moves in one episode
MAXMOVES = 10000

# qnn input
if GAME == 'alepong':
    STATE_FEATURES = 784
elif GAME == 'maze':
    STATE_FEATURES = 2
elif GAME == 'pygamepong':
    STATE_FEATURES = 120
    #STATE_FEATURES = 14

# Maximising q function
MAXIMISE = False

##################################################


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
        self.exploit = 0.7
        self.param = nnparam


        if nnparam['layers'] == 0:
            layers = [ PerceptronLayer(len(MOVES), STATE_FEATURES, nnparam['out']) ]
        else:
            layers = [PerceptronLayer(len(MOVES), nnparam['hid'][0], nnparam['out'])]
            for l in xrange(nnparam['layers']-1):
                layers.append( PerceptronLayer(nnparam['hid'][l], nnparam['hid'][l+1]) )
            layers.append( PerceptronLayer(nnparam['hid'][-1], STATE_FEATURES) )
 
        if MAXIMISE: self.func = np.max
        else: self.func = np.min

        self.qnn = qnn.Qnn( layers, func=self.func )
        


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
        inputs = s
        tar = self.qnn.predict(inputs).T
        aqs = [ [a,tar[a]] for a in self.movesa(s)]

        return sorted(aqs, key=lambda x:x[1], reverse= MAXIMISE)


    def qv_set(self,sa):
        ''' set the q value of state s action a'''
        s, s_prime, a, r, term = sa
        self.qnn.train(s, s_prime, a, r, self.gamma, term, self.param)

   
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
        if t: return r

        a,q = self.qv(s)[0]
        return r + self.lam*q


    def train(self, epoch=1000, replay_rate=4000):
        """ Trains the Q matrix on the maze
        
        Inputs:
            epoch: number of cycles
        """
        self.evaluate(1000)

        exp = {}
        tstates = None

        for ep in range(epoch):
            if ((ep+1) % 100) == 0 :
                print '[qlearn] epoch:', ep+1
                print 'EXP size:', len(exp)
                self.evaluate()
                self.evaluate_avgqv(tstates)
                self.saveexp(exp)
                self.savenn()

            if ep > 0 and ep < 99:
                for i in xrange(5):
                    ind = random.choice(exp.keys())
                    if tstates != None: tstates = np.vstack((tstates,exp[ind][2]))
                    else: tstates = exp[ind][2]  # index 2 is reconstructed state

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(MOVEREGEX % RESET) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]
            s, r, t = self.parse(response)
            sf = self.reconstruct(s)

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
                sf = self.reconstruct(s)
                s_f = self.reconstruct(s_)
                exp[( s, a, s_ )] = (r_, t_, sf, s_f)

                # train one                
                self.replay(exp)

                if t_ == 1 or j == MAXMOVES-1:
                    # Terminal state
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    break
                else:
                    # current state is next state
                    s, r = s_, r_
                    sf = s_f

            # train nn on exp
            for i in range( min(len(exp),replay_rate) ):
               self.replay(exp)

        # Final evaluation
        self.evaluate(testcount = 1000)


    def replay(self, exp):
        ''' Train qnn based on experience'''

        ind = random.choice(exp.keys())
        r_, t_, sf, s_f = exp[ind]
        s,a,s_ = ind

        self.qv_set((sf, s_f, a, r_, t_))
              


    def saveexp(self, exp, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (GAME, self.header)

        pickle.dump(exp, open(filename.replace(' ',''), 'wb'))


    def savenn(self, filename= None):
        if filename == None:
            l = [len(MOVES)] + self.param['hid'] + [STATE_FEATURES]
            filename = 'store/qnn_%s_%s.p' % (GAME,str(l).replace(' ','') )

        pickle.dump(self.qnn, open(filename, 'wb'))


    def loadexp(self, filename=None):
        if filename == None:
            filename = 'store/exp_%s_%s.p' % (GAME, self.header)

        return pickle.load( open(filename.replace(' ',''),'rb') )


    def loadnn(self, filename=None ):
        if filename == None:
            l = [len(MOVES)] + self.param['hid'] + [STATE_FEATURES]
            filename = 'store/qnn_%s_%s.p' % (GAME,str(l).replace(' ',''))

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
                        totalscore += score
                    
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    break

                sf = self.reconstruct(s)
                a = self.pi(sf,opt=1)
                self.fout.write(MOVEREGEX % (a,)) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds
 

    ### game specific functions ###

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

    param = {
             'learn_rate':0.2,
             'layers': 3,
             'hid':[400,400,400],
             'out':'sigmoid'
            }

    # BEGIN
    q1 = gameclient( nnparam=param)

    #q1.loadnn()
    #q1.evaluate()

    print param

    time1 = time.time()
    q1.train(epoch=5000)
    time2 = time.time()

    print 'TIME:', time2-time1
