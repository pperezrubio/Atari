import re
import numpy as np
import random
import time
import sys, os
import itertools

from mlp import *
import qnn

# Settings for different gaming environments ####


# GAME: pong


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
STATE_FEATURE = 14


# Maximising q function
MAXIMISE = True

##################################################


# Generate moves table to discover problem


class gameclient():

    def __init__(self, td=10, nnparam={}):
        """ Initialise qlearning module
        """
        
        self.fin = open( GAMEOUT )
        self.fout = open( GAMEIN ,'w')
        
        self.header = self.handshake()

        # Temporal Difference window
        self.td = td
        # lambda
        self.gamma = 0.8
        # exploration probability
        self.explore = 0.2
        self.param = nnparam


        l1 = PerceptronLayer(len(MOVES), 100, "sum")
        l2 = PerceptronLayer(100,100)
        l3 = PerceptronLayer(100, STATE_FEATURE)
 
        #self.qnn = qnn.Qnn(param = nnparam)
        self.qnn = qnn.Qnn([l1,l2,l3])



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
        inputs = self.prepare_input([s])
        tar = self.qnn.predict(inputs.T).T

        aqs = [ [a,tar[a]] for a in self.movesa(s)]

        return sorted(aqs, key=lambda x:x[1], reverse= MAXIMISE)
        


    def qv_set(self,sa):
        ''' set the q value of state s action a'''
        

        s, s_prime, a, r, term = sa

        s = np.array([s])
        s_prime = np.array([s_prime])

        self.qnn.train(s, s_prime, a, r, self.gamma, term, self.param)

    
    def prepare_input(self,sa):
        ''' Convert input into numpy array
        Input:
            sa: a list of states (and action pair)
        '''
        rtn = [ list(s) for s in sa]
        
        return np.array(rtn).T


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

        s = s.split(',')        
        s = [float(c)/([w,h][i % 2]) for i,c in enumerate(s)]
        
        return tuple(s)


    def normalise_move(self,a):
        '''Game specific '''
        minm = min(MOVES)
        rang = max(MOVES)-minm
    
        return (a-minm)/float(rang)


    def pi(self,s, opt=0):
        """ Select an action given a state based on present policy (may be optimal or not)
        Input:
            s: current state, should not be terminal state
        Return:
            a: action based on present policy 
        """
        
        # if given state is terminal state
        #if self.maze.terminal(s): return None

        # get action list for given state
        aqs = self.qv(s)

        if random.random() < self.explore*(1-opt):
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


    def stoch_train(self, epoch=1000):
        """ Trains the Q matrix on the maze
        
        Inputs:
            epoch: number of cycles
        """
        self.evaluate()

        for ep in range(epoch):
            if ((ep+1) % 1000) == 0 :
                print '[qlearn] epoch:', ep+1
                self.evaluate()

            # restart game
            str_in = self.fin.readline()
            # send in reset signal
            self.fout.write(MOVEREGEX % RESET) 
            self.fout.flush()

            # get initial state
            str_in = self.fin.readline()
            response = str_in.strip().split(':')[:-1]
            s, r, t = self.parse(response)

            # if first state is terminal already, next epoch
            if t == 1:
                self.fout.write(MOVEREGEX % RESET) 
                self.fout.flush()
                continue

            # path for this epoch
            path = []

            for j in range( MAXMOVES ):
            
                # action
                a = self.pi(s, opt=0)

                # send in action
                self.fout.write(MOVEREGEX % (a,)) 
                self.fout.flush()

                # get next state
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                s_, r_, t_ = self.parse(response)

                # add (state,action) pair to path
                path.append((s, r, a, s_, r_, t_))

                # if end state is still not reached within td window
                if len(path)>= self.td:
                    for (ps, pr, pa, ps_, pr_, pt_) in path[::-1]:
                        self.qv_set((ps, ps_, pa, pr_, pt_))
                    path.pop(0)
                    
                # Terminal state
                if t_ == 1 or j == MAXMOVES-1:
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    break
                else:
                    s, r = s_, r_ 

            # update q-values for all states travelled in this epoch,
            # starting from the last state encountered
            for (s, r, a, s_, r_, t_) in path[::-1]:
                self.qv_set((s, s_, a, r_, t_))
                
        # Final evaluation
        self.evaluate(testcount = 1000)


    def batch_train(self, epoch=100, batch=1):
        """ Trains the Q matrix on the maze
        
        Inputs:st
            epoch: number of cycles
        """

        self.evaluate()

        for ep in range(epoch):

            if ((ep+1) % 100) == 0 :
                print '[qlearn] epoch:', ep+1
                self.evaluate()

            batchsa = []
            batchv = []
            
            for ba in range(batch):
                #print 'batch',ba 

                # restart game
                str_in = self.fin.readline()
                # send in reset signal
                self.fout.write(MOVEREGEX % RESET) 
                self.fout.flush()

                # get initial state
                str_in = self.fin.readline()
                response = str_in.strip().split(':')[:-1]
                s, r, t = self.parse(response)

                # if first state is terminal already, next epoch
                if t == 1:
                    self.fout.write(MOVEREGEX % RESET) 
                    self.fout.flush()
                    continue

                # path for this epoch
                path = []

                for j in range( MAXMOVES ):
                
                    # action
                    a = self.pi(s, opt=0)

                    # send in action
                    self.fout.write(MOVEREGEX % (a,) ) 
                    self.fout.flush()

                    # get next state
                    str_in = self.fin.readline()
                    response = str_in.strip().split(':')[:-1]
                    s_, r_, t_ = self.parse(response)

                    # add (state,action) pair to path
                    path.append((s, r, a, s_, r_, t_))

                    # if end state is still not reached within td window
                    if len(path)>= self.td:
                        pathsa = []
                        pathv = []
                    
                        for (ps, pr, pa, ps_, pr_, pt_) in path[::-1]:
                            pathsa.append( (ps, pa) )
                            pathv.append( pr + self.lam * self.maxq(ps_, pr_, pt_) )
                            
                            
                        batchsa.append(pathsa)
                        batchv.append(pathv)
                        path.pop(0)
                        
                    # Terminal state
                    if t_ == 1 or j == MAXMOVES-1:
                        self.fout.write('%s\n' % RESET) 
                        self.fout.flush()
                        break
                    else:
                        s, r = s_, r_ 


                # update q-values for all states travelled in this epoch,
                # starting from the last state encountered
                pathsa = []
                pathv = []
                
                for (ps, pr, pa, ps_, pr_, pt_) in path[::-1]:
                    pathsa.append( (ps, pa) )
                    pathv.append( pr + self.lam * self.maxq(ps_, pr_, pt_) )
                    
                batchsa.append(pathsa)
                batchv.append(pathv)
                    
            batchsa = timzip(*batchsa)
            batchv = timzip(*batchv)
                   
            for i,t in zip(batchsa,batchv):
                self.qv_set(i, t)

        # Final evaluation
        self.evaluate(testcount = 1000)        


    def train(self, train_method =0, epoch=1000, batch=100):
        if train_method == 0:
            self.stoch_train(epoch=epoch)
        else:
            self.batch_train(epoch=epoch, batch = batch)



    def evaluate (self, testcount = 50):
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

                a = self.pi(s,opt=1)
                self.fout.write(MOVEREGEX % (a,)) 
                self.fout.flush()
                
                j += 1
 
            totalmoves += j

        print 'AVG NUM MOVES:', float(totalmoves)/rounds
        print 'AVG SCORE:', float(totalscore)/rounds


def timzip(*args):
    a = []
    for i in itertools.izip_longest(*args):
        b = list(i)
        while None in b:
            b.remove(None)
        a.append(b)
    return a

if __name__ == '__main__':

    # train: 0 - stochic, 1 - batch
    t = 0

    td = 1000
   
    param = {
             'learn_rate':0.2,
            }


    # BEGIN

    q1 = gameclient(td = td, nnparam=param)

    print 'TRAIN:', ['stoch','batch' ][t]
    print 'TD:', td

    print param

    time1 = time.time()
    #q1.train(train_method=1, epoch=2000, batch=10)
    q1.train(train_method=0, epoch=4000, batch=1)
    time2 = time.time()

    print 'TIME:', time2-time1


