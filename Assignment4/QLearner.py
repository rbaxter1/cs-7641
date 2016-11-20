"""
Template for implementing QLearner  (c) 2015 Tucker Balch
This is Rob Baxter's implementation from ML4T (dyna removed!)
Modified to work with the mdptoolbox structure
"""

import numpy as np
import numpy as _np
import time as _time
import math as _math
import random as rand

import mdptoolbox.util as _util
import mdptoolbox.mdp

## from mdptoolbox.mdp
def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


# mdptoolbox QLearning customized by Rob Baxter
class QLearningEx(mdptoolbox.mdp.MDP):

    """A discounted MDP solved using the Q learning algorithm.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an
        integer greater than the default value. Defaut: 10,000.

    Data Attributes
    ---------------
    Q : array
        learned Q matrix (SxA)
    V : tuple
        learned value function (S).
    policy : tuple
        learned optimal policy (S).
    mean_discrepancy : array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).

    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import mdptoolbox, mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    >>> ql.run()
    >>> ql.Q
    array([[ 11.198909  ,  10.34652034],
           [ 10.74229967,  11.74105792],
           [  2.86980001,  12.25973286]])
    >>> expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (0, 1, 1)

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> np.random.seed(0)
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
    >>> ql.run()
    >>> ql.Q
    array([[ 33.33010866,  40.82109565],
           [ 34.37431041,  29.67236845]])
    >>> expected = (40.82109564847122, 34.37431040682546)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, start, goals, n_restarts=1000, alpha = 0.2, gamma = 0.9, rar = 0.9, radr = 0.99, n_iter=100000):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000." 

        # We don't want to send this to MDP because _computePR should not be
        # run on it, so check that it defines an MDP
        _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        
        #self.learner = QLearner(self.S, self.A, alpha, gamma, rar, radr, dyna, verbose=False)
        #self.learner.querysetstate(a)
        
        self.R = reward

        #self.discount = discount
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.orig_rar = rar
        self.radr = radr
        #self.num_actions = self.A
        self.start = start
        self.goals = goals
        self.n_restarts = n_restarts
        
        # Initialisations
        #self.Q = _np.zeros((self.S, self.A))
        self.Q = np.random.uniform(-1, 1, (self.S, self.A))
        
        self.mean_discrepancy = []
        
    def __querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        a = self.__select_action(s)
        
        self.a = a
        return a

    # this is the "policy function"
    def __query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        
        # update Q
        self.__update_q(self.s, self.a, s_prime, r)
        
        # select next action (could be random or best)
        a = self.__select_action(s_prime)
        
        # save a and a
        self.s = s_prime
        self.a = a
        
        # decay rar
        self.rar = self.rar * self.radr
        
        return a
    
    def __select_action(self, s_prime):
        s = np.random.binomial(1, self.rar, 1)[0]
        if s == 1:
            a = np.random.randint(0, self.A) # -1
        else:
            a = np.argmax(self.Q[s_prime,:])
        
        return a
    
    def __move(self, s, a):
        possible_moves = np.where(self.P[a][s] > 0.)
        num_possible_moves = len(possible_moves)
        if num_possible_moves == 0:
            print('no possible moves!')
            s_new = s
        elif num_possible_moves == 1:
            s_new = int(possible_moves[0])
        else:
            # TODO handle probablilty
            raise 'not supported'
         
        ## TODO use T and R
        #self.P[a, s]
        #p_s_new = _np.random.random()
        #p = 0
        #s_new = -1
        #while (p < p_s_new) and (s_new < (self.S - 1)):
        #    s_new = s_new + 1
        #    p = p + self.P[a][s, s_new]
        
        try:
            r = self.R[a][s, s_new]
        except IndexError:
            try:
                r = self.R[s, a]
            except IndexError:
                r = self.R[s]
                
        return s_new, r
    
    def __update_q(self, s, a, s_prime, r):
        # argmax does not tie break well, it just takes the first value
        #argmaxa = np.argmax(self.Q[s_prime])
        
        # better option...
        ## Random tie breaking option (slower)
        n = np.amax(self.Q[s_prime,:])
        i = np.nonzero(self.Q[s_prime,:] == n)[0]
        argmaxa = rand.choice(i)

        # caluculate discounted reward
        dr = r + self.gamma * self.Q[s_prime, argmaxa]
        
        # update the Q matrix
        self.Q[s, a] = (1. - self.alpha) * self.Q[s, a] + self.alpha * dr
        
    def run(self):
        
        # restart logic
        for i in range(self.n_restarts):
            
            # set / reset state to starting position
            s = self.start
            
            # get first action from starting positions
            a = self.__querysetstate(s)
            
            c = 0
            while (s not in self.goals) & (c < self.max_iter):
                c += 1
                
                # get next state and reward based on action
                s_prime, r = self.__move(s, a)
                
                # give learner reward and get next action based on policy
                a = self.__query(s_prime, r)
                
                # update s for this loop
                s = s_prime
                
                if s in self.goals:
                    print('goal! iterations: ', c)
                
                # compute the value function and the policy
                self.V = self.Q.max(axis=1)
                self.policy = self.Q.argmax(axis=1)
                
                
    
    '''
    def run(self):
        self.run(None, None)
        
    def run(self, start_pos, goal_pos):
        # Run the Q-learning algoritm.
        discrepancy = []

        self.time = _time.time()

        # initial state choice
        #s = _np.random.randint(0, self.S)
        # equivalent of querysetstate
        if start_pos == None:
            s = _np.random.randint(0, self.S)
        else:
            s = start_pos
        
        for n in range(1, self.max_iter + 1):

            # Reinitialisation of trajectories every 100 transitions
            #if (n % 100) == 0:
            #    s = _np.random.randint(0, self.S)
            # reset to the start when the goal is reached
            if goal_pos != None:
                if s == goal_pos:
                    if start_pos == None:
                        s = _np.random.randint(0, self.S)
                    else:
                        s = start_pos
            else:
                if (n % 100) == 0:
                    if start_pos == None:
                        s = _np.random.randint(0, self.S)
                    else:
                        s = start_pos
                
            
            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            #pn = _np.random.random()
            #if pn < (1 - (1 / _math.log(n + 2))):
            #    # optimal_action = self.Q[s, :].max()
            #    a = self.Q[s, :].argmax()
            #else:
            #    a = _np.random.randint(0, self.A)
            a = self.__select_action(s)

            # Simulating next state s_new and reward associated to <s,s_new,a>
            #p_s_new = _np.random.random()
            #p = 0
            #s_new = -1
            #while (p < p_s_new) and (s_new < (self.S - 1)):
            #    s_new = s_new + 1
            #    p = p + self.P[a][s, s_new]

            #try:
            #    r = self.R[a][s, s_new]
            #except IndexError:
            #    try:
            #        r = self.R[s, a]
            #    except IndexError:
            #        r = self.R[s]

            # Updating the value of Q
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = (1 / _math.sqrt(n + 2)) * delta
            self.Q[s, a] = self.Q[s, a] + dQ

            # current state is updated
            s = s_new

            # Computing and saving maximal values of the Q variation
            discrepancy.append(_np.absolute(dQ))

            # Computing means all over maximal Q variations values
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(_np.mean(discrepancy))
                discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self.time = _time.time() - self.time

        # convert V and policy to tuples
        self.V = tuple(self.V.tolist())
        self.policy = tuple(self.policy.tolist())

    '''

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # "initialize q table with small random numbers"
        # source: video 3-06 Learning Procedure
        sz = (num_states, num_actions)
        self.Q = np.random.uniform(-1, 1, sz)
        
    def __select_action(self, s_prime):        
        s = np.random.binomial(1, self.rar, 1)[0]
        if s == 1:
            action = np.random.randint(0, self.num_actions) # -1
        else:
            action = np.argmax(self.Q[s_prime,:])
        
        if self.verbose: print("s =", s_prime, "a =", action, "r =", r)            
        return action
    
    def __update_q(self, s, a, s_prime, r):
        # argmax does not tie break well, it just takes the first value
        #argmaxa = np.argmax(self.Q[s_prime])
        
        # better option...
        ## Random tie breaking option (slower)
        n = np.amax(self.Q[s_prime,:])
        i = np.nonzero(self.Q[s_prime,:] == n)[0]
        argmaxa = rand.choice(i)

        # caluculate discounted reward
        dr = r + self.gamma * self.Q[s_prime, argmaxa]
        
        # update the Q matrix
        self.Q[s, a] = (1. - self.alpha) * self.Q[s, a] + self.alpha * dr
    
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.__select_action(s)
        if self.verbose: print("s =", s, "a =", action)
        self.a = action
        return action

    # this is the "policy function"
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        
        # decay rar
        self.rar = self.rar * self.radr
        
        # update Q
        self.__update_q(self.s, self.a, s_prime, r)
        
        # select answer
        action = self.__select_action(s_prime)
        
        # save a and a
        self.s = s_prime
        self.a = action
        
        return action
    
if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
