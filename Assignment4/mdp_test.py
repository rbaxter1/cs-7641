from mdp import *

import mdptoolbox, mdptoolbox.example
import numpy as np
import pandas as pd
import random as rand
import math
from plot_helper import *

from QLearner import QLearningEx
from QLearning import QLearning

class part1():
    def __init__(self):
        self.out_dir = 'output'
        
        ## space definitions
        self.space_free = 0
        self.space_start_free = 1
        self.space_goal = 2
        self.space_void = 3
        self.space_quicksand = 4
        
        self.rewards = [-1, -1, 1, None, -100]
        
        ## action definitions
        self.action_up = 0
        self.action_down = 1
        self.action_left = 2
        self.action_right = 3
        self.action_total_num = 4
        
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        
        ## random grid config
        self.max_void_coverage = 0.25
        self.max_quicksand_coverage = 0.25
    
    def __get_unused_position(self, used, rows, cols, max_tries=100):
        i = 0
        while True:
            p = (np.random.randint(0, rows), np.random.randint(0, cols))
            if not p in used:
                used.append(p)
                return used, p
            i += 1
            if i > max_tries:
                return used, None
        
    def __generate_random_grid(self, rows, cols, pct_void=0., pct_quicksand=0.):
        grid = np.zeros((rows, cols))
        used = []
        
        # find a start position
        used, p = self.__get_unused_position(used, rows, cols)
        if p == None:
            raise()
        
        grid[p] = self.space_start_free
                
        # find an end position
        used, p = self.__get_unused_position(used, rows, cols)
        if p == None:
            raise()
        
        grid[p] = self.space_goal
        
        # add void
        n = math.floor(rows * cols * self.max_void_coverage * pct_void)
        for i in range(n):
            # find an available space
            used, p = self.__get_unused_position(used, rows, cols)
            if p == None:
                break
            
            grid[p] = self.space_void
        
        # add quicksand
        n = math.floor(rows * cols * self.max_quicksand_coverage * pct_quicksand)
        for i in range(n):
            # find an available space
            used, p = self.__get_unused_position(used, rows, cols)
            if p == None:
                break
            
            grid[p] = self.self.space_quicksand
        
        # done
        return grid
    
    def test_move(self, pos, grid, action_index):
        newpos = tuple(map(sum, zip(pos, self.actions[action_index])))
        print('ai: ', action_index, 'newpos: ', newpos)   
        if newpos[0] < 0:
            return None
        if newpos[0] >= grid.shape[0]:
            return None
        if newpos[1] < 0:
            return None
        if newpos[1] >= grid.shape[1]:
            return None
        if grid[newpos] == self.space_void:
            return None
        
        return newpos
    
    def __get_grid_terminals(self, grid):
        flat_grid = grid.reshape((grid.shape[0] * grid.shape[1]))
        start_index = np.where(flat_grid == self.space_start_free)
        assert len(start_index) == 1, "Only 1 starting position is allowed."
        start = int(start_index[0])
        goal_indices = np.where(flat_grid == self.space_goal)
        assert len(goal_indices) <= 1, "Must have at least 1 goal position."
        goals = goal_indices[0].tolist()
        
        return start, goals
    
    def __create_reward_grid(self, grid):
        rows, cols = grid.shape
        num_states = rows * cols
        
        # for the A, S, S reward matrix, just copy the grid rewards to 
        # each A, S... not very pythonic
        r = grid.reshape(num_states)
        for i in range(len(r)):
            r[i] = self.rewards[r[i]]
        
        return r
    
    def __convert_grid_to_mdp(self, grid, chance_wrong_action=0.):
        rows, cols = grid.shape
        num_states = rows * cols
        
        ## alpha and omega
        start, goals = self.__get_grid_terminals(grid)
        ## transitions (A, S, S)
        T = np.zeros((len(self.actions), num_states, num_states))
        ## rewards (A, S, S)
        R = np.zeros((len(self.actions), num_states, num_states))
        ## rewards (S, A)
        R2 = np.zeros((num_states, len(self.actions)))
        
        # for the A, S, S reward matrix, just copy the grid rewards to 
        # each A, S... not very pythonic
        r_temp = self.__create_reward_grid(grid)
            
        for i in range(len(self.actions)):
            for s in range(num_states):
                R[i, s] = r_temp
                
        for i in range(len(self.actions)):
            for s in range(num_states):
                pos = divmod(s, cols)
                print('i: ', i, 's: ', s, 'pos: ', pos)
                t = np.zeros(grid.shape)
                #r = np.zeros(grid.shape)
                
                if grid[pos] == self.space_goal:
                    #t[pos] = 0.
                    
                    t[pos] = 1.
                    #r[pos] = 1. # not sure
                    R2[s,i] = 1.
                else:
                    
                    # TODO insert randomness
                    #if rand.uniform(0.0, 1.0) <= chance_wrong_action:
                    #    a = rand.randint(0, len(self.actions)-1)
                    #    print('random action')
                    #else:
                    a = i
        
                    # try move
                    newpos = self.test_move(pos, grid, a)
                    if newpos != None:
                        t[newpos] = 1.
                        #r[newpos] = self.rewards[int(grid[newpos])]
                        R2[s,i] = self.rewards[int(grid[newpos])]
                    else:
                        t[pos] = 1. # can't move
                        #r[pos] = self.rewards[0]
                        R2[s,i] = -1
                        
                # reshape
                t.shape = num_states
                #r.shape = num_states
                T[i, s] = t
                #R[i, s] = r
               
        return T, R, R2, start, goals
    
    def run2(self):
        fn = './input/grid1.csv'
        grid = pd.read_csv(fn, header=None).values
        
        r = self.__create_reward_grid(grid)
        start, goals = self.__get_grid_terminals(grid)
        
        mdp = GridMDP([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-100,-100,-100,-100,-100,-100,-100,-100,-100],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]],
                      terminals=[(9, 5)],
                      init=(0, 5))
        '''
        mdp = GridMDP([[-0.01, -0.01, -0.01, +1.00],
                       [-0.01, None,  -0.01, -0.01],
                       [-0.01, -0.01, -0.01, -0.01]],
                      terminals=[(3, 1)])

        mdp = GridMDP([[-1.0, +1.0]],
                      terminals=[(0,0),(1,0)])
        '''
        
        vi = value_iteration(mdp, .01)
        print(vi)
        
        vi_grid = np.ndarray((10,10))
        for k in vi.keys():
            vi_grid[k] = vi[k]
            
            
        ph = plot_helper()
        ph.plot_heatmap(vi_grid, None, None, 'title', 'value_iter_mdp.png')
        
            
        pi = policy_iteration(mdp)
        print(pi)
        
        b = best_policy(mdp, vi)
        print(b)
        
        
    def run(self):
        print('Running part 1')
        '''
        grid = self.__generate_random_grid(2, 2, 0., 0.)
        print(grid)
        
        T, R = self.__convert_grid_to_mdp(grid)
        pi = mdptoolbox.mdp.PolicyIteration(T, R, 0.9)
        pi.run()
        print(pi.policy)
        
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.9)
        vi.run()
        print(vi.V)
        print(vi.policy)
        print(vi.iter)
        '''
        
        fn = './input/grid1.csv'
        grid = pd.read_csv(fn, header=None).values
        
        T, R, R2, start, goals = self.__convert_grid_to_mdp(grid)
        
        rar = 0.9
        q = QLearningEx(T, R, start, goals, n_restarts=1000, alpha = 0.2, gamma = 0.9, rar = rar, radr = 0.99, n_iter=100000)
        q.run()
        print(q.Q)
        
        p = np.array(q.policy)
        p.shape = grid.shape
        
        v = np.array(q.V)
        v.shape = grid.shape
        
        ph = plot_helper()
        fn = 'qlearn_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '.png'
        title = '10x10 Quicksand Grid QLearner\nalpha: ' + str(q.alpha) + ', gamma: ' + str(q.gamma) + ', rar: ' + str(q.orig_rar) + ', radr: ' + str(q.radr)
        ph.plot_heatmap(v, grid, p, title, fn)
        
        '''
        ## qlearning
        q = mdptoolbox.mdp.QLearning(T, R, 0.9)
        q.run()
        v = np.array(q.V)
        v.shape = grid.shape
        p = np.array(q.policy)
        p.shape = grid.shape
        
        fn = 'qlearner_' + str(q.discount) + '.png'
        title = '10x10 Quicksand Grid Q-Learning\ndiscount: ' + str(q.discount)
        ph.plot_heatmap(v, grid, p, title, fn)
        '''
        
        ## policy iteration
        pi = mdptoolbox.mdp.PolicyIteration(T, R, 0.9, max_iter=1000)
        pi.run()
        v = np.array(pi.V)
        v.shape = grid.shape
        p = np.array(pi.policy)
        p.shape = grid.shape
        
        fn = 'policyiter_' + str(pi.discount) + '.png'
        title = '10x10 Quicksand Grid Value Iteration\ndiscount: ' + str(pi.discount)
        ph.plot_heatmap(v, grid, p, title, fn)
        
        ## value iteration
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.9, max_iter=1000)
        vi.run()
        v = np.array(vi.V)
        v.shape = grid.shape
        p = np.array(vi.policy)
        p.shape = grid.shape
        
        fn = 'policyiter_' + str(pi.discount) + '.png'
        title = '10x10 Quicksand Grid Policy Iteration\ndiscount: ' + str(pi.discount)
        ph.plot_heatmap(v, grid, p, title, fn)
        
        
        
        
        
    
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.9)
        vi.run()
        print(vi.V)
        print(vi.policy)
        print(vi.iter)
        v = np.array(vi.V)
        v.shape = grid.shape
        #ph.plot_grid(v, 'x', 'y', 'title', 'value.png')
        ph.plot_heatmap(v, grid, 'title', 'value.png')
        
        
        

        q = QLearning(T, R, 0.9, start, goals)
        q.run()
        print(q.Q)
        
        p = np.array(q.policy)
        p.shape = grid.shape
        
        ph = plot_helper()
        v = np.array(q.V)
        v.shape = grid.shape
        #ph.plot_grid(v, 'x', 'y', 'title', 'qlearn_grid.png')
        ph.plot_heatmap(v, grid, p, '', 'qlearn_mod.png')
        
        
        #pi = policy_iteration(mdp)
        #print(pi)
        
        #b = best_policy(mdp, vi)
        #print(b)
        
        
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.5)
        vi.run()
        print(vi.V)
        print(vi.policy)
        print(vi.iter)
        v = np.array(vi.V)
        v.shape = grid.shape
        p = np.array(vi.policy)
        p.shape = grid.shape
        
        #ph.plot_grid(v, 'x', 'y', 'title', 'value.png')
        ph.plot_heatmap(v, grid, p, 'title', 'value.png')
        
        
       
        #ph.plot_grid(p, 'x', 'y', 'title', 'qlearn_grid_policy.png')
        #ph.plot_heatmap(p, grid, 'title', 'qlearn_policy.png')
        
        q = mdptoolbox.mdp.QLearning(T, R2, 0.9, 10000)
        q.run()
        print(q.Q)
        
        ph = plot_helper()
        v = np.array(q.V)
        v.shape = grid.shape
        ph.plot_grid(v, 'x', 'y', 'title', 'qlearn.png')
        ph.plot_heatmap(v, grid, 'title', 'qlearn.png')
            
        
        
        pi = mdptoolbox.mdp.PolicyIteration(T, R2, 0.9, max_iter=1000)
        pi.run()
        print(pi.policy)
        
        vi = mdptoolbox.mdp.ValueIteration(T, R, 0.9)
        vi.run()
        print(vi.V)
        print(vi.policy)
        print(vi.iter)
        v = np.array(vi.V)
        v.shape = grid.shape
        #ph.plot_grid(v, 'x', 'y', 'title', 'value.png')
        ph.plot_heatmap(v, grid, 'title', 'value.png')
        
        
        
        P, R = mdptoolbox.example.forest()
        
        #mdptoolbox.error.InvalidError: 'PyMDPToolbox - The transition probability array must have the shape
        #(A, S, S) with S : number of states greater than 0 and A : number of actions greater than 0. i.e. R.shape = (A, S, S)'
        
        sz = (10, 10, 10)
        P2 = np.zeros(sz)* 1/100
        #for i in range(100): 
        #    P[0,i] = 1
            
       # P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
        R2 = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-100,-100,-100,-100,-100,-100,-100,-100,-100],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                      [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
        np.random.seed(0)
        ql = mdptoolbox.mdp.QLearning(P2, R2, 0.9)
        ql.run()
        ql.Q



        
        mdp = GridMDP([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-100,-100,-100,-100,-100,-100,-100,-100,-100],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                       [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]],
                      terminals=[(9, 5)],
                      init=(0, 5))
        '''
        mdp = GridMDP([[-0.01, -0.01, -0.01, +1.00],
                       [-0.01, None,  -0.01, -0.01],
                       [-0.01, -0.01, -0.01, -0.01]],
                      terminals=[(3, 1)])

        mdp = GridMDP([[-1.0, +1.0]],
                      terminals=[(0,0),(1,0)])
        '''
        
        vi = value_iteration(mdp, .01)
        print(vi)
        
        pi = policy_iteration(mdp)
        print(pi)
        
        b = best_policy(mdp, vi)
        print(b)
        
        print('done')
        
        #{(3, 2): 1.0, (3, 1): -1.0,
        # (3, 0): 0.12958868267972745, (0, 1): 0.39810203830605462,
        # (0, 2): 0.50928545646220924, (1, 0): 0.25348746162470537,
        # (0, 0): 0.29543540628363629, (1, 2): 0.64958064617168676,
        # (2, 0): 0.34461306281476806, (2, 1): 0.48643676237737926,
        # (2, 2): 0.79536093684710951}
        

def main():
    p = part1()
    p.run()

if __name__== '__main__':
    main()
    
