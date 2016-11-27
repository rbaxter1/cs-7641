import matplotlib
matplotlib.use('Agg')

#from mdp import *
from sklearn.preprocessing import normalize

import mdptoolbox, mdptoolbox.example
import numpy as np
import pandas as pd
import random as rand
import math
from plot_helper import *

from QLearner import QLearningEx
from QLearning import QLearning

from timeit import default_timer as time

class part1():
    def __init__(self):
        self.out_dir = 'output'
        
        ## space definitions
        self.space_free = 0
        self.space_start_free = 1
        self.space_goal = 2
        self.space_void = 3
        self.space_quicksand = 4
        self.space_bonus = 5
        
        self.rewards = [-1, -1, +1, 0, -100, +1000000]
        
        ## action definitions
        self.action_up = 0
        self.action_down = 1
        self.action_left = 2
        self.action_right = 3
        self.action_total_num = 5
        
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        ## random grid config
        self.max_void_coverage = 0.25
        self.max_quicksand_coverage = 0.25
        self.move_success_pct = 0.9
    
    def __test_movement(self):
        grid = pd.read_csv('./input/grid6.csv', header=None).values
        
        newpos = tuple(map(sum, zip((0,0), self.actions[self.action_up])))
        
        
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
        goal_indices = np.where(np.logical_or(flat_grid == self.space_goal, flat_grid == self.space_bonus))
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
    
    def __get_stochastic_action(self, a, success_pct=0.9, directional=True):
        # 0 = up
        # 1 = down
        # 2 = left
        # 3 = right
        
        if success_pct == 1.:
            other = 0.
        elif directional:
            other = (1. - success_pct) / 2.
        else:
            other = (1. - success_pct) / 3.
        
        if a == 0:
            if directional:
                return [success_pct, 0.00, other, other]
            else:
                return [success_pct, other, other, other]
        if a == 1:
            if directional:
                return [0.00, success_pct, other, other]
            else:
                return [other, success_pct, other, other]
        if a == 2:
            if directional:
                return [other, other, success_pct, 0.00]
            else:
                return [other, other, success_pct, other]
        if a == 3:
            if directional:
                return [other, other, 0.00, success_pct]
            else:
                return [other, other, other, success_pct]
            
    def __convert_grid_to_mdp(self, grid, action_success_pct=1.0, directional=True):
        rows, cols = grid.shape
        num_states = rows * cols
        
        ## alpha and omega
        start, goals = self.__get_grid_terminals(grid)
        ## transitions (A, S, S)
        T = np.zeros((len(self.actions), num_states, num_states))
        ## rewards (A, S, S)
        R = np.zeros((len(self.actions), num_states, num_states))
        
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
                    t[pos] = 1.
                else:            
                    a = i
                    # try move
                    stochastic_action = self.__get_stochastic_action(a, action_success_pct, directional)
                    assert np.sum(stochastic_action) == 1., 'not stochastic'

                    for f in range(len(stochastic_action)):
                        newpos = self.test_move(pos, grid, f)
                        if newpos != None:
                            t[newpos] += stochastic_action[f]
                        else:
                            t[pos] += stochastic_action[f] # can't move
                        
                # reshape
                t.shape = num_states
                T[i, s] = t
               
        return T, R, start, goals
    
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
        
    def run_value_iteration_and_plot(self, grid, k, d, discount, epsilon=0.001):
        ## policy iteration
        T, R, start, goals = self.__convert_grid_to_mdp(grid, k, d)
        
        
        vi = mdptoolbox.mdp.ValueIteration(T, R, discount, epsilon=epsilon, max_iter=1000)
        
        with open('./output/valueiter.txt', 'a') as text_file:            
            t0 = time()
            vi.run()
            text_file.write('ValueIteration: %0.3f seconds. Iters: %i\n' % (time() - t0, vi.iter))
        
        p = np.array(vi.policy)
        p.shape = grid.shape
        p = p
        
        v = np.array(vi.V)
        v.shape = grid.shape
        v = v
        if d:
            d_str = 'dir'
        else:
            d_str = 'non-dir'
        
        ph = plot_helper()
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Grid r: ' + str(k) + '(' + d_str + '), discount: ', str(discount)
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'valueiter_' + str(k) + '_' + d_str + '_' + str(discount) + '.png' '.png'
        ph.plot_results2(v, grid, p, title, fn)
        
        #ph.plot_heatmap(v, grid, p, title, fn)
        #ph.plot_heatmap_simple(v[::-1], title, fn)
        print('done')
        
    def run_policy_iteration_and_plot(self, grid, k, d, discount, epsilon=0.001):
        ## policy iteration
        T, R, start, goals = self.__convert_grid_to_mdp(grid, k, d)
        #pi = mdptoolbox.mdp.PolicyIteration(T, R, discount, max_iter=1000000)
        pi = mdptoolbox.mdp.PolicyIterationModified(T, R, discount, epsilon=epsilon, max_iter=1000000)
        
        
        
        with open('./output/policyiter.txt', 'a') as text_file:            
            t0 = time()
            pi.run()
            text_file.write('PolicyIteration: %0.3f seconds. Iters: %i\n' % (time() - t0, pi.iter))
            
        
        p = np.array(pi.policy)
        p.shape = grid.shape
        p = p
        
        v = np.array(pi.V)
        v.shape = grid.shape
        v = v
        if d:
            d_str = 'dir'
        else:
            d_str = 'non-dir'
            
        ph = plot_helper()
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Grid r: ' + str(k) + '(' + d_str + '), discount: ', str(discount)
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'policyiter_' + str(k) + '_' + d_str + '_' + str(discount) + '.png'
        #ph.plot_heatmap(v, grid, p, title, fn)
        ph.plot_results2(v, grid, p, title, fn)
        
        #ph.plot_heatmap_simple(v[::-1], title, fn)
        print('done')

        
        
    def run_and_plot_qlearner(self, grid, d, k, alpha, gamma, rar, rard=0.99, n_restarts=1000, n_iter=100000):
        T, R, start, goals = self.__convert_grid_to_mdp(grid, k, d)
                
        q = QLearningEx(T, R, grid, start, goals, n_restarts=n_restarts, alpha = alpha, gamma = gamma, rar = rar, radr = rard, n_iter=n_iter)
        #q = mdptoolbox.mdp.QLearning(T, R, 0.9)
        q.run()
        print(q.Q)
        
        p = np.array(q.policy)
        p.shape = grid.shape
        p = p
        
        v = np.array(q.V)
        v.shape = grid.shape
        v = v
        if d:
            d_str = 'dir'
        else:
            d_str = 'non-dir'
            
        ph = plot_helper()
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Tracker\na: ' + str(q.alpha) + ', g: ' + str(q.gamma) + ', d: ' + str(q.orig_rar) + '@' + str(q.radr) + ', r: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'tracker_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '_' + str(k) + '_' + d_str + '.png'
        #tracker = normalize(q.tracker[::-1], axis=1, norm='l1')
        ph.plot_heatmap_simple(q.tracker[::-1], title, fn)
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Iterations\na: ' + str(q.alpha) + ', g: ' + str(q.gamma) + ', d: ' + str(q.orig_rar) + '@' + str(q.radr) + ', r: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'iterations_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '_' + str(k) + '_' + d_str + '.png'
        ph.plot_series(range(len(q.episode_iterations)),
                    [q.episode_iterations],
                    [None],
                    ['iterations'],
                    #cm.viridis(np.linspace(0, 1, 1)),
                    ['black'],
                    [''],
                    title,
                    'Episodes',
                    'Iterations',
                    fn)
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Rewards\na: ' + str(q.alpha) + ', g: ' + str(q.gamma) + ', d: ' + str(q.orig_rar) + '@' + str(q.radr) + ', r: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'rewards_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '_' + str(k) + '_' + d_str + '.png'
        ph.plot_series(range(len(q.episode_reward)),
                    [q.episode_reward],
                    [None],
                    ['rewards'],
                    #cm.viridis(np.linspace(0, 1, 1)),
                    ['black'],
                    [''],
                    title,
                    'Episodes',
                    'Rewards',
                    fn)
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Timing\na: ' + str(q.alpha) + ', g: ' + str(q.gamma) + ', d: ' + str(q.orig_rar) + '@' + str(q.radr) + ', r: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'timing_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '_' + str(k) + '_' + d_str + '.png'
        ph.plot_series(range(len(q.episode_times)),
                    [q.episode_times],
                    [None],
                    ['seconds'],
                    #cm.viridis(np.linspace(0, 1, 1)),
                    ['black'],
                    [''],
                    title,
                    'Iterations',
                    'Time in seconds',
                    fn)
        
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Grid\na: ' + str(q.alpha) + ', g: ' + str(q.gamma) + ', d: ' + str(q.orig_rar) + '@' + str(q.radr) + ', r: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'qlearn_' + str(q.alpha) + '_' + str(q.gamma) + '_' + str(q.orig_rar) + '_' + str(q.radr) + '_' + str(k) + '_' + d_str + '.png'
        #ph.plot_heatmap(v, grid[::-1], p, title, fn)
        ph.plot_results2(v, grid, p, title, fn)
        
        '''
        title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Grid\nr: ' + str(k) + '(' + d_str + ')'
        fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + 'qlearn_' + str(k) + '_' + d_str + '.png'
        ph.plot_results2(v, grid, p, title, fn)
        '''
        
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
        
        #self.__test_movement()
        
        for grid_file in ['./input/grid1.csv', './input/grid2.csv']:
        #for grid_file in ['./input/grid2.csv']:
            
            #fn = './input/grid1.csv'
            grid = pd.read_csv(grid_file, header=None).values
            ph = plot_helper()
            
            title = str(grid.shape[0]) + 'x' + str(grid.shape[1]) + ' Grid Layout'
            fn = './output/' + str(grid.shape[0]) + 'x' + str(grid.shape[1]) + '_layout.png'        
            ph.plot_layout(grid, title, fn)
            
            self.run_and_plot_qlearner(grid, d=True, k=1.0, alpha=0.2, gamma=0.8, rar=0.99, rard=0.99999, n_restarts=10000, n_iter=1000000)
            #self.run_value_iteration_and_plot(grid, k=1.0, d=True, discount=0.9, epsilon=0.00001)
            #self.run_policy_iteration_and_plot(grid, k=1.0, d=True, discount=0.9, epsilon=0.00001)
            
            '''
            for k in [1.00, 0.90, 0.85, 0.80, 0.75]:
                for d in [False, True]:
                    for discount in [0.9, 0.8, 0.7, 0.6]:
                        self.run_value_iteration_and_plot(grid, k=k, d=d, discount=discount)
            
            for k in [1.00, 0.90, 0.85, 0.80, 0.75]:
                for d in [False, True]:
                    for discount in [0.9, 0.8, 0.7, 0.6]:
                        self.run_policy_iteration_and_plot(grid, k=k, d=d, discount=discount)
            '''
            '''
            for k in [1.00, 0.90, 0.85, 0.80, 0.75]:
                for d in [False, True]:
                    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        for gamma in [1.0, 0.8, 0.6, 0.4, 0.2]:
                            for rard in [0.99, 0.9999, 0.999999]:
                                self.run_and_plot_qlearner(grid, d, k, alpha, gamma, rar=0.99, rard=rard)
            '''
            print('done qlearner')
            
            
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
        
        fn = 'valueiter_' + str(pi.discount) + '.png'
        title = '10x10 Quicksand Grid Value Iteration\ndiscount: ' + str(pi.discount)
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



        '''


def main():
    p = part1()
    p.run()

if __name__== '__main__':
    main()
    
