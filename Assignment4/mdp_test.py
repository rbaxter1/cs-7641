from mdp import *

class part1():
    def __init__(self):
        self.out_dir = 'output_part1'
        self.time_filename = './' + self.out_dir + '/time.txt'
    
    def run(self):
        print('Running part 1')
        
        mdp = GridMDP([[-0.01, -0.01, -0.01, +1.00],
                       [-0.01, None,  -0.01, -0.01],
                       [-0.01, -0.01, -0.01, -0.01]],
                      terminals=[(3, 1)])

        mdp = GridMDP([[-0.01, -0.01, -0.01, +1.00]],
                      terminals=[(3, 1)])

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
    
