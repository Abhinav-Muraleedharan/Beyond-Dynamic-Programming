import numpy as np
import random as rand 


class BusEngineEnvironment():
    def __init__(self,x,p,q):
        self.state = x
        self.p = p 
        self.q = q
        self.cost_fun = lambda x: -2*x


    def reset(self):
        self.state = 0

    def set_state(self,X):
        self.state = X

    def step(self, action):
        # new state - 

        # compute cost / reward

        # return that. 
        if action == 1:
            next_state = 0
            utility = self.cost_fun((1-action)*self.state) - action*100
            return next_state, utility
        else:
            u = np.random.uniform()
            if u < self.p:
            # Sample from [0, 5000)
                delta_x =  np.random.uniform(0, 5000)
            elif self.p < u < self.p + self.q:
            # Sample from [5000, 10000)
                delta_x =  np.random.uniform(5000, 10000)
            else:
            # Sample from [10000, ∞)
                delta_x =  np.random.uniform(10000, 100000000)  # Use a large upper bound for practical purposes

        next_state = self.state + delta_x
        self.state = next_state 
        reward = 0 
        utility = self.cost_fun((1-action)*self.state) - action*100

        return next_state, utility



if __name__ == '__main__':
    b = BusEngineEnvironment(x=2,p=0.1,q=0.3)
    for i in range(10):
        r = np.random.choice([0, 1], p=[0.5, 0.5])
        print(r)
        next_state, utility = b.step(r)
        print(next_state)
        print("Utility:",utility)