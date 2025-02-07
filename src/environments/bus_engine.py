import time 
import numpy as np
import random as rand 
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import integrate

class ActionSpace():
    def __init__(self):
        self.n = 2
        self.actions = [0,1]

class BusEngineEnvironment():
    def __init__(self,x,p,q):
        self.state = x
        self.p = p 
        self.q = q
        self.cost_fun = lambda x: -2*x
        self.action_space = ActionSpace()


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
            done = True
            terminated = True
            self.state = next_state
            return next_state, utility, done, terminated
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
        done = False
        terminated = False
        return next_state, utility, done, terminated



def value_iteration_2(env, gamma=0.95, epsilon=1e-6, max_iterations=100):
    # Initialize value function
    max_state = 100001
    V = np.zeros(max_state)  # Assuming max state is 100000
    
    for iteration in range(max_iterations):
        delta = 0
        for state in range(max_state):
            v = V[state]
            
            # Action 0: Keep running
            def integrand0(x):
                next_state = min(state + x, 100000)
                return (-2 * next_state + gamma * V[int(next_state)]) * \
                       (env.p * (x < 5000) / 5000 + 
                        env.q * (5000 <= x < 10000) / 5000 + 
                        (1 - env.p - env.q) * (x >= 10000) / 90000000)
            
            Q0 = integrate.quad(integrand0, 0, 100000000)[0]
            
            # Action 1: Replace engine
            Q1 = -100 + gamma * V[0]
            
            # Update value function
            V[state] = max(Q0, Q1)
            
            delta = max(delta, abs(v - V[state]))
        
        if delta < epsilon:
            break

    print("Finished Value Iteration!")
    # Compute optimal policy
    policy = np.zeros(max_state, dtype=int)
    for state in range(max_state):
        def integrand(x):
            next_state = min(state + x, 100000)
            return (-2 * next_state + gamma * V[int(next_state)]) * \
                   (env.p * (x < 5000) / 5000 + 
                    env.q * (5000 <= x < 10000) / 5000 + 
                    (1 - env.p - env.q) * (x >= 10000) / 90000000)
        
        Q0 = integrate.quad(integrand, 0, 100000000)[0]
        Q1 = -100 + gamma * V[0]
        
        policy[state] = 0 if Q0 > Q1 else 1
    
    return V, policy



def plot_and_save_results(V, policy, max_state=100000, filename_prefix="bus_engine"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot Value Function
    ax1.plot(range(max_state + 1), V)
    ax1.set_title('Optimal Value Function')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Plot Optimal Policy
    ax2.step(range(max_state + 1), policy, where='post')
    ax2.set_title('Optimal Policy (0: Keep Running, 1: Replace)')
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Keep Running', 'Replace'])
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the plot
    filename = f"{filename_prefix}_plot_{max_state}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")



def value_iteration(env: BusEngineEnvironment, gamma: float = 0.99, epsilon: float = 1e-6, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    # Discretize the state space
    state_space = np.linspace(0, 100000, 100)
    
    # Initialize value function
    V = np.zeros_like(state_space)
    
    for _ in range(max_iterations):
        delta = 0
        for i, state in enumerate(state_space):
            env.set_state(state)
            # Compute Q-values for both actions
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(100):  # Monte Carlo sampling
                    next_state, utility = env.step(action)
                    next_state_idx = np.abs(state_space - next_state).argmin()
                    q_value += utility + gamma * V[next_state_idx]
                q_values.append(q_value / 100)
            
            # Update value function
            best_q = max(q_values)
            delta = max(delta, abs(V[i] - best_q))
            V[i] = best_q
        
        if delta < epsilon:
            break
    print("Finished Value Iteration!")
    # Compute optimal policy
    policy = np.zeros_like(state_space, dtype=int)
    for i, state in enumerate(state_space):
        env.set_state(state)
        q_values = []
        for action in [0, 1]:
            q_value = 0
            for _ in range(100):  # Monte Carlo sampling
                next_state, utility = env.step(action)
                next_state_idx = np.abs(state_space - next_state).argmin()
                q_value += utility + gamma * V[next_state_idx]
            q_values.append(q_value / 100)
        policy[i] = np.argmax(q_values)
    
    return V, policy

if __name__ == '__main__':
    
    b = BusEngineEnvironment(x=2,p=0.1,q=0.3)
    print("Computing Solution via Value Iteration - ")
    start_time = time.time()
    V, policy = value_iteration_2(b)
    print("Finished Value Iteration")
    end_time = time.time()
    print("time taken to complete computing solution:",end_time-start_time)
    replacement_threshold = np.argmax(policy)
    print(f"Replacement threshold: {replacement_threshold}")

    # Plot and save the full results
    plot_and_save_results(V, policy, max_state=100000, filename_prefix="bus_engine_full")

    for i in range(5):
        r = np.random.choice([0, 1], p=[0.5, 0.5])
        print(r)
        next_state, utility = b.step(r)
        print(next_state)
        print("Utility:",utility)