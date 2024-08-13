# src/benchmarks/value_iteration.py

import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    V = np.zeros(env.observation_space.n)
    
    for i in range(max_iterations):
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            V[s] = max([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                        for a in range(env.action_space.n)])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        policy[s] = np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                               for a in range(env.action_space.n)])
    
    return V, policy

# You can add a function to run the value iteration algorithm
def run_value_iteration(env, gamma=0.99):
    V, policy = value_iteration(env, gamma)
    return V, policy