# experiments/run_cartpole.py

import gym
import numpy as np
from src.score_life_programming.exact_methods import compute_faber_schauder_coefficients, compute_optimal_l
from src.score_life_programming.approximate_methods import compute_Q
from src.benchmarks.value_iteration import run_value_iteration

def run_exact_method(env, gamma, N, j_max):
    observation, _ = env.reset()
    total_reward = 0
    
    for _ in range(500):  # Run for 500 steps
        a_0, a_1, coefficients = compute_faber_schauder_coefficients(observation, gamma, N, j_max, env)
        l_optimal, _ = compute_optimal_l(a_0, a_1, coefficients)
        
        # Extract action from l_optimal
        action = int(l_optimal * env.action_space.n)
        
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def run_approximate_method(env, gamma, N, n):
    observation, _ = env.reset()
    total_reward = 0
    
    for _ in range(500):  # Run for 500 steps
        Q = compute_Q(observation, env, n, N, gamma)
        action = np.argmin(Q)
        
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def run_value_iteration_method(env, gamma):
    V, policy = run_value_iteration(env, gamma)
    
    observation, _ = env.reset()
    total_reward = 0
    
    for _ in range(500):  # Run for 500 steps
        action = policy[observation]
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def main():
    env = gym.make('CartPole-v1')
    gamma = 0.99
    N = 100
    j_max = 10
    n = 100
    
    exact_reward = run_exact_method(env, gamma, N, j_max)
    approximate_reward = run_approximate_method(env, gamma, N, n)
    value_iteration_reward = run_value_iteration_method(env, gamma)
    
    print(f"Exact Method Total Reward: {exact_reward}")
    print(f"Approximate Method Total Reward: {approximate_reward}")
    print(f"Value Iteration Total Reward: {value_iteration_reward}")

if __name__ == "__main__":
    main()