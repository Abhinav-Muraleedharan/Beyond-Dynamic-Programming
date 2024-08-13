# experiments/compare_all_methods.py

import gym
import numpy as np
from src.score_life_programming.exact_methods import compute_faber_schauder_coefficients, compute_optimal_l
from src.score_life_programming.approximate_methods import compute_Q
from src.benchmarks.value_iteration import run_value_iteration
from src.benchmarks.policy_iteration import run_policy_iteration
from src.benchmarks.dqn import run_dqn
from src.benchmarks.ppo import run_ppo
from src.environments.mountain_car import MountainCarEnv
from src.environments.acrobot import AcrobotEnv
from src.environments.lunar_lander import LunarLanderEnv

def run_experiment(env, method, **kwargs):
    if method == 'exact':
        return run_exact_method(env, **kwargs)
    elif method == 'approximate':
        return run_approximate_method(env, **kwargs)
    elif method == 'value_iteration':
        return run_value_iteration_method(env, **kwargs)
    elif method == 'policy_iteration':
        return run_policy_iteration_method(env, **kwargs)
    elif method == 'dqn':
        return run_dqn_method(env, **kwargs)
    elif method == 'ppo':
        return run_ppo_method(env, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def run_exact_method(env, gamma, N, j_max):
    observation, _ = env.reset()
    total_reward = 0
    
    for _ in range(1000):  # Run for 1000 steps
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
    
    for _ in range(1000):  # Run for 1000 steps
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
    
    for _ in range(1000):  # Run for 1000 steps
        action = policy[env.discretize_state(observation)]
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def run_policy_iteration_method(env, gamma):
    policy = run_policy_iteration(env, gamma)
    
    observation, _ = env.reset()
    total_reward = 0
    
    for _ in range(1000):