# experiments/run_experiments.py

import gym
import numpy as np
from src.score_life_programming.exact_methods import run_exact_method
from src.score_life_programming.approximate_methods import run_approximate_method
from src.benchmarks.sb3_benchmarks import run_benchmarks
from src.environments.mountain_car import MountainCarEnv
from src.environments.acrobot import AcrobotEnv
from src.environments.lunar_lander import LunarLanderEnv

def run_all_experiments():
    environments = {
        'CartPole-v1': gym.make('CartPole-v1'),
        'MountainCar-v0': MountainCarEnv(),
        'Acrobot-v1': AcrobotEnv(),
        'LunarLander-v2': LunarLanderEnv()
    }

    results = {}

    for env_name, env in environments.items():
        print(f"Running experiments for {env_name}")
        
        # Run Stable Baselines 3 benchmarks
        sb3_results = run_benchmarks(env)
        
        # Run exact method
        exact_reward = run_exact_method(env, gamma=0.99, N=100, j_max=10)
        
        # Run approximate method
        approx_reward = run_approximate_method(env, gamma=0.99, N=100, n=100)
        
        results[env_name] = {
            'SB3': sb3_results,
            'Exact': exact_reward,
            'Approximate': approx_reward
        }

        env.close()

    return results

if __name__ == "__main__":
    results = run_all_experiments()
    
    # Here you would typically save the results to a file
    # and possibly call a function to visualize the results
    print(results)