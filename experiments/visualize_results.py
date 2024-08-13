# experiments/visualize_results.py

import matplotlib.pyplot as plt
import numpy as np

def visualize_results(results):
    for env_name, env_results in results.items():
        plt.figure(figsize=(12, 6))
        
        # Plot SB3 results
        sb3_results = env_results['SB3']
        algorithms = list(sb3_results.keys())
        mean_rewards = [r['mean_reward'] for r in sb3_results.values() if 'mean_reward' in r]
        std_rewards = [r['std_reward'] for r in sb3_results.values() if 'std_reward' in r]
        
        x = np.arange(len(algorithms))
        plt.bar(x, mean_rewards, yerr=std_rewards, align='center', alpha=0.5, ecolor='black', capsize=10)
        
        # Plot exact and approximate method results
        if env_results['Exact'] is not None:
            plt.axhline(y=env_results['Exact'], color='r', linestyle='-', label='Exact Method')
        if env_results['Approximate'] is not None:
            plt.axhline(y=env_results['Approximate'], color='g', linestyle='--', label='Approximate Method')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Mean Reward')
        plt.title(f'Algorithm Performance Comparison - {env_name}')
        plt.xticks(x, algorithms)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{env_name}_comparison.png')
        plt.close()

if __name__ == "__main__":
    import json
    with open('results/experiment_results.json', 'r') as f:
        results = json.load(f)
    visualize_results(results)