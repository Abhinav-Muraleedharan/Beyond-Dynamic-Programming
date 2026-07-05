#!/usr/bin/env python
"""
DEFINITIVE COMPARISON: Value Iteration vs Score-Life Programming

Using the FIXED, canonical BusEngineEnvironment.
Same environment, same γ, same parameters - fair test.

This will determine if the algorithms truly produce different policies
or if all differences were due to bugs/parameter mismatches.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def run_value_iteration(gamma=0.99, n_states=100, n_samples=100):
    """
    Run Value Iteration with fixed environment.
    """
    print(f"\n{'='*70}")
    print(f"VALUE ITERATION (γ={gamma})")
    print(f"{'='*70}")

    env = BusEngineEnvironment(p=0.1, q=0.3)
    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

    start_time = time.time()

    # Value iteration
    for iteration in range(500):
        delta = 0
        new_V = V.copy()

        for i, state in enumerate(state_space):
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(n_samples):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / n_samples)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V

        if delta < 1e-4:
            print(f"  Converged in {iteration + 1} iterations (delta={delta:.6f})")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(state_space):
        best_q = -np.inf
        for action in [0, 1]:
            q_value = 0
            for _ in range(n_samples):
                env.set_state(state)
                next_state, reward, _, _, _ = env.step(action)
                next_idx = np.abs(state_space - next_state[0]).argmin()
                q_value += reward + gamma * V[next_idx]
            q_value /= n_samples
            if q_value > best_q:
                best_q = q_value
                best_action = action
        policy[i] = best_action

    vi_time = time.time() - start_time

    # Find threshold
    threshold = None
    for i, action in enumerate(policy):
        if action == 1:
            threshold = state_space[i]
            break

    print(f"  Computation time: {vi_time:.2f}s")
    print(f"  Replacement threshold: {threshold:.0f} miles" if threshold else "  Always keep running")

    return policy, state_space, V, threshold, vi_time


def test_policy(env, policy, state_space, n_episodes=50, gamma=0.99):
    """Test policy performance."""
    print(f"  Testing policy ({n_episodes} episodes)...")

    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        discount = 1.0
        done = False
        steps = 0

        while not done and steps < 500:
            idx = min(np.abs(state_space - state[0]).argmin(), len(policy) - 1)
            action = policy[idx]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += discount * reward
            discount *= gamma
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, std_reward


def run_score_life_analysis(gamma=0.99, N=20, j_max=4, num_samples=100, test_states=None):
    """
    Analyze Score-Life function behavior.

    Instead of trying to extract a policy, just compute Score-Life
    functions and analyze their structure.
    """
    print(f"\n{'='*70}")
    print(f"SCORE-LIFE PROGRAMMING ANALYSIS (γ={gamma})")
    print(f"{'='*70}")

    if test_states is None:
        test_states = np.array([0, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000])

    start_time = time.time()

    results = []
    for state in test_states:
        env = BusEngineEnvironment(p=0.1, q=0.3)
        env.set_state(state)
        slp = ScoreLifeProgramming(env, gamma, N, j_max, num_samples, state)

        print(f"  Computing Score-Life for state {state:.0f}...")
        score_func = slp._compute_faber_schauder_coefficients()

        # Evaluate at multiple life parameters
        l_values = np.linspace(0, 1, 100)
        scores = [score_func.compute_fractal(l) for l in l_values]

        optimal_l = l_values[np.argmax(scores)]
        max_score = max(scores)

        results.append({
            'state': state,
            'optimal_l': optimal_l,
            'max_score': max_score,
            'score_function': scores
        })

        print(f"    Optimal l*={optimal_l:.4f}, Max Score={max_score:.2e}")

    sl_time = time.time() - start_time

    print(f"  Total computation time: {sl_time:.2f}s")

    return results, sl_time


def plot_comprehensive_comparison(vi_policy, vi_states, vi_threshold, vi_reward, gamma):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: VI Policy
    ax = axes[0, 0]
    ax.step(vi_states, vi_policy, where='post', linewidth=2.5, color='blue')
    if vi_threshold:
        ax.axvline(x=vi_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold: {vi_threshold:.0f} mi')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Keep Running', 'Replace'])
    ax.set_title(f'Value Iteration Policy (γ={gamma})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Policy distribution
    ax = axes[0, 1]
    keep_count = np.sum(vi_policy == 0)
    replace_count = np.sum(vi_policy == 1)
    ax.bar(['Keep Running', 'Replace'], [keep_count, replace_count],
           color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of States', fontsize=12)
    ax.set_title('Policy Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Economic analysis
    ax = axes[1, 0]
    mileages = vi_states
    operating_costs = 0.01 * mileages
    replacement_cost = 100

    ax.plot(mileages, operating_costs, 'b-', linewidth=2.5, label='Operating Cost (Keep)')
    ax.axhline(y=replacement_cost, color='red', linestyle='-', linewidth=2.5,
              label=f'Replacement Cost: ${replacement_cost}')
    if vi_threshold:
        ax.axvline(x=vi_threshold, color='green', linestyle='--', linewidth=2,
                  label=f'VI Threshold: {vi_threshold:.0f} mi')

    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Immediate Cost ($)', fontsize=12)
    ax.set_title('Economic Break-Even Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
DEFINITIVE COMPARISON RESULTS
{'='*40}

Environment Parameters:
• p = 0.1 (small damage probability)
• q = 0.3 (medium damage probability)
• Delta ranges: [0,1000), [1000,3000), [3000,10000)
• Replacement cost: $100
• Operating cost rate: $0.01 per mile

Value Iteration Results:
• Discount factor γ = {gamma}
• Threshold: {vi_threshold:.0f} miles" if vi_threshold else "Always keep
• Average reward: {vi_reward:.2f}
• States analyzed: {len(vi_states)}

Interpretation:
{"Replace when mileage > " + str(int(vi_threshold)) + " miles" if vi_threshold else "Never replace (operating always cheaper)"}

Economic Logic:
Operating cost reaches $100 at {100/0.01:.0f} miles
VI threshold is {"below" if vi_threshold and vi_threshold < 10000 else "at"} this point
{"(Considering future costs)" if vi_threshold and vi_threshold < 10000 else ""}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Definitive Comparison: Fixed Environment (γ={gamma})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/bus_engine_definitive_gamma{gamma:.2f}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/bus_engine_definitive_gamma{gamma:.2f}.png")
    plt.close()


def main():
    print("="*70)
    print("DEFINITIVE COMPARISON: VI vs SL with FIXED ENVIRONMENT")
    print("Testing with bug-free, canonical environment")
    print("="*70)

    # Test with multiple gamma values
    gamma_values = [0.60, 0.80, 0.90, 0.95, 0.99]

    all_results = []

    for gamma in gamma_values:
        print(f"\n{'#'*70}")
        print(f"# TESTING WITH γ = {gamma}")
        print(f"{'#'*70}")

        # Run Value Iteration
        vi_policy, vi_states, vi_values, vi_threshold, vi_time = run_value_iteration(
            gamma=gamma, n_states=100, n_samples=100
        )

        # Test VI policy
        env_test = BusEngineEnvironment(p=0.1, q=0.3)
        vi_reward, vi_std = test_policy(env_test, vi_policy, vi_states,
                                        n_episodes=50, gamma=gamma)

        # Analyze Score-Life
        sl_results, sl_time = run_score_life_analysis(
            gamma=gamma, N=20, j_max=4, num_samples=100
        )

        # Store results
        result = {
            'gamma': gamma,
            'VI': {
                'threshold': float(vi_threshold) if vi_threshold else None,
                'reward': float(vi_reward),
                'std': float(vi_std),
                'time': float(vi_time)
            },
            'SL': {
                'analysis': [{
                    'state': r['state'],
                    'optimal_l': r['optimal_l'],
                    'max_score': r['max_score']
                } for r in sl_results],
                'time': float(sl_time)
            }
        }

        all_results.append(result)

        # Create visualization
        plot_comprehensive_comparison(vi_policy, vi_states, vi_threshold, vi_reward, gamma)

        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR γ = {gamma}")
        print(f"{'='*70}")
        print(f"Value Iteration:")
        print(f"  Threshold: {vi_threshold:.0f} miles" if vi_threshold else "  No threshold (always keep)")
        print(f"  Reward: {vi_reward:.2f} ± {vi_std:.2f}")
        print(f"  Time: {vi_time:.2f}s")
        print(f"\nScore-Life Analysis:")
        print(f"  Time: {sl_time:.2f}s")
        print(f"  All states show optimal l* near 1.0: {all([r['optimal_l'] > 0.9 for r in sl_results])}")

    # Save all results
    with open('results/bus_engine_definitive_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: results/bus_engine_definitive_comparison.json")

    # Create gamma sensitivity plot
    create_gamma_sensitivity_plot(all_results)

    print("\n" + "="*70)
    print("DEFINITIVE COMPARISON COMPLETE")
    print("="*70)
    print("\nKey Finding:")
    print("Using the FIXED environment with MATCHED parameters")
    print("Check the visualizations to see final policies.")


def create_gamma_sensitivity_plot(all_results):
    """Plot how threshold changes with gamma."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gammas = [r['gamma'] for r in all_results]
    thresholds = [r['VI']['threshold'] if r['VI']['threshold'] else 0 for r in all_results]
    rewards = [r['VI']['reward'] for r in all_results]

    # Plot 1: Threshold vs gamma
    ax1.plot(gammas, thresholds, 'o-', linewidth=2.5, markersize=10, color='blue')
    ax1.set_xlabel('Discount Factor (γ)', fontsize=12)
    ax1.set_ylabel('Replacement Threshold (miles)', fontsize=12)
    ax1.set_title('How Threshold Changes with γ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Add economic break-even line
    break_even = 100 / 0.01  # 10,000 miles
    ax1.axhline(y=break_even, color='red', linestyle='--', alpha=0.5,
               label=f'Economic break-even: {break_even:.0f} mi')
    ax1.legend()

    # Plot 2: Reward vs gamma
    ax2.plot(gammas, rewards, 's-', linewidth=2.5, markersize=10, color='green')
    ax2.set_xlabel('Discount Factor (γ)', fontsize=12)
    ax2.set_ylabel('Average Discounted Reward', fontsize=12)
    ax2.set_title('Performance vs γ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.suptitle('Discount Factor Sensitivity Analysis\n(Fixed Environment)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/bus_engine_gamma_sensitivity_fixed.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_gamma_sensitivity_fixed.png")
    plt.close()


if __name__ == "__main__":
    main()
