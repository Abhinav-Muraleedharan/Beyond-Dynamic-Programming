#!/usr/bin/env python
"""
Fair Comparison: Value Iteration vs Score-Life Programming
WITH THE SAME DISCOUNT FACTOR

This is the definitive test to see if policy differences are due to:
1. Different discount factors (γ mismatch)
2. OR genuinely different algorithms finding different optima

We test with γ = 0.99 for BOTH methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def run_value_iteration(env, gamma=0.99, n_states=100, n_samples=50, max_iters=200):
    """Run Value Iteration with specified gamma."""
    print(f"\n{'='*70}")
    print(f"Running Value Iteration (γ={gamma})")
    print(f"{'='*70}")

    start_time = time.time()

    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

    # Value iteration
    for iteration in range(max_iters):
        delta = 0
        new_V = V.copy()

        for i, state in enumerate(state_space):
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(n_samples):
                    env.set_state(state)
                    result = env.step(action)
                    if len(result) == 5:
                        next_state, reward, _, _, _ = result
                    else:
                        next_state, reward, _, _ = result
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / n_samples)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V

        if delta < 1e-4:
            print(f"  Converged in {iteration + 1} iterations")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(state_space):
        best_q = -np.inf
        for action in [0, 1]:
            q_value = 0
            for _ in range(n_samples):
                env.set_state(state)
                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, _, _, _ = result
                else:
                    next_state, reward, _, _ = result
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
    print(f"  Replacement threshold: {threshold:.0f} miles" if threshold else "  No threshold found")

    return policy, state_space, V, threshold, vi_time


def run_score_life(env, gamma=0.99, N=20, j_max=4, num_samples=100, test_states=None):
    """Run Score-Life Programming with specified gamma."""
    print(f"\n{'='*70}")
    print(f"Running Score-Life Programming (γ={gamma})")
    print(f"{'='*70}")

    start_time = time.time()

    if test_states is None:
        test_states = np.linspace(0, 50000, 100)

    sl_policy = np.zeros(len(test_states), dtype=int)

    # For each state, compute Score-Life and determine action
    for i, state in enumerate(test_states):
        if i % 10 == 0:
            print(f"  State {i+1}/{len(test_states)}: {state:.0f} miles")

        env_test = BusEngineEnvironment(x=state, p=0.1, q=0.3)
        slp = ScoreLifeProgramming(env_test, gamma, N, j_max, num_samples, state)

        # Compute Score-Life function
        score_func = slp._compute_faber_schauder_coefficients()

        # Evaluate at multiple life parameters to find optimal action
        l_values = np.linspace(0, 1, 50)
        scores = [score_func.compute_fractal(l) for l in l_values]

        # For now, use a simple heuristic:
        # If max score is very negative, replace. Otherwise keep.
        max_score = max(scores)
        optimal_l = l_values[np.argmax(scores)]

        # Decision heuristic based on score magnitude and optimal l
        # This is simplified - proper method would use life->action mapping
        if max_score < -1e6:  # Very bad score -> replace
            sl_policy[i] = 1
        else:
            sl_policy[i] = 0

    sl_time = time.time() - start_time

    # Find threshold
    threshold = None
    for i, action in enumerate(sl_policy):
        if action == 1:
            threshold = test_states[i]
            break

    print(f"  Computation time: {sl_time:.2f}s")
    print(f"  Replacement threshold: {threshold:.0f} miles" if threshold else "  No threshold found")

    return sl_policy, test_states, threshold, sl_time


def test_policy_performance(env, policy, state_space, n_episodes=20, gamma=0.99):
    """Test a policy and return average discounted reward."""
    print(f"\n  Testing policy performance ({n_episodes} episodes)...")

    episode_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        discount = 1.0
        done = False
        steps = 0

        while not done and steps < 500:
            idx = min(np.abs(state_space - state[0]).argmin(), len(policy) - 1)
            action = policy[idx]

            result = env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
            else:
                state, reward, terminated, truncated = result
            total_reward += discount * reward
            discount *= gamma
            steps += 1

            done = terminated or truncated

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, std_reward


def plot_fair_comparison(vi_policy, vi_states, vi_threshold,
                         sl_policy, sl_states, sl_threshold,
                         vi_reward, sl_reward, gamma):
    """Create comprehensive comparison plot."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Policy comparison
    ax = axes[0, 0]
    ax.step(vi_states, vi_policy, where='post', linewidth=2.5,
            color='blue', label='Value Iteration', alpha=0.7)
    ax.step(sl_states, sl_policy, where='post', linewidth=2.5,
            color='red', label='Score-Life Programming', alpha=0.7, linestyle='--')

    if vi_threshold:
        ax.axvline(x=vi_threshold, color='blue', linestyle=':', linewidth=2,
                  label=f'VI Threshold: {vi_threshold:.0f} mi')
    if sl_threshold:
        ax.axvline(x=sl_threshold, color='red', linestyle=':', linewidth=2,
                  label=f'SL Threshold: {sl_threshold:.0f} mi')

    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Keep Running', 'Replace'])
    ax.set_title(f'Policy Comparison (γ={gamma})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Performance comparison
    ax = axes[0, 1]
    methods = ['Value\nIteration', 'Score-Life\nProgramming']
    rewards = [vi_reward, sl_reward]
    colors = ['blue', 'red']

    bars = ax.bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Average Discounted Reward', fontsize=12)
    ax.set_title(f'Performance Comparison (γ={gamma})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.annotate(f'{reward:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', fontsize=11, fontweight='bold')

    # Add improvement percentage
    if vi_reward != 0:
        improvement = ((sl_reward - vi_reward) / abs(vi_reward)) * 100
        ax.text(0.5, 0.95, f'SL vs VI: {improvement:+.1f}%',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', horizontalalignment='center')

    # Plot 3: Threshold comparison
    ax = axes[1, 0]
    if vi_threshold and sl_threshold:
        thresholds = [vi_threshold, sl_threshold]
        ax.bar(methods, thresholds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Replacement Threshold (miles)', fontsize=12)
        ax.set_title('Threshold Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (method, threshold) in enumerate(zip(methods, thresholds)):
            ax.annotate(f'{threshold:.0f}',
                       xy=(i, threshold),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', fontsize=11, fontweight='bold')

        # Add difference
        diff = sl_threshold - vi_threshold
        diff_pct = (diff / vi_threshold) * 100
        ax.text(0.5, 0.95, f'Difference: {diff:.0f} miles ({diff_pct:+.1f}%)',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
               verticalalignment='top', horizontalalignment='center')

    # Plot 4: Agreement analysis
    ax = axes[1, 1]
    # Interpolate to common grid
    common_states = np.linspace(0, 50000, 200)
    vi_interp = np.zeros(len(common_states), dtype=int)
    sl_interp = np.zeros(len(common_states), dtype=int)

    for i, state in enumerate(common_states):
        vi_idx = np.abs(vi_states - state).argmin()
        sl_idx = np.abs(sl_states - state).argmin()
        vi_interp[i] = vi_policy[vi_idx]
        sl_interp[i] = sl_policy[sl_idx]

    agreement = (vi_interp == sl_interp).astype(int)
    agreement_pct = np.mean(agreement) * 100

    ax.fill_between(common_states, 0, agreement, alpha=0.3, color='green',
                    label='Agreement', step='post')
    ax.fill_between(common_states, 0, 1-agreement, alpha=0.3, color='red',
                    label='Disagreement', step='post')

    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Policy Agreement', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Disagree', 'Agree'])
    ax.set_title(f'Policy Agreement: {agreement_pct:.1f}%',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Fair Comparison: VI vs SL (Both with γ={gamma})\n' +
                 'Testing if policy differences persist with matched discount factor',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(f'results/bus_engine_fair_comparison_gamma{gamma:.2f}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/bus_engine_fair_comparison_gamma{gamma:.2f}.png")
    plt.close()


def main():
    print("=" * 70)
    print("FAIR COMPARISON: VALUE ITERATION vs SCORE-LIFE PROGRAMMING")
    print("Same Discount Factor, Same Environment, Fair Test")
    print("=" * 70)

    # Test with multiple gamma values
    gamma_values = [0.99, 0.95, 0.90, 0.80, 0.60]

    all_results = []

    for gamma in gamma_values:
        print(f"\n{'#'*70}")
        print(f"# TESTING WITH γ = {gamma}")
        print(f"{'#'*70}")

        # Create environments
        env_vi = BusEngineEnvironment(x=0, p=0.1, q=0.3)
        env_sl = BusEngineEnvironment(x=0, p=0.1, q=0.3)
        env_test = BusEngineEnvironment(x=0, p=0.1, q=0.3)

        # Run Value Iteration
        vi_policy, vi_states, vi_values, vi_threshold, vi_time = run_value_iteration(
            env_vi, gamma=gamma, n_states=100, n_samples=50
        )

        # Test VI policy
        vi_reward, vi_std = test_policy_performance(env_test, vi_policy, vi_states,
                                                     n_episodes=20, gamma=gamma)

        # Run Score-Life Programming
        sl_policy, sl_states, sl_threshold, sl_time = run_score_life(
            env_sl, gamma=gamma, N=20, j_max=4, num_samples=100, test_states=vi_states
        )

        # Test SL policy
        sl_reward, sl_std = test_policy_performance(env_test, sl_policy, sl_states,
                                                     n_episodes=20, gamma=gamma)

        # Calculate agreement
        agreement = np.mean(vi_policy == sl_policy) * 100

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
                'threshold': float(sl_threshold) if sl_threshold else None,
                'reward': float(sl_reward),
                'std': float(sl_std),
                'time': float(sl_time)
            },
            'agreement': float(agreement),
            'threshold_diff': float(sl_threshold - vi_threshold) if (vi_threshold and sl_threshold) else None
        }
        all_results.append(result)

        # Create comparison plot
        plot_fair_comparison(vi_policy, vi_states, vi_threshold,
                           sl_policy, sl_states, sl_threshold,
                           vi_reward, sl_reward, gamma)

        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY FOR γ = {gamma}")
        print(f"{'='*70}")
        print(f"Value Iteration:")
        print(f"  Threshold: {vi_threshold:.0f} miles" if vi_threshold else "  No threshold")
        print(f"  Reward: {vi_reward:.2f} ± {vi_std:.2f}")
        print(f"  Time: {vi_time:.2f}s")
        print(f"\nScore-Life Programming:")
        print(f"  Threshold: {sl_threshold:.0f} miles" if sl_threshold else "  No threshold")
        print(f"  Reward: {sl_reward:.2f} ± {sl_std:.2f}")
        print(f"  Time: {sl_time:.2f}s")
        print(f"\nComparison:")
        print(f"  Policy agreement: {agreement:.1f}%")
        if vi_threshold and sl_threshold:
            print(f"  Threshold difference: {sl_threshold - vi_threshold:.0f} miles")
        if vi_reward != 0:
            improvement = ((sl_reward - vi_reward) / abs(vi_reward)) * 100
            print(f"  SL performance vs VI: {improvement:+.1f}%")

    # Save all results
    with open('results/bus_engine_fair_comparison_all_gammas.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: results/bus_engine_fair_comparison_all_gammas.json")

    # Create summary plot across all gammas
    create_gamma_comparison_plot(all_results)

    print("\n" + "=" * 70)
    print("FAIR COMPARISON COMPLETE")
    print("=" * 70)
    print("\nKey Question: Do VI and SL converge to the same policy with same γ?")
    print("Check the visualizations to see if differences persist.")


def create_gamma_comparison_plot(all_results):
    """Create plot showing how results vary with gamma."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    gammas = [r['gamma'] for r in all_results]

    # Plot 1: Thresholds vs gamma
    ax = axes[0, 0]
    vi_thresholds = [r['VI']['threshold'] for r in all_results if r['VI']['threshold']]
    sl_thresholds = [r['SL']['threshold'] for r in all_results if r['SL']['threshold']]
    valid_gammas_vi = [r['gamma'] for r in all_results if r['VI']['threshold']]
    valid_gammas_sl = [r['gamma'] for r in all_results if r['SL']['threshold']]

    if vi_thresholds:
        ax.plot(valid_gammas_vi, vi_thresholds, 'o-', linewidth=2.5, markersize=10,
               color='blue', label='Value Iteration')
    if sl_thresholds:
        ax.plot(valid_gammas_sl, sl_thresholds, 's-', linewidth=2.5, markersize=10,
               color='red', label='Score-Life Programming')

    ax.set_xlabel('Discount Factor (γ)', fontsize=12)
    ax.set_ylabel('Replacement Threshold (miles)', fontsize=12)
    ax.set_title('Threshold vs Discount Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Plot 2: Rewards vs gamma
    ax = axes[0, 1]
    vi_rewards = [r['VI']['reward'] for r in all_results]
    sl_rewards = [r['SL']['reward'] for r in all_results]

    ax.plot(gammas, vi_rewards, 'o-', linewidth=2.5, markersize=10,
           color='blue', label='Value Iteration')
    ax.plot(gammas, sl_rewards, 's-', linewidth=2.5, markersize=10,
           color='red', label='Score-Life Programming')

    ax.set_xlabel('Discount Factor (γ)', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Performance vs Discount Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Plot 3: Policy agreement vs gamma
    ax = axes[1, 0]
    agreements = [r['agreement'] for r in all_results]

    ax.plot(gammas, agreements, 'o-', linewidth=2.5, markersize=10, color='green')
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Discount Factor (γ)', fontsize=12)
    ax.set_ylabel('Policy Agreement (%)', fontsize=12)
    ax.set_title('Policy Agreement vs Discount Factor', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Plot 4: Threshold difference vs gamma
    ax = axes[1, 1]
    threshold_diffs = [r['threshold_diff'] for r in all_results if r['threshold_diff']]
    valid_gammas_diff = [r['gamma'] for r in all_results if r['threshold_diff']]

    if threshold_diffs:
        ax.plot(valid_gammas_diff, threshold_diffs, 'o-', linewidth=2.5, markersize=10,
               color='purple')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Discount Factor (γ)', fontsize=12)
        ax.set_ylabel('Threshold Difference (SL - VI) miles', fontsize=12)
        ax.set_title('How Thresholds Diverge with γ', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.suptitle('How Discount Factor Affects VI vs SL Comparison\n' +
                 'Do methods converge with higher γ?',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig('results/bus_engine_gamma_sensitivity.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_gamma_sensitivity.png")
    plt.close()


if __name__ == "__main__":
    main()
