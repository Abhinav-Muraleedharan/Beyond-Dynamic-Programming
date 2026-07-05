#!/usr/bin/env python
"""
Proper Policy Extraction from Score-Life Programming

CORRECT APPROACH:
1. For each state X, compute V(X) = max_l S(l, X)
2. Build value function V(X) for dense state grid
3. Extract policy by comparing Q(X, keep) vs Q(X, replace)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
import time
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def compute_value_function_from_score(gamma, n_states=50, l_resolution=20, num_samples=100):
    """
    Compute V(X) = max_l S(l, X) for a grid of states.

    Args:
        gamma: Discount factor
        n_states: Number of states to sample
        l_resolution: Number of l values to try (search resolution)
        num_samples: Monte Carlo samples for Score function

    Returns:
        states, V_estimates, optimal_ls
    """
    env = BusEngineEnvironment()

    # State grid
    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    V_estimates = []
    optimal_ls = []

    print(f"\nComputing Value Function for γ={gamma}")
    print("=" * 70)
    print(f"States to evaluate: {n_states}")
    print(f"l search resolution: {l_resolution}")
    print()

    start_time = time.time()

    for i, state in enumerate(states):
        if i % 5 == 0:
            elapsed = time.time() - start_time
            print(f"State {i}/{n_states}: {state:.0f} miles (elapsed: {elapsed:.1f}s)")

        # Set environment to this state and compute Score function
        env.set_state(state)
        slp_state = ScoreLifeProgramming(
            env, gamma=gamma, N=10, j_max=5,
            num_samples=num_samples, reference_state=np.array([state])
        )

        # Use the SAME method as definitive comparison
        score_func = slp_state._compute_faber_schauder_coefficients()

        # Search over l values to maximize S(l, X)
        l_values = np.linspace(0.001, 0.999, l_resolution)
        scores = [score_func.compute_fractal(l) for l in l_values]

        # V(X) = max_l S(l, X)
        max_idx = np.argmax(scores)
        V_estimate = scores[max_idx]
        l_optimal = l_values[max_idx]

        V_estimates.append(V_estimate)
        optimal_ls.append(l_optimal)

    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.1f}s")
    print(f"Average per state: {total_time/n_states:.2f}s")

    return np.array(states), np.array(V_estimates), np.array(optimal_ls)


def extract_policy_from_value_function(states, V, gamma):
    """
    Extract optimal policy from value function V(X).

    Policy: π(X) = argmax_a Q(X, a)
    Where:
        Q(X, replace) = -100 + γ * V(0)
        Q(X, keep) = E[-cost(X) + γ * V(X')]
    """
    # Interpolate V for smooth evaluation
    V_interp = interp1d(states, V, kind='cubic', fill_value='extrapolate')

    # Create dense evaluation grid
    eval_states = np.linspace(0, states[-1], 500)

    # Environment parameters
    p, q = 0.1, 0.3
    operating_cost_rate = 0.01
    replacement_cost = 100.0

    # Q(X, replace) = -replacement_cost + γ * V(0)
    Q_replace = -replacement_cost + gamma * V_interp(0)

    Q_keep_values = []
    policy = []

    for state in eval_states:
        # Operating cost
        operating_cost = -operating_cost_rate * state

        # Expected next state under "keep" action
        # E[X' | keep] = X + E[ΔX]
        # where ΔX ~ p*Unif(0,1000) + q*Unif(1000,3000) + (1-p-q)*Unif(3000,10000)
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(state + expected_delta, states[-1])

        # Q(X, keep) = immediate_cost + γ * V(next_state)
        Q_keep = operating_cost + gamma * V_interp(next_state)
        Q_keep_values.append(Q_keep)

        # Optimal action
        action = 1 if Q_replace > Q_keep else 0
        policy.append(action)

    policy = np.array(policy)
    Q_keep_values = np.array(Q_keep_values)

    # Find threshold
    replace_indices = np.where(policy == 1)[0]
    if len(replace_indices) > 0:
        threshold = eval_states[replace_indices[0]]
    else:
        threshold = eval_states[-1]

    print(f"\nExtracted Policy:")
    print(f"  Threshold: {threshold:.0f} miles")
    print(f"  Rule: Replace when mileage ≥ {threshold:.0f}")

    return eval_states, policy, Q_keep_values, Q_replace, threshold


def visualize_comparison(states, V, eval_states, policy, Q_keep, Q_replace,
                         threshold_sl, threshold_vi, gamma):
    """Create comprehensive visualization."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Value Function
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(states, V, 'b-', linewidth=2.5, label='V(X) from Score-Life', marker='o', markersize=4)
    ax1.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2.5,
               label=f'Score-Life threshold={threshold_sl:.0f} mi', alpha=0.8)
    ax1.axvline(threshold_vi, color='green', linestyle='--', linewidth=2.5,
               label=f'VI threshold={threshold_vi:.0f} mi', alpha=0.8)
    ax1.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value Function V(X)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Value Function from Score-Life (γ={gamma})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Policy
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['blue' if p == 0 else 'red' for p in policy]
    ax2.scatter(eval_states, policy, c=colors, s=5, alpha=0.6)
    ax2.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2.5,
               label=f'SL={threshold_sl:.0f}', alpha=0.8)
    ax2.axvline(threshold_vi, color='green', linestyle='--', linewidth=2.5,
               label=f'VI={threshold_vi:.0f}', alpha=0.8)
    ax2.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Action', fontsize=12, fontweight='bold')
    ax2.set_title('Extracted Policy', fontsize=14, fontweight='bold')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Keep', 'Replace'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Q-values
    ax3 = fig.add_subplot(gs[1, :])
    Q_replace_array = np.full_like(eval_states, Q_replace)
    ax3.plot(eval_states, Q_keep, 'b-', linewidth=2.5, label='Q(X, keep)', alpha=0.8)
    ax3.plot(eval_states, Q_replace_array, 'r--', linewidth=2.5, label='Q(X, replace)', alpha=0.8)
    ax3.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2, alpha=0.6)
    ax3.axvline(threshold_vi, color='green', linestyle='--', linewidth=2, alpha=0.6)

    # Mark crossover point
    ax3.scatter([threshold_sl], [Q_replace], s=150, c='red', marker='*',
               zorder=5, edgecolors='black', linewidth=1.5, label='Crossover')

    ax3.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Q-value', fontsize=12, fontweight='bold')
    ax3.set_title('Q-values: Keep vs Replace', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Advantage Function
    ax4 = fig.add_subplot(gs[2, :])
    advantage = Q_keep - Q_replace_array
    ax4.plot(eval_states, advantage, 'm-', linewidth=2.5, alpha=0.8)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax4.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2.5,
               label=f'Score-Life={threshold_sl:.0f} mi', alpha=0.8)
    ax4.axvline(threshold_vi, color='green', linestyle='--', linewidth=2.5,
               label=f'VI={threshold_vi:.0f} mi', alpha=0.8)
    ax4.fill_between(eval_states, 0, advantage, where=(advantage > 0),
                    alpha=0.3, color='blue', label='Keep better')
    ax4.fill_between(eval_states, 0, advantage, where=(advantage <= 0),
                    alpha=0.3, color='red', label='Replace better')
    ax4.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('A(X) = Q(keep) - Q(replace)', fontsize=12, fontweight='bold')
    ax4.set_title('Advantage Function', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Policy Extraction via Value Function Maximization (γ={gamma})',
                fontsize=16, fontweight='bold', y=0.995)

    filename = f'results/proper_policy_extraction_gamma{gamma:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()


def compare_with_vi(threshold_sl, gamma):
    """Compare with VI results."""
    import json

    with open('results/bus_engine_definitive_comparison.json', 'r') as f:
        results = json.load(f)

    for result in results:
        if abs(result['gamma'] - gamma) < 0.01:
            threshold_vi = result['VI']['threshold']
            reward_vi = result['VI']['reward']

            diff = abs(threshold_sl - threshold_vi)
            pct_diff = (diff / threshold_vi) * 100

            print("\n" + "=" * 70)
            print("COMPARISON WITH VALUE ITERATION")
            print("=" * 70)
            print(f"\n{'Method':<25} {'Threshold (miles)':<20} {'Avg Reward':<15}")
            print("-" * 60)
            print(f"{'Value Iteration':<25} {threshold_vi:<20.0f} {reward_vi:<15.2f}")
            print(f"{'Score-Life (extracted)':<25} {threshold_sl:<20.0f} {'N/A':<15}")
            print(f"\n{'Difference (absolute)':<25} {diff:<20.0f}")
            print(f"{'Difference (relative)':<25} {pct_diff:<20.1f}%")

            if pct_diff < 10:
                print("\n✅ Policies MATCH closely!")
            elif pct_diff < 25:
                print("\n⚠️  Policies differ moderately")
            else:
                print("\n❌ Policies differ significantly")

            return threshold_vi

    return None


def main():
    """Main execution."""
    print("=" * 70)
    print("PROPER POLICY EXTRACTION FROM SCORE-LIFE PROGRAMMING")
    print("=" * 70)
    print("\nApproach:")
    print("1. Compute V(X) = max_l S(l, X) for state grid")
    print("2. Extract policy by comparing Q(X, keep) vs Q(X, replace)")
    print("3. Compare with Value Iteration results")

    # Test with γ=0.9
    gamma = 0.9

    print(f"\n{'='*70}")
    print(f"TESTING WITH γ = {gamma}")
    print(f"{'='*70}")

    # Step 1: Compute value function
    states, V, optimal_ls = compute_value_function_from_score(
        gamma=gamma,
        n_states=30,  # Reasonable resolution
        l_resolution=15,  # Search resolution for l
        num_samples=100  # Same as definitive comparison
    )

    # Step 2: Extract policy from V
    eval_states, policy, Q_keep, Q_replace, threshold_sl = extract_policy_from_value_function(
        states, V, gamma
    )

    # Step 3: Compare with VI
    threshold_vi = compare_with_vi(threshold_sl, gamma)

    # Step 4: Visualize
    if threshold_vi:
        visualize_comparison(states, V, eval_states, policy, Q_keep, Q_replace,
                           threshold_sl, threshold_vi, gamma)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
