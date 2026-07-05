#!/usr/bin/env python
"""
Extract Policy from Score-Life Programming Results

The Score function S(l*, X) at its maximum gives us an estimate of V(X).
From V(X), we can extract the optimal policy by comparing Q-values.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def extract_value_function_from_score_life(gamma=0.9, n_states=100):
    """
    Estimate the value function V(X) by computing max_l S(l, X) for many states.

    The Score function at its maximum gives us V(X).
    """
    env = BusEngineEnvironment()
    slp = ScoreLifeProgramming(env, gamma=gamma, N=10, num_samples=100)

    # Sample states from 0 to max_mileage
    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    value_estimates = []
    optimal_ls = []

    print(f"\nEstimating Value Function via Score-Life (γ={gamma})")
    print("=" * 70)

    for i, state in enumerate(states):
        if i % 10 == 0:
            print(f"Processing state {i}/{n_states}: {state:.0f} miles...")

        # Compute Score function and find maximum
        score_func = slp._compute_faber_schauder_coefficients()

        # Search for optimal l*
        l_values = np.linspace(0, 1, 100)
        scores = []

        for l in l_values:
            score = slp.S(l, np.array([state]))
            scores.append(score)

        # Maximum score is our estimate of V(state)
        max_idx = np.argmax(scores)
        V_estimate = scores[max_idx]
        l_optimal = l_values[max_idx]

        value_estimates.append(V_estimate)
        optimal_ls.append(l_optimal)

    return states, np.array(value_estimates), np.array(optimal_ls)


def extract_policy_from_value_function(states, V, gamma=0.9):
    """
    Given V(X) for all states, extract the optimal policy.

    Policy: Replace if Q(X, replace) > Q(X, keep)

    Where:
    - Q(X, keep) = -cost(X) + γ * V(X + ΔX)  (ΔX is random transition)
    - Q(X, replace) = -100 + γ * V(0)
    """
    env = BusEngineEnvironment()

    # Interpolate V for arbitrary states
    from scipy.interpolate import interp1d
    V_interp = interp1d(states, V, kind='linear', fill_value='extrapolate')

    policy = []
    Q_keep_values = []
    Q_replace_values = []

    print("\nExtracting Policy from Value Function")
    print("=" * 70)

    for state in states:
        # Q(X, replace) = -replacement_cost + γ * V(0)
        Q_replace = -100 + gamma * V_interp(0)

        # Q(X, keep) = E[-cost(X) + γ * V(X + ΔX)]
        # We need to account for stochastic transitions
        # Approximate by sampling expected next state

        # Expected transitions (from environment definition):
        # - Small increase (0-1000): p=0.1
        # - Medium increase (1000-3000): q=0.3
        # - Large increase (3000-10000): 1-p-q=0.6

        p, q = 0.1, 0.3
        small_delta = np.random.uniform(0, 1000)
        medium_delta = np.random.uniform(1000, 3000)
        large_delta = np.random.uniform(3000, 10000)

        # Expected next state
        next_state_expected = (
            p * (state + small_delta) +
            q * (state + medium_delta) +
            (1 - p - q) * (state + large_delta)
        )

        # Operating cost
        operating_cost = -0.01 * state

        # Q(X, keep)
        Q_keep = operating_cost + gamma * V_interp(next_state_expected)

        Q_keep_values.append(Q_keep)
        Q_replace_values.append(Q_replace)

        # Optimal action: 1 if replace, 0 if keep
        action = 1 if Q_replace > Q_keep else 0
        policy.append(action)

    # Find threshold
    policy_array = np.array(policy)
    threshold_idx = np.where(policy_array == 1)[0]

    if len(threshold_idx) > 0:
        threshold = states[threshold_idx[0]]
    else:
        threshold = states[-1]

    print(f"\nExtracted Policy Threshold: {threshold:.0f} miles")
    print(f"  (Replace when mileage ≥ {threshold:.0f})")

    return policy_array, threshold, Q_keep_values, Q_replace_values


def visualize_results(states, V, policy, Q_keep, Q_replace, threshold, gamma):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Value Function
    ax = axes[0, 0]
    ax.plot(states, V, 'b-', linewidth=2, label='V(X) from Score-Life')
    ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.0f} mi')
    ax.set_xlabel('Mileage (miles)', fontsize=11)
    ax.set_ylabel('Value Function V(X)', fontsize=11)
    ax.set_title(f'Value Function Estimated from Score-Life (γ={gamma})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Policy
    ax = axes[0, 1]
    ax.plot(states, policy, 'g-', linewidth=2)
    ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.0f} mi')
    ax.set_xlabel('Mileage (miles)', fontsize=11)
    ax.set_ylabel('Action (0=Keep, 1=Replace)', fontsize=11)
    ax.set_title('Extracted Policy', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Q-values
    ax = axes[1, 0]
    ax.plot(states, Q_keep, 'b-', linewidth=2, label='Q(X, keep)')
    ax.plot(states, Q_replace, 'r-', linewidth=2, label='Q(X, replace)')
    ax.axvline(threshold, color='g', linestyle='--', label=f'Threshold={threshold:.0f} mi')
    ax.set_xlabel('Mileage (miles)', fontsize=11)
    ax.set_ylabel('Q-value', fontsize=11)
    ax.set_title('Q-values: Keep vs Replace', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Advantage Function
    ax = axes[1, 1]
    advantage = np.array(Q_keep) - np.array(Q_replace)
    ax.plot(states, advantage, 'm-', linewidth=2)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.0f} mi')
    ax.fill_between(states, 0, advantage, where=(advantage > 0), alpha=0.3, color='blue', label='Keep better')
    ax.fill_between(states, 0, advantage, where=(advantage <= 0), alpha=0.3, color='red', label='Replace better')
    ax.set_xlabel('Mileage (miles)', fontsize=11)
    ax.set_ylabel('A(X) = Q(X, keep) - Q(X, replace)', fontsize=11)
    ax.set_title('Advantage Function', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'results/policy_from_score_life_gamma{gamma:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("EXTRACTING POLICY FROM SCORE-LIFE PROGRAMMING")
    print("=" * 70)

    # Test with γ=0.9 (one of our tested values)
    gamma = 0.9
    n_states = 50

    # Step 1: Estimate V(X) from Score-Life
    states, V, optimal_ls = extract_value_function_from_score_life(
        gamma=gamma,
        n_states=n_states
    )

    # Step 2: Extract policy from V(X)
    policy, threshold, Q_keep, Q_replace = extract_policy_from_value_function(
        states, V, gamma=gamma
    )

    # Step 3: Visualize
    visualize_results(states, V, policy, Q_keep, Q_replace, threshold, gamma)

    # Step 4: Compare with VI results
    print("\n" + "=" * 70)
    print("COMPARISON WITH VALUE ITERATION")
    print("=" * 70)

    # Load VI results
    with open('results/bus_engine_definitive_comparison.json', 'r') as f:
        vi_results = json.load(f)

    # Find VI result for this gamma
    vi_threshold = None
    for result in vi_results:
        if abs(result['gamma'] - gamma) < 0.01:
            vi_threshold = result['VI']['threshold']
            vi_reward = result['VI']['reward']
            break

    if vi_threshold:
        print(f"\nValue Iteration threshold: {vi_threshold:.0f} miles")
        print(f"Score-Life extracted threshold: {threshold:.0f} miles")
        print(f"Difference: {abs(vi_threshold - threshold):.0f} miles ({abs(vi_threshold - threshold)/vi_threshold*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
