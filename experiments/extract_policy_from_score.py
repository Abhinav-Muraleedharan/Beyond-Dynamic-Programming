#!/usr/bin/env python
"""
Extract optimal policy from Score-Life Programming.

Compute V(state) = max_l S(l, state) for all states using DIRECT evaluation
(not Faber-Schauder), then extract policy and compare with Value Iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def compute_vi_policy(gamma=0.9, n_states=30):
    """Compute Value Iteration value function and policy."""

    print("=" * 70)
    print("COMPUTING VALUE ITERATION POLICY")
    print("=" * 70)

    env = BusEngineEnvironment()
    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    # Run value iteration
    V = np.zeros(n_states)
    tolerance = 1e-6
    max_iterations = 1000

    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Q(state, keep)
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(state + expected_delta, max_mileage)

            # Cost on NEXT state
            operating_cost = -0.01 * next_state

            # Interpolate V at next_state
            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next

            # Q(state, replace)
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        if np.max(np.abs(V_new - V)) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break

        V = V_new.copy()

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(states):
        p, q = 0.1, 0.3
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(state + expected_delta, max_mileage)
        next_idx = np.argmin(np.abs(states - next_state))

        operating_cost = -0.01 * next_state
        Q_keep = operating_cost + gamma * V[next_idx]
        Q_replace = -100 + gamma * V[0]

        policy[i] = 1 if Q_replace > Q_keep else 0  # 1=replace, 0=keep

    # Find threshold
    replace_indices = np.where(policy == 1)[0]
    threshold = states[replace_indices[0]] if len(replace_indices) > 0 else max_mileage

    print(f"  VI Threshold: {threshold:.0f} miles")
    print(f"  V(0) = {V[0]:.2f}")
    print(f"  V({max_mileage}) = {V[-1]:.2f}")

    return states, V, policy, threshold


def compute_score_life_value_function(gamma=0.9, N=50, num_samples=1000, n_states=30, n_l_points=30):
    """Compute Score-Life value function using direct evaluation."""

    print("\n" + "=" * 70)
    print("COMPUTING SCORE-LIFE VALUE FUNCTION")
    print("=" * 70)
    print(f"Parameters: γ={gamma}, N={N}, num_samples={num_samples}")
    print(f"States: {n_states}, L-grid points: {n_l_points}")

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    V_sl = np.zeros(n_states)
    optimal_l_values = np.zeros(n_states)

    print("\nComputing V(state) = max_l S(l, state) for each state...")

    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}: {state:.0f} miles")

        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        # Direct grid search over l
        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = []

        for l in l_values:
            score = slp.S(l, np.array([state]))
            scores.append(score)

        scores = np.array(scores)
        max_idx = np.argmax(scores)

        V_sl[i] = scores[max_idx]
        optimal_l_values[i] = l_values[max_idx]

    print("\n  Done!")
    print(f"  V(0) = {V_sl[0]:.2f}, optimal l={optimal_l_values[0]:.3f}")
    print(f"  V({max_mileage}) = {V_sl[-1]:.2f}, optimal l={optimal_l_values[-1]:.3f}")

    return states, V_sl, optimal_l_values


def extract_policy_from_value_function(states, V, gamma=0.9):
    """Extract policy from value function."""

    print("\n" + "=" * 70)
    print("EXTRACTING SCORE-LIFE POLICY")
    print("=" * 70)

    max_mileage = 10000
    policy = np.zeros(len(states), dtype=int)

    for i, state in enumerate(states):
        # Q(state, keep)
        p, q = 0.1, 0.3
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(state + expected_delta, max_mileage)

        # Cost on NEXT state
        operating_cost = -0.01 * next_state

        # Interpolate V at next_state
        next_idx = np.argmin(np.abs(states - next_state))
        V_next = V[next_idx]

        Q_keep = operating_cost + gamma * V_next

        # Q(state, replace)
        Q_replace = -100 + gamma * V[0]

        policy[i] = 1 if Q_replace > Q_keep else 0  # 1=replace, 0=keep

    # Find threshold
    replace_indices = np.where(policy == 1)[0]
    threshold = states[replace_indices[0]] if len(replace_indices) > 0 else max_mileage

    print(f"  Score-Life Threshold: {threshold:.0f} miles")

    return policy, threshold


def plot_comparison(states, V_vi, policy_vi, threshold_vi,
                   V_sl, policy_sl, threshold_sl, optimal_l_values, gamma, N):
    """Create comprehensive comparison plot."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Plot 1: Value functions comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(states, V_vi, 'g-', linewidth=3, label='Value Iteration',
             marker='o', markersize=6, alpha=0.8)
    ax1.plot(states, V_sl, 'b--', linewidth=3, label='Score-Life (Direct)',
             marker='s', markersize=6, alpha=0.8)
    ax1.axvline(threshold_vi, color='green', linestyle=':', linewidth=2.5,
               label=f'VI threshold = {threshold_vi:.0f} mi', alpha=0.7)
    ax1.axvline(threshold_sl, color='blue', linestyle=':', linewidth=2.5,
               label=f'SL threshold = {threshold_sl:.0f} mi', alpha=0.7)

    ax1.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value V(X)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Value Functions: VI vs Score-Life (γ={gamma}, N={N})',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Policies comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(states, policy_vi, 'g-', linewidth=3, label='VI Policy',
             marker='o', markersize=7, alpha=0.7)
    ax2.plot(states, policy_sl, 'b--', linewidth=3, label='Score-Life Policy',
             marker='s', markersize=7, alpha=0.7)
    ax2.axvline(threshold_vi, color='green', linestyle=':', linewidth=2, alpha=0.6)
    ax2.axvline(threshold_sl, color='blue', linestyle=':', linewidth=2, alpha=0.6)

    ax2.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Action (0=Keep, 1=Replace)', fontsize=12, fontweight='bold')
    ax2.set_title('Policy Comparison', fontsize=13, fontweight='bold')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Keep', 'Replace'])
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Optimal l values
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(states, optimal_l_values, 'purple', linewidth=3, marker='d', markersize=6)
    ax3.axvline(threshold_sl, color='blue', linestyle='--', linewidth=2,
               alpha=0.6, label=f'SL threshold = {threshold_sl:.0f}')
    ax3.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Optimal l*', fontsize=12, fontweight='bold')
    ax3.set_title('Optimal Life Parameter vs State', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Value function difference
    ax4 = fig.add_subplot(gs[2, 0])
    diff = V_sl - V_vi
    ax4.plot(states, diff, 'r-', linewidth=2.5, marker='o', markersize=5)
    ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.axhline(np.mean(diff), color='orange', linestyle='--', linewidth=2,
               label=f'Mean diff = {np.mean(diff):.2f}', alpha=0.7)
    ax4.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('V_SL - V_VI', fontsize=12, fontweight='bold')
    ax4.set_title('Value Function Difference', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Correlation plot
    ax5 = fig.add_subplot(gs[2, 1])
    corr = np.corrcoef(V_vi, V_sl)[0, 1]

    ax5.scatter(V_vi, V_sl, s=100, alpha=0.7, c=states, cmap='viridis', edgecolors='black')

    # Perfect match line
    v_min = min(V_vi.min(), V_sl.min())
    v_max = max(V_vi.max(), V_sl.max())
    ax5.plot([v_min, v_max], [v_min, v_max], 'r--', linewidth=2.5,
            label='Perfect match', alpha=0.7)

    # Linear fit
    z = np.polyfit(V_vi, V_sl, 1)
    p = np.poly1d(z)
    ax5.plot(V_vi, p(V_vi), 'b-', linewidth=2, alpha=0.5,
            label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')

    ax5.set_xlabel('VI V(X)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score-Life V(X)', fontsize=12, fontweight='bold')
    ax5.set_title(f'Correlation: r={corr:.4f}', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='best')
    ax5.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                              norm=plt.Normalize(vmin=states.min(), vmax=states.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5)
    cbar.set_label('Mileage (miles)', fontsize=10)

    plt.suptitle(f'Complete Policy Extraction from Score-Life (γ={gamma}, N={N})',
                fontsize=17, fontweight='bold', y=0.998)

    filename = 'results/policy_extraction_score_life.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    return corr, diff


def main():
    """Main execution."""

    gamma = 0.9
    N = 50
    num_samples = 1000
    n_states = 30
    n_l_points = 30

    print("=" * 70)
    print("POLICY EXTRACTION FROM SCORE-LIFE PROGRAMMING")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  γ (gamma):       {gamma}")
    print(f"  N (horizon):     {N}")
    print(f"  num_samples:     {num_samples}")
    print(f"  n_states:        {n_states}")
    print(f"  n_l_points:      {n_l_points}")
    print()

    # Compute VI policy
    states_vi, V_vi, policy_vi, threshold_vi = compute_vi_policy(gamma=gamma, n_states=n_states)

    # Compute Score-Life value function
    states_sl, V_sl, optimal_l_values = compute_score_life_value_function(
        gamma=gamma, N=N, num_samples=num_samples, n_states=n_states, n_l_points=n_l_points
    )

    # Extract policy from Score-Life value function
    policy_sl, threshold_sl = extract_policy_from_value_function(states_sl, V_sl, gamma=gamma)

    # Plot comparison
    corr, diff = plot_comparison(states_vi, V_vi, policy_vi, threshold_vi,
                                V_sl, policy_sl, threshold_sl, optimal_l_values,
                                gamma, N)

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    print(f"\nValue Iteration:")
    print(f"  V(0):        {V_vi[0]:.2f}")
    print(f"  V(10000):    {V_vi[-1]:.2f}")
    print(f"  Range:       {V_vi.max() - V_vi.min():.2f}")
    print(f"  Threshold:   {threshold_vi:.0f} miles")

    print(f"\nScore-Life (Direct Evaluation):")
    print(f"  V(0):        {V_sl[0]:.2f}")
    print(f"  V(10000):    {V_sl[-1]:.2f}")
    print(f"  Range:       {V_sl.max() - V_sl.min():.2f}")
    print(f"  Threshold:   {threshold_sl:.0f} miles")

    print(f"\nValue Function Comparison:")
    print(f"  Correlation:     r = {corr:.4f}")
    print(f"  Mean difference: {np.mean(diff):.2f}")
    print(f"  Max difference:  {np.max(np.abs(diff)):.2f}")
    print(f"  RMSE:            {np.sqrt(np.mean(diff**2)):.2f}")

    print(f"\nPolicy Comparison:")
    policy_match = np.sum(policy_vi == policy_sl) / len(policy_vi) * 100
    threshold_diff = abs(threshold_vi - threshold_sl)

    print(f"  Policy agreement:    {policy_match:.1f}%")
    print(f"  Threshold difference: {threshold_diff:.0f} miles")

    if policy_match == 100:
        print(f"\n  ✅ PERFECT MATCH! Policies are identical!")
    elif policy_match >= 90:
        print(f"\n  ✅ EXCELLENT! Policies match {policy_match:.1f}%")
    elif policy_match >= 70:
        print(f"\n  ⚠️  GOOD: Policies match {policy_match:.1f}%")
    else:
        print(f"\n  ❌ Policies differ significantly ({policy_match:.1f}% match)")

    if threshold_diff == 0:
        print(f"  ✅ Thresholds are identical!")
    elif threshold_diff <= 500:
        print(f"  ⚠️  Thresholds differ by {threshold_diff:.0f} miles")
    else:
        print(f"  ❌ Thresholds differ by {threshold_diff:.0f} miles")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
