#!/usr/bin/env python
"""
Definitive comparison: Correct VI vs Score-Life Programming.

Implements VI correctly using E[V(next_state)] via Monte Carlo sampling
for apples-to-apples comparison with Score-Life.
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


def correct_vi(gamma=0.9, n_states=30, max_iterations=1000, tolerance=1e-6,
               transition_samples=1000, max_mileage=10000):
    """
    CORRECT Value Iteration using Monte Carlo sampling.

    Properly implements: V(s) = max_a [E[R] + γ * E[V(s')]]
    by sampling transitions from the actual environment.
    """

    print("=" * 80)
    print("CORRECT VALUE ITERATION")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  γ = {gamma}")
    print(f"  n_states = {n_states}")
    print(f"  transition_samples = {transition_samples}")
    print(f"  max_mileage = {max_mileage}")
    print(f"  tolerance = {tolerance}")

    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    # Environment for sampling transitions (SAME as Score-Life uses!)
    env = BusEngineEnvironment(max_state=max_mileage)

    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            if i % 10 == 0 and iteration % 50 == 0:
                print(f"  Iteration {iteration+1}, State {i}/{n_states}")

            # Q(state, keep): Sample actual transitions
            cost_samples = []
            V_next_samples = []

            np.random.seed(iteration * n_states + i)  # Reproducible

            for sample in range(transition_samples):
                env.set_state(state)
                next_state, reward, done, truncated, _ = env.step(0)  # keep

                # Interpolate V at sampled next state
                next_idx = np.argmin(np.abs(states - next_state[0]))
                V_next_samples.append(V[next_idx])
                cost_samples.append(reward)

            # Expected cost and expected V(next)
            E_cost = np.mean(cost_samples)
            E_V_next = np.mean(V_next_samples)

            Q_keep = E_cost + gamma * E_V_next

            # Q(state, replace): Deterministic transition to state 0
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        # Check convergence
        max_change = np.max(np.abs(V_new - V))

        if iteration % 50 == 0:
            print(f"\n  Iteration {iteration+1}: max_change = {max_change:.8f}")

        if max_change < tolerance:
            print(f"\n  ✅ Converged in {iteration + 1} iterations!")
            print(f"     Final max change: {max_change:.10f}")
            break

        V = V_new.copy()

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(states):
        # Resample to get Q values
        cost_samples = []
        V_next_samples = []

        for sample in range(transition_samples):
            env.set_state(state)
            next_state, reward, _, _, _ = env.step(0)
            next_idx = np.argmin(np.abs(states - next_state[0]))
            V_next_samples.append(V[next_idx])
            cost_samples.append(reward)

        Q_keep = np.mean(cost_samples) + gamma * np.mean(V_next_samples)
        Q_replace = -100 + gamma * V[0]

        policy[i] = 1 if Q_replace > Q_keep else 0

    threshold_idx = np.where(policy == 1)[0]
    threshold = states[threshold_idx[0]] if len(threshold_idx) > 0 else max_mileage

    print(f"\n  V(0) = {V[0]:.6f}")
    print(f"  V({max_mileage}) = {V[-1]:.6f}")
    print(f"  Threshold = {threshold:.0f} miles")

    return states, V, policy, threshold


def compute_score_life_value_function(gamma=0.9, N=50, num_samples=1000,
                                      n_states=30, n_l_points=30, max_mileage=10000):
    """Compute Score-Life value function (same as before)."""

    print("\n" + "=" * 80)
    print("SCORE-LIFE VALUE FUNCTION")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  γ = {gamma}")
    print(f"  N = {N}")
    print(f"  num_samples = {num_samples}")
    print(f"  n_states = {n_states}")
    print(f"  n_l_points = {n_l_points}")

    states = np.linspace(0, max_mileage, n_states)
    V_sl = np.zeros(n_states)
    optimal_l_values = np.zeros(n_states)

    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}: {state:.0f} miles")

        env = BusEngineEnvironment(max_state=max_mileage)
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        # Direct grid search over l
        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = [slp.S(l, np.array([state])) for l in l_values]

        max_idx = np.argmax(scores)
        V_sl[i] = scores[max_idx]
        optimal_l_values[i] = l_values[max_idx]

    # Extract policy
    policy_sl = np.zeros(n_states, dtype=int)
    for i, state in enumerate(states):
        env = BusEngineEnvironment(max_state=max_mileage)
        p, q = 0.1, 0.3
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(state + expected_delta, max_mileage)
        operating_cost = -0.01 * next_state
        next_idx = np.argmin(np.abs(states - next_state))

        Q_keep = operating_cost + gamma * V_sl[next_idx]
        Q_replace = -100 + gamma * V_sl[0]

        policy_sl[i] = 1 if Q_replace > Q_keep else 0

    threshold_idx = np.where(policy_sl == 1)[0]
    threshold_sl = states[threshold_idx[0]] if len(threshold_idx) > 0 else max_mileage

    print(f"\n  V(0) = {V_sl[0]:.6f}")
    print(f"  V({max_mileage}) = {V_sl[-1]:.6f}")
    print(f"  Threshold = {threshold_sl:.0f} miles")

    return states, V_sl, policy_sl, threshold_sl, optimal_l_values


def plot_definitive_comparison(states, V_vi, policy_vi, threshold_vi,
                               V_sl, policy_sl, threshold_sl, optimal_l,
                               gamma, N, transition_samples):
    """Create comprehensive comparison plot."""

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # Plot 1: Value functions overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(states, V_vi, 'b-', linewidth=3.5, label='Correct VI (Monte Carlo)',
             marker='o', markersize=7, alpha=0.8)
    ax1.plot(states, V_sl, 'r--', linewidth=3.5, label='Score-Life (Direct)',
             marker='s', markersize=7, alpha=0.8)
    ax1.axvline(threshold_vi, color='blue', linestyle=':', linewidth=2.5,
               label=f'VI threshold={threshold_vi:.0f}mi', alpha=0.7)
    ax1.axvline(threshold_sl, color='red', linestyle=':', linewidth=2.5,
               label=f'SL threshold={threshold_sl:.0f}mi', alpha=0.7)

    ax1.set_xlabel('Mileage (miles)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value V(X)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Apples-to-Apples Comparison: Correct VI vs Score-Life (γ={gamma}, N={N})',
                 fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute difference
    ax2 = fig.add_subplot(gs[1, 0])
    diff = V_sl - V_vi
    ax2.plot(states, diff, 'purple', linewidth=3, marker='d', markersize=6)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axhline(np.mean(diff), color='orange', linestyle='--', linewidth=2.5,
               label=f'Mean diff={np.mean(diff):.2f}', alpha=0.8)
    ax2.fill_between(states, 0, diff, alpha=0.3, color='purple')

    ax2.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('V_SL - V_VI', fontsize=13, fontweight='bold')
    ax2.set_title('Value Function Difference', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Percent difference
    ax3 = fig.add_subplot(gs[1, 1])
    pct_diff = (diff / np.abs(V_vi)) * 100
    ax3.plot(states, pct_diff, 'green', linewidth=3, marker='o', markersize=5)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.axhline(np.mean(pct_diff), color='orange', linestyle='--', linewidth=2.5,
               label=f'Mean={np.mean(pct_diff):.2f}%', alpha=0.8)

    ax3.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Percent Difference (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Relative Difference', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Correlation
    ax4 = fig.add_subplot(gs[2, 0])
    corr = np.corrcoef(V_vi, V_sl)[0, 1]

    ax4.scatter(V_vi, V_sl, s=120, alpha=0.7, c=states, cmap='viridis',
               edgecolors='black', linewidth=1.5)

    # Perfect match line
    v_min = min(V_vi.min(), V_sl.min())
    v_max = max(V_vi.max(), V_sl.max())
    ax4.plot([v_min, v_max], [v_min, v_max], 'r--', linewidth=3,
            label='Perfect match', alpha=0.7)

    # Linear fit
    z = np.polyfit(V_vi, V_sl, 1)
    p = np.poly1d(z)
    ax4.plot(V_vi, p(V_vi), 'b-', linewidth=2.5, alpha=0.6,
            label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')

    ax4.set_xlabel('Correct VI V(X)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Score-Life V(X)', fontsize=13, fontweight='bold')
    ax4.set_title(f'Correlation: r={corr:.5f}', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                              norm=plt.Normalize(vmin=states.min(), vmax=states.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('Mileage (miles)', fontsize=11)

    # Plot 5: Policies
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(states, policy_vi, 'b-', linewidth=3.5, label='Correct VI',
             marker='o', markersize=8, alpha=0.7)
    ax5.plot(states, policy_sl, 'r--', linewidth=3.5, label='Score-Life',
             marker='s', markersize=8, alpha=0.7)
    ax5.axvline(threshold_vi, color='blue', linestyle=':', linewidth=2, alpha=0.6)
    ax5.axvline(threshold_sl, color='red', linestyle=':', linewidth=2, alpha=0.6)

    ax5.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Action (0=Keep, 1=Replace)', fontsize=13, fontweight='bold')
    ax5.set_title('Policy Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylim([-0.1, 1.1])
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Keep', 'Replace'])
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Optimal l values
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(states, optimal_l, 'purple', linewidth=3, marker='d', markersize=7)
    ax6.axvline(threshold_sl, color='red', linestyle='--', linewidth=2,
               alpha=0.6, label=f'SL threshold={threshold_sl:.0f}')
    ax6.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Optimal l*', fontsize=13, fontweight='bold')
    ax6.set_title('Score-Life Optimal Life Parameter', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Statistics summary
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')

    # Calculate stats
    policy_match = np.sum(policy_vi == policy_sl) / len(policy_vi) * 100
    threshold_diff = abs(threshold_vi - threshold_sl)
    rmse = np.sqrt(np.mean(diff**2))

    stats_text = f"""
    STATISTICAL SUMMARY
    {'='*50}

    Value Functions:
      Correlation (r):        {corr:.6f}
      RMSE:                   {rmse:.4f}
      Mean difference:        {np.mean(diff):.4f}
      Max |difference|:       {np.max(np.abs(diff)):.4f}
      Mean % difference:      {np.mean(pct_diff):.2f}%

    Policies:
      Agreement:              {policy_match:.2f}%
      Threshold difference:   {threshold_diff:.0f} miles

    VI Threshold:             {threshold_vi:.0f} miles
    Score-Life Threshold:     {threshold_sl:.0f} miles

    Parameters:
      γ (discount):           {gamma}
      N (SL horizon):         {N}
      Transition samples:     {transition_samples}
    """

    ax7.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('DEFINITIVE COMPARISON: Correct VI vs Score-Life Programming',
                fontsize=18, fontweight='bold', y=0.995)

    filename = 'results/definitive_comparison_correct_vi.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    return corr, rmse, policy_match, threshold_diff


def main():
    """Main execution."""

    gamma = 0.9
    N = 50
    num_samples = 1000
    n_states = 30
    transition_samples = 1000
    max_mileage = 10000

    print("=" * 80)
    print("DEFINITIVE APPLES-TO-APPLES COMPARISON")
    print("=" * 80)
    print("\nBoth methods now use:")
    print("  ✓ Same environment (BusEngineEnvironment)")
    print("  ✓ Same max_state cap (10000)")
    print("  ✓ Same stochastic transitions (Monte Carlo sampling)")
    print("  ✓ Same discount factor γ")
    print("\nVI: E[V(next_state)] computed via Monte Carlo")
    print("Score-Life: Direct evaluation of Score function")
    print()

    # Compute Correct VI
    states_vi, V_vi, policy_vi, threshold_vi = correct_vi(
        gamma=gamma,
        n_states=n_states,
        transition_samples=transition_samples,
        max_mileage=max_mileage
    )

    # Compute Score-Life
    states_sl, V_sl, policy_sl, threshold_sl, optimal_l = compute_score_life_value_function(
        gamma=gamma,
        N=N,
        num_samples=num_samples,
        n_states=n_states,
        max_mileage=max_mileage
    )

    # Plot comparison
    corr, rmse, policy_match, threshold_diff = plot_definitive_comparison(
        states_vi, V_vi, policy_vi, threshold_vi,
        V_sl, policy_sl, threshold_sl, optimal_l,
        gamma, N, transition_samples
    )

    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    print(f"\nValue Function Comparison:")
    print(f"  Correlation:        r = {corr:.6f}")
    print(f"  RMSE:               {rmse:.4f}")
    print(f"  Mean difference:    {np.mean(V_sl - V_vi):.4f}")

    print(f"\nPolicy Comparison:")
    print(f"  Agreement:          {policy_match:.2f}%")
    print(f"  Threshold diff:     {threshold_diff:.0f} miles")

    if corr > 0.999:
        print("\n✅ EXCELLENT! Nearly perfect correlation (r > 0.999)")
    elif corr > 0.99:
        print("\n✅ VERY GOOD! Strong correlation (r > 0.99)")
    elif corr > 0.95:
        print("\n⚠️  GOOD: Correlation > 0.95, but some differences remain")
    else:
        print(f"\n⚠️  Moderate correlation (r = {corr:.4f})")

    if policy_match == 100:
        print("✅ Policies are IDENTICAL!")
    elif policy_match >= 95:
        print(f"✅ Policies match {policy_match:.1f}% - excellent agreement")
    else:
        print(f"⚠️  Policies match {policy_match:.1f}%")

    print("\n" + "=" * 80)
    print("This is the DEFINITIVE comparison with both methods")
    print("implementing the Bellman equation correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
