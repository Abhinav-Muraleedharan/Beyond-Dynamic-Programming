#!/usr/bin/env python
"""
FINAL definitive comparison with cached transitions for fast convergence.

Pre-computes transition samples once, then reuses them for deterministic VI.
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


def precompute_transitions(n_states, max_mileage, transition_samples=1000):
    """Pre-compute transition samples for each state to enable deterministic VI."""

    print("Pre-computing transition samples...")
    states = np.linspace(0, max_mileage, n_states)
    env = BusEngineEnvironment(max_state=max_mileage)

    # Store samples for each state
    transitions = {}

    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}")

        next_states = []
        rewards = []

        np.random.seed(i)  # Reproducible
        for sample in range(transition_samples):
            env.set_state(state)
            next_state, reward, _, _, _ = env.step(0)  # keep action
            next_states.append(next_state[0])
            rewards.append(reward)

        transitions[i] = {
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }

    print(f"  Done! Cached {len(transitions)} state transitions")
    return transitions


def correct_vi_cached(gamma, n_states, max_mileage, transitions, tolerance=1e-6):
    """Correct VI using pre-computed transitions for fast deterministic convergence."""

    print("\n" + "=" * 80)
    print("CORRECT VALUE ITERATION (Cached Transitions)")
    print("=" * 80)

    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Use cached transition samples
            next_states = transitions[i]['next_states']
            rewards = transitions[i]['rewards']

            # Interpolate V at each sampled next state
            V_next_samples = []
            for ns in next_states:
                next_idx = np.argmin(np.abs(states - ns))
                V_next_samples.append(V[next_idx])

            # Expected cost and expected V(next)
            E_reward = np.mean(rewards)
            E_V_next = np.mean(V_next_samples)

            Q_keep = E_reward + gamma * E_V_next
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        max_change = np.max(np.abs(V_new - V))

        if iteration % 50 == 0:
            print(f"  Iteration {iteration+1}: max_change = {max_change:.10f}")

        if max_change < tolerance:
            print(f"\n  ✅ Converged in {iteration + 1} iterations!")
            print(f"     Final max change: {max_change:.12f}")
            break

        V = V_new.copy()

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(states):
        next_states = transitions[i]['next_states']
        rewards = transitions[i]['rewards']

        V_next_samples = [V[np.argmin(np.abs(states - ns))] for ns in next_states]

        Q_keep = np.mean(rewards) + gamma * np.mean(V_next_samples)
        Q_replace = -100 + gamma * V[0]

        policy[i] = 1 if Q_replace > Q_keep else 0

    threshold_idx = np.where(policy == 1)[0]
    threshold = states[threshold_idx[0]] if len(threshold_idx) > 0 else max_mileage

    print(f"\n  V(0) = {V[0]:.6f}")
    print(f"  V({max_mileage}) = {V[-1]:.6f}")
    print(f"  Threshold = {threshold:.0f} miles")

    return states, V, policy, threshold


def compute_score_life(gamma, N, num_samples, n_states, n_l_points, max_mileage):
    """Score-Life value function."""

    print("\n" + "=" * 80)
    print("SCORE-LIFE VALUE FUNCTION")
    print("=" * 80)

    states = np.linspace(0, max_mileage, n_states)
    V_sl = np.zeros(n_states)
    optimal_l = np.zeros(n_states)

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

        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = [slp.S(l, np.array([state])) for l in l_values]

        max_idx = np.argmax(scores)
        V_sl[i] = scores[max_idx]
        optimal_l[i] = l_values[max_idx]

    # Extract policy
    policy_sl = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        p, q = 0.1, 0.3
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(states[i] + expected_delta, max_mileage)
        next_idx = np.argmin(np.abs(states - next_state))

        Q_keep = -0.01 * next_state + gamma * V_sl[next_idx]
        Q_replace = -100 + gamma * V_sl[0]
        policy_sl[i] = 1 if Q_replace > Q_keep else 0

    threshold_idx = np.where(policy_sl == 1)[0]
    threshold_sl = states[threshold_idx[0]] if len(threshold_idx) > 0 else max_mileage

    print(f"\n  V(0) = {V_sl[0]:.6f}")
    print(f"  V({max_mileage}) = {V_sl[-1]:.6f}")
    print(f"  Threshold = {threshold_sl:.0f} miles")

    return states, V_sl, policy_sl, threshold_sl, optimal_l


def create_comprehensive_plot(states, V_vi, policy_vi, threshold_vi,
                              V_sl, policy_sl, threshold_sl, optimal_l, gamma, N):
    """Create final comparison plot."""

    from scipy.interpolate import interp1d

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Plot 1: Value functions (large, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(states, V_vi, 'b-', linewidth=3.5, label='Correct VI (Cached MC)',
             marker='o', markersize=7, alpha=0.8)
    ax1.plot(states, V_sl, 'r--', linewidth=3.5, label='Score-Life',
             marker='s', markersize=7, alpha=0.8)
    ax1.axvline(threshold_vi, color='blue', linestyle=':', linewidth=2.5, alpha=0.7)
    ax1.axvline(threshold_sl, color='red', linestyle=':', linewidth=2.5, alpha=0.7)

    ax1.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value V(X)', fontsize=13, fontweight='bold')
    ax1.set_title(f'DEFINITIVE: Correct VI vs Score-Life (γ={gamma}, N={N})',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Statistics box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    diff = V_sl - V_vi
    corr = np.corrcoef(V_vi, V_sl)[0, 1]
    rmse = np.sqrt(np.mean(diff**2))
    policy_match = np.sum(policy_vi == policy_sl) / len(policy_vi) * 100

    stats_text = f"""STATISTICS

Correlation:   {corr:.6f}
RMSE:          {rmse:.4f}
Mean diff:     {np.mean(diff):.4f}
Max |diff|:    {np.max(np.abs(diff)):.4f}

Policy match:  {policy_match:.1f}%
Threshold diff: {abs(threshold_vi-threshold_sl):.0f} mi

VI threshold:  {threshold_vi:.0f} mi
SL threshold:  {threshold_sl:.0f} mi"""

    ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 3: Difference
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(states, diff, 'purple', linewidth=3, marker='d', markersize=5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(np.mean(diff), color='orange', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(diff):.2f}')
    ax3.fill_between(states, 0, diff, alpha=0.3, color='purple')
    ax3.set_xlabel('Mileage', fontsize=12, fontweight='bold')
    ax3.set_ylabel('V_SL - V_VI', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Difference', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Percent difference
    ax4 = fig.add_subplot(gs[1, 1])
    pct_diff = (diff / np.abs(V_vi)) * 100
    ax4.plot(states, pct_diff, 'green', linewidth=3, marker='o', markersize=5)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(np.mean(pct_diff), color='orange', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(pct_diff):.2f}%')
    ax4.set_xlabel('Mileage', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% Difference', fontsize=12, fontweight='bold')
    ax4.set_title('Relative Difference', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Correlation
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(V_vi, V_sl, s=100, alpha=0.7, c=states, cmap='viridis', edgecolors='black')
    v_min, v_max = min(V_vi.min(), V_sl.min()), max(V_vi.max(), V_sl.max())
    ax5.plot([v_min, v_max], [v_min, v_max], 'r--', linewidth=2.5, label='Perfect', alpha=0.7)
    z = np.polyfit(V_vi, V_sl, 1)
    p = np.poly1d(z)
    ax5.plot(V_vi, p(V_vi), 'b-', linewidth=2, alpha=0.6, label=f'Fit: {z[0]:.3f}x+{z[1]:.1f}')
    ax5.set_xlabel('VI V(X)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('SL V(X)', fontsize=12, fontweight='bold')
    ax5.set_title(f'r={corr:.6f}', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Policies
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(states, policy_vi, 'b-', linewidth=3, label='VI', marker='o', markersize=7, alpha=0.7)
    ax6.plot(states, policy_sl, 'r--', linewidth=3, label='SL', marker='s', markersize=7, alpha=0.7)
    ax6.set_xlabel('Mileage', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Action', fontsize=12, fontweight='bold')
    ax6.set_title(f'Policies ({policy_match:.1f}% match)', fontsize=13, fontweight='bold')
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Keep', 'Replace'])
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Optimal l
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(states, optimal_l, 'purple', linewidth=3, marker='d', markersize=6)
    ax7.axvline(threshold_sl, color='red', linestyle='--', alpha=0.6)
    ax7.set_xlabel('Mileage', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Optimal l*', fontsize=12, fontweight='bold')
    ax7.set_title('Score-Life l* Parameter', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Residuals histogram
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.hist(diff, bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2)
    ax8.axvline(np.mean(diff), color='orange', linestyle='--', linewidth=2)
    ax8.set_xlabel('V_SL - V_VI', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax8.set_title('Residuals Distribution', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DEFINITIVE COMPARISON: Correct VI vs Score-Life Programming',
                fontsize=17, fontweight='bold', y=0.998)

    filename = 'results/definitive_comparison_final.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    return corr, rmse, policy_match


def main():
    gamma = 0.9
    N = 50
    num_samples = 1000
    n_states = 30
    transition_samples = 1000
    max_mileage = 10000

    print("=" * 80)
    print("DEFINITIVE APPLES-TO-APPLES COMPARISON (FINAL)")
    print("=" * 80)
    print("\nKey improvements:")
    print("  ✓ Pre-cached transition samples (deterministic VI)")
    print("  ✓ Same environment and parameters")
    print("  ✓ Both use correct Bellman equation")
    print()

    # Pre-compute transitions
    transitions = precompute_transitions(n_states, max_mileage, transition_samples)

    # Correct VI
    states_vi, V_vi, policy_vi, threshold_vi = correct_vi_cached(
        gamma, n_states, max_mileage, transitions
    )

    # Score-Life
    states_sl, V_sl, policy_sl, threshold_sl, optimal_l = compute_score_life(
        gamma, N, num_samples, n_states, 30, max_mileage
    )

    # Plot
    corr, rmse, policy_match = create_comprehensive_plot(
        states_vi, V_vi, policy_vi, threshold_vi,
        V_sl, policy_sl, threshold_sl, optimal_l, gamma, N
    )

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"\nCorrelation:     r = {corr:.6f}")
    print(f"RMSE:            {rmse:.4f}")
    print(f"Policy match:    {policy_match:.1f}%")

    if corr > 0.999:
        print("\n✅ EXCELLENT! Nearly perfect (r > 0.999)")
    elif corr > 0.99:
        print("\n✅ VERY GOOD! (r > 0.99)")
    elif corr > 0.95:
        print("\n⚠️  GOOD (r > 0.95)")
    else:
        print(f"\n⚠️  Moderate (r = {corr:.4f})")

    if policy_match == 100:
        print("✅ Policies IDENTICAL!")
    elif policy_match >= 95:
        print(f"✅ Policies {policy_match:.1f}% match")


if __name__ == "__main__":
    main()
