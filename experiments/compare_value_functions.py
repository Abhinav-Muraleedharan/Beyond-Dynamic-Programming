#!/usr/bin/env python
"""
Compare Value Functions from VI and Score-Life side-by-side.
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


def compute_vi_value_function(gamma=0.9, n_states=50):
    """
    Compute VI value function using value iteration.
    """
    print("Computing VI value function...")

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
            env.set_state(state)

            # Expected next state
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(state + expected_delta, max_mileage)

            # CRITICAL: Cost is on NEXT state, not current!
            operating_cost = -0.01 * next_state

            # Interpolate V at next_state
            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next

            # Q(state, replace)
            Q_replace = -100 + gamma * V[0]

            # V(state) = max(Q_keep, Q_replace)
            V_new[i] = max(Q_keep, Q_replace)

        # Check convergence
        if np.max(np.abs(V_new - V)) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break

        V = V_new.copy()

    # Extract threshold
    policy = []
    for i, state in enumerate(states):
        env.set_state(state)
        operating_cost = -0.01 * state
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        next_state = min(state + expected_delta, max_mileage)
        next_idx = np.argmin(np.abs(states - next_state))

        Q_keep = operating_cost + gamma * V[next_idx]
        Q_replace = -100 + gamma * V[0]

        action = 1 if Q_replace > Q_keep else 0
        policy.append(action)

    policy = np.array(policy)
    threshold_indices = np.where(policy == 1)[0]
    threshold = states[threshold_indices[0]] if len(threshold_indices) > 0 else states[-1]

    print(f"  VI Threshold: {threshold:.0f} miles")

    return states, V, threshold


def compute_sl_value_function(gamma=0.9, n_states=50, num_samples=1000):
    """
    Compute Score-Life value function.
    """
    print(f"\nComputing Score-Life value function ({num_samples} samples)...")

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    V_sl = []

    for i, state in enumerate(states):
        if i % 10 == 0:
            print(f"  State {i}/{n_states}")

        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=10, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Search for max
        l_values = np.linspace(0.01, 0.99, 30)
        scores = [score_func.compute_fractal(l) for l in l_values]

        V_sl.append(max(scores))

    print("  Done!")

    return states, np.array(V_sl)


def create_comparison_plot(states_vi, V_vi, threshold_vi, states_sl, V_sl, gamma):
    """Create comprehensive comparison plot."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Plot 1: Value functions overlaid
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(states_vi, V_vi, 'g-', linewidth=3, label='Value Iteration', marker='o', markersize=6, alpha=0.7)
    ax1.plot(states_sl, V_sl, 'b--', linewidth=3, label='Score-Life', marker='s', markersize=6, alpha=0.7)
    ax1.axvline(threshold_vi, color='green', linestyle=':', linewidth=2.5,
               label=f'VI threshold={threshold_vi:.0f} mi', alpha=0.8)
    ax1.set_xlabel('Mileage (miles)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value Function V(X)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Value Function Comparison: VI vs Score-Life (γ={gamma})',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: VI value function alone
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(states_vi, V_vi, 'g-', linewidth=2.5, marker='o', markersize=5)
    ax2.axvline(threshold_vi, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('V(X)', fontsize=12, fontweight='bold')
    ax2.set_title('Value Iteration V(X)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Range: [{V_vi.min():.1f}, {V_vi.max():.1f}]\nThreshold: {threshold_vi:.0f} mi"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 3: Score-Life value function alone
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(states_sl, V_sl, 'b-', linewidth=2.5, marker='s', markersize=5)
    ax3.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('V(X)', fontsize=12, fontweight='bold')
    ax3.set_title('Score-Life V(X)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Range: [{V_sl.min():.1f}, {V_sl.max():.1f}]"
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 4: Normalized comparison (scale to [0, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    V_vi_norm = (V_vi - V_vi.min()) / (V_vi.max() - V_vi.min())
    V_sl_norm = (V_sl - V_sl.min()) / (V_sl.max() - V_sl.min())

    ax4.plot(states_vi, V_vi_norm, 'g-', linewidth=2.5, label='VI (normalized)', marker='o', markersize=5, alpha=0.7)
    ax4.plot(states_sl, V_sl_norm, 'b--', linewidth=2.5, label='SL (normalized)', marker='s', markersize=5, alpha=0.7)
    ax4.axvline(threshold_vi, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Normalized V(X)', fontsize=12, fontweight='bold')
    ax4.set_title('Normalized Value Functions (0-1 scale)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Difference and correlation
    ax5 = fig.add_subplot(gs[2, 1])

    # Interpolate to same grid for comparison
    from scipy.interpolate import interp1d
    V_sl_interp = interp1d(states_sl, V_sl, kind='cubic')(states_vi)

    # Correlation
    corr = np.corrcoef(V_vi, V_sl_interp)[0, 1]

    ax5.scatter(V_vi, V_sl_interp, s=80, alpha=0.6, c=states_vi, cmap='viridis')
    ax5.plot([V_vi.min(), V_vi.max()], [V_vi.min(), V_vi.max()],
            'r--', linewidth=2, label='Perfect match', alpha=0.7)
    ax5.set_xlabel('VI V(X)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score-Life V(X)', fontsize=12, fontweight='bold')
    ax5.set_title(f'Correlation: r={corr:.3f}', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                              norm=plt.Normalize(vmin=states_vi.min(), vmax=states_vi.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax5)
    cbar.set_label('Mileage', fontsize=10)

    plt.suptitle(f'Complete Value Function Comparison (γ={gamma})',
                fontsize=17, fontweight='bold', y=0.995)

    filename = f'results/value_function_comparison_gamma{gamma:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALUE FUNCTION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nValue Iteration:")
    print(f"  Range: [{V_vi.min():.2f}, {V_vi.max():.2f}]")
    print(f"  Span: {V_vi.max() - V_vi.min():.2f}")
    print(f"  Threshold: {threshold_vi:.0f} miles")

    print(f"\nScore-Life:")
    print(f"  Range: [{V_sl.min():.2f}, {V_sl.max():.2f}]")
    print(f"  Span: {V_sl.max() - V_sl.min():.2f}")

    print(f"\nCorrelation: r = {corr:.3f}")

    if corr > 0.9:
        print("  ✅ Strong positive correlation - value functions are similar in shape!")
    elif corr > 0.7:
        print("  ⚠️  Moderate correlation - some similarity")
    else:
        print("  ❌ Weak correlation - value functions differ significantly")


def main():
    gamma = 0.9

    print("=" * 70)
    print("COMPARING VALUE FUNCTIONS: VI vs SCORE-LIFE")
    print("=" * 70)

    # Compute VI value function
    states_vi, V_vi, threshold_vi = compute_vi_value_function(gamma=gamma, n_states=50)

    # Compute Score-Life value function
    states_sl, V_sl = compute_sl_value_function(gamma=gamma, n_states=50, num_samples=1000)

    # Create comparison plot
    create_comparison_plot(states_vi, V_vi, threshold_vi, states_sl, V_sl, gamma)


if __name__ == "__main__":
    main()
