#!/usr/bin/env python
"""
Capital Asset Replacement: VI vs Score-Life Comparison

Economics perspective on asset management:
- Firm owns capital asset (bus engine)
- Asset degrades over time (increasing maintenance costs)
- Decision: Continue using (maintenance cost) vs Replace (fixed cost)
- Objective: Minimize expected discounted costs

This is a canonical problem in:
- Industrial organization (capital investment)
- Operations research (equipment replacement)
- Public economics (infrastructure management)

Economic interpretation:
- State = Asset age/condition (mileage)
- Action = Keep (pay maintenance) vs Replace (pay fixed cost)
- Dynamics = Stochastic deterioration
- Costs = Operating costs + replacement costs
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
    """Pre-compute transition samples for deterministic VI."""

    print("Pre-computing transitions...")
    states = np.linspace(0, max_mileage, n_states)
    env = BusEngineEnvironment(max_state=max_mileage)

    transitions = {}

    for i, mileage in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}")

        next_states = []
        rewards = []

        np.random.seed(i)  # Reproducible
        for sample in range(transition_samples):
            env.set_state(mileage)
            next_state, reward, _, _, _ = env.step(0)  # Keep action
            next_states.append(next_state[0])
            rewards.append(reward)

        transitions[i] = {
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }

    print(f"  Done! Cached {len(transitions)} state transitions")
    return transitions


def value_iteration_bus_engine(gamma, n_states, max_mileage, transitions, tolerance=1e-6):
    """VI for bus engine replacement using cached transitions."""

    print("\n" + "=" * 80)
    print("VALUE ITERATION: Capital Asset Replacement")
    print("=" * 80)

    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, mileage in enumerate(states):
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
            break

        V = V_new.copy()

    print(f"\n  V(mileage=0) = {V[0]:.6f}")
    print(f"  V(mileage={max_mileage}) = {V[-1]:.6f}")

    return states, V


def score_life_bus_engine(gamma, N, num_samples, n_states, n_l_points, max_mileage):
    """Score-Life for bus engine replacement."""

    print("\n" + "=" * 80)
    print("SCORE-LIFE: Capital Asset Replacement")
    print("=" * 80)

    states = np.linspace(0, max_mileage, n_states)
    V_sl = np.zeros(n_states)
    optimal_l = np.zeros(n_states)

    for i, mileage in enumerate(states):
        if i % 5 == 0:
            print(f"  Mileage {mileage:.0f} miles ({i}/{n_states})")

        env = BusEngineEnvironment(max_state=max_mileage)
        env.set_state(mileage)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([mileage])
        )

        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = [slp.S(l, np.array([mileage])) for l in l_values]

        max_idx = np.argmax(scores)
        V_sl[i] = scores[max_idx]
        optimal_l[i] = l_values[max_idx]

    print(f"\n  V(mileage=0) = {V_sl[0]:.6f}")
    print(f"  V(mileage={max_mileage}) = {V_sl[-1]:.6f}")

    return states, V_sl, optimal_l


def plot_comparison(states, V_vi, V_sl, optimal_l, gamma, N):
    """Create comparison visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Value functions
    ax1 = axes[0, 0]
    ax1.plot(states, V_vi, 'b-', linewidth=3, marker='o', markersize=6,
             alpha=0.8, label='Value Iteration')
    ax1.plot(states, V_sl, 'r--', linewidth=3, marker='s', markersize=6,
             alpha=0.8, label='Score-Life')
    ax1.set_xlabel('Asset Age (mileage)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value V(mileage)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Capital Asset Replacement: VI vs Score-Life (γ={gamma}, N={N})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax2 = axes[0, 1]
    diff = V_sl - V_vi
    ax2.plot(states, diff, 'purple', linewidth=3, marker='d', markersize=5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(states, 0, diff, alpha=0.3, color='purple')
    ax2.set_xlabel('Asset Age (mileage)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('V_SL - V_VI', fontsize=12, fontweight='bold')
    ax2.set_title('Difference', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Correlation
    ax3 = axes[1, 0]
    ax3.scatter(V_vi, V_sl, s=100, alpha=0.6, c=states, cmap='viridis', edgecolors='black')
    v_min, v_max = min(V_vi.min(), V_sl.min()), max(V_vi.max(), V_sl.max())
    ax3.plot([v_min, v_max], [v_min, v_max], 'r--', linewidth=2, label='Perfect')
    z = np.polyfit(V_vi, V_sl, 1)
    p = np.poly1d(z)
    ax3.plot(V_vi, p(V_vi), 'b-', linewidth=2, alpha=0.6,
             label=f'Fit: {z[0]:.3f}x+{z[1]:.1f}')
    corr = np.corrcoef(V_vi, V_sl)[0, 1]
    ax3.set_xlabel('VI Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score-Life Value', fontsize=12, fontweight='bold')
    ax3.set_title(f'Correlation: r={corr:.6f}', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Optimal l parameter
    ax4 = axes[1, 1]
    ax4.plot(states, optimal_l, 'green', linewidth=3, marker='o', markersize=6)
    ax4.set_xlabel('Asset Age (mileage)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Optimal l*', fontsize=12, fontweight='bold')
    ax4.set_title('Score-Life l* Parameter', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Capital Asset Replacement: VI vs Score-Life',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = 'results/economics_asset_replacement_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def main():
    gamma = 0.9
    N = 50
    num_samples = 2000  # Sweet spot found empirically
    n_states = 30
    n_l_points = 50  # Good balance of accuracy vs speed
    max_mileage = 10000
    transition_samples = 1000

    print("=" * 80)
    print("CAPITAL ASSET REPLACEMENT (Economics Perspective)")
    print("=" * 80)
    print("\nEconomics problem: When should a firm replace its capital asset?")
    print("- State: Asset age/condition (bus engine mileage)")
    print("- Action: Continue using vs Replace")
    print("- Costs: Maintenance (increasing) vs Replacement (fixed)")
    print("- Objective: Minimize expected discounted costs")
    print("\nApplications:")
    print("- Industrial organization (capital investment)")
    print("- Operations research (equipment replacement)")
    print("- Public economics (infrastructure management)")
    print()

    # Pre-compute transitions
    transitions = precompute_transitions(n_states, max_mileage, transition_samples)

    # Value Iteration
    states_vi, V_vi = value_iteration_bus_engine(
        gamma, n_states, max_mileage, transitions
    )

    # Score-Life
    states_sl, V_sl, optimal_l = score_life_bus_engine(
        gamma, N, num_samples, n_states, n_l_points, max_mileage
    )

    # Compare
    plot_comparison(states_vi, V_vi, V_sl, optimal_l, gamma, N)

    # Statistics
    corr = np.corrcoef(V_vi, V_sl)[0, 1]
    rmse = np.sqrt(np.mean((V_sl - V_vi)**2))
    mean_diff = np.mean(V_sl - V_vi)

    # Detailed analysis
    diff = V_sl - V_vi

    print("\n" + "=" * 80)
    print("DETAILED DIAGNOSTICS")
    print("=" * 80)
    print(f"\nFirst few states:")
    for i in range(min(5, len(states_vi))):
        print(f"  State {states_vi[i]:.0f}: VI={V_vi[i]:.2f}, SL={V_sl[i]:.2f}, Diff={diff[i]:.2f}")

    print(f"\nLast few states:")
    for i in range(max(0, len(states_vi)-5), len(states_vi)):
        print(f"  State {states_vi[i]:.0f}: VI={V_vi[i]:.2f}, SL={V_sl[i]:.2f}, Diff={diff[i]:.2f}")

    print("\n" + "=" * 80)
    print("VERIFICATION: DO VI AND SCORE-LIFE MATCH?")
    print("=" * 80)
    print(f"\nCorrelation:  r = {corr:.6f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"Mean diff:    {mean_diff:.4f}")
    print(f"Std of diff:  {np.std(diff):.4f}")
    print(f"Min diff:     {np.min(diff):.4f}")
    print(f"Max diff:     {np.max(diff):.4f}")

    if corr > 0.99:
        print("\n✅ EXCELLENT! Nearly perfect agreement (r > 0.99)")
        print("   Value functions match - ready to scale to other environments")
    elif corr > 0.95:
        print("\n✅ GOOD agreement (r > 0.95)")
        print("   Methods produce similar value functions")
    elif corr > 0.90:
        print("\n⚠️  Moderate agreement (r > 0.90)")
        print("   Some discrepancy - may need parameter tuning")
    else:
        print(f"\n❌ LOW agreement (r = {corr:.4f})")
        print("   Need to debug - check parameters and convergence")


if __name__ == "__main__":
    main()
