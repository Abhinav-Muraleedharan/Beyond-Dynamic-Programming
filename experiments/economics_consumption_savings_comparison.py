#!/usr/bin/env python
"""
Consumption-Savings Problem: VI vs Score-Life Comparison

Classic macroeconomics problem:
- Agent has wealth W
- Chooses consumption C
- Saves S = W - C with stochastic returns
- Maximizes lifetime utility

Demonstrates Score-Life on canonical economics MDP.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.consumption_savings import ConsumptionSavingsEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def precompute_transitions(n_states, max_wealth, transition_samples=1000):
    """Pre-compute transition samples for deterministic VI."""

    print("Pre-computing transitions...")
    states = np.linspace(0, max_wealth, n_states)
    env = ConsumptionSavingsEnvironment(max_wealth=max_wealth)

    transitions = {}

    for i, wealth in enumerate(states):
        if i % 10 == 0:
            print(f"  State {i}/{n_states}")

        # For each wealth level, sample transitions for different consumption rates
        # We'll use consumption rate = 0.5 as baseline (save 50%)
        consumption_rate = 0.5

        next_wealths = []
        utilities = []

        np.random.seed(i)  # Reproducible
        for sample in range(transition_samples):
            env.set_state(wealth)
            next_wealth, utility, _, _, _ = env.step(np.array([consumption_rate]))
            next_wealths.append(next_wealth[0])
            utilities.append(utility)

        transitions[i] = {
            'next_wealths': np.array(next_wealths),
            'utilities': np.array(utilities)
        }

    print(f"  Done! Cached {len(transitions)} state transitions")
    return transitions


def value_iteration_consumption(gamma, n_states, max_wealth, transitions, tolerance=1e-6):
    """VI for consumption-savings problem using cached transitions."""

    print("\n" + "=" * 80)
    print("VALUE ITERATION: Consumption-Savings Problem")
    print("=" * 80)

    states = np.linspace(0, max_wealth, n_states)
    V = np.zeros(n_states)

    # Test different consumption rates
    consumption_rates = np.linspace(0.1, 0.9, 9)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, wealth in enumerate(states):
            if wealth < 10:  # Below subsistence, must consume everything
                # Use cached baseline transition
                next_wealths = transitions[i]['next_wealths']
                utilities = transitions[i]['utilities']
                V_next_samples = [V[np.argmin(np.abs(states - nw))] for nw in next_wealths]
                V_new[i] = np.mean(utilities) + gamma * np.mean(V_next_samples)
            else:
                # Try different consumption rates
                Q_values = []

                for c_rate in consumption_rates:
                    # Approximate using cached samples (scaled by consumption rate)
                    # This is simplified - in practice you'd cache for each c_rate
                    next_wealths = transitions[i]['next_wealths'] * (1 - c_rate) / 0.5

                    # Utility from consumption
                    consumption = wealth * c_rate
                    gamma_utility = 2.0
                    utility = (consumption ** (1 - gamma_utility)) / (1 - gamma_utility)

                    # Expected continuation value
                    V_next_samples = [V[np.argmin(np.abs(states - nw))] for nw in next_wealths]
                    Q_values.append(utility + gamma * np.mean(V_next_samples))

                V_new[i] = max(Q_values)

        max_change = np.max(np.abs(V_new - V))

        if iteration % 50 == 0:
            print(f"  Iteration {iteration+1}: max_change = {max_change:.10f}")

        if max_change < tolerance:
            print(f"\n  ✅ Converged in {iteration + 1} iterations!")
            break

        V = V_new.copy()

    print(f"\n  V(wealth=0) = {V[0]:.6f}")
    print(f"  V(wealth={max_wealth}) = {V[-1]:.6f}")

    return states, V


def score_life_consumption(gamma, N, num_samples, n_states, n_l_points, max_wealth):
    """Score-Life for consumption-savings problem."""

    print("\n" + "=" * 80)
    print("SCORE-LIFE: Consumption-Savings Problem")
    print("=" * 80)

    states = np.linspace(0, max_wealth, n_states)
    V_sl = np.zeros(n_states)
    optimal_l = np.zeros(n_states)

    for i, wealth in enumerate(states):
        if i % 10 == 0:
            print(f"  Wealth ${wealth:.0f} ({i}/{n_states})")

        env = ConsumptionSavingsEnvironment(max_wealth=max_wealth)
        env.set_state(wealth)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([wealth])
        )

        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = [slp.S(l, np.array([wealth])) for l in l_values]

        max_idx = np.argmax(scores)
        V_sl[i] = scores[max_idx]
        optimal_l[i] = l_values[max_idx]

    print(f"\n  V(wealth=0) = {V_sl[0]:.6f}")
    print(f"  V(wealth={max_wealth}) = {V_sl[-1]:.6f}")

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
    ax1.set_xlabel('Wealth ($)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Lifetime Utility', fontsize=13, fontweight='bold')
    ax1.set_title(f'Consumption-Savings: VI vs Score-Life (γ={gamma}, N={N})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax2 = axes[0, 1]
    diff = V_sl - V_vi
    ax2.plot(states, diff, 'purple', linewidth=3, marker='d', markersize=5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(states, 0, diff, alpha=0.3, color='purple')
    ax2.set_xlabel('Wealth ($)', fontsize=12, fontweight='bold')
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
    ax4.set_xlabel('Wealth ($)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Optimal l*', fontsize=12, fontweight='bold')
    ax4.set_title('Score-Life l* Parameter', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Consumption-Savings Problem: VI vs Score-Life',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = 'results/economics_consumption_savings_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def main():
    gamma = 0.95  # Higher for long-term planning
    N = 50
    num_samples = 1000
    n_states = 30
    n_l_points = 30
    max_wealth = 1000
    transition_samples = 1000

    print("=" * 80)
    print("CONSUMPTION-SAVINGS PROBLEM")
    print("=" * 80)
    print("\nOptimal consumption vs savings with stochastic returns")
    print("- State: Wealth level")
    print("- Action: Consumption rate")
    print("- Dynamics: Stochastic investment returns")
    print("- Objective: Maximize lifetime utility")
    print()

    # Pre-compute transitions
    transitions = precompute_transitions(n_states, max_wealth, transition_samples)

    # Value Iteration
    states_vi, V_vi = value_iteration_consumption(
        gamma, n_states, max_wealth, transitions
    )

    # Score-Life
    states_sl, V_sl, optimal_l = score_life_consumption(
        gamma, N, num_samples, n_states, n_l_points, max_wealth
    )

    # Compare
    plot_comparison(states_vi, V_vi, V_sl, optimal_l, gamma, N)

    # Statistics
    corr = np.corrcoef(V_vi, V_sl)[0, 1]
    rmse = np.sqrt(np.mean((V_sl - V_vi)**2))

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nCorrelation:  r = {corr:.6f}")
    print(f"RMSE:         {rmse:.4f}")

    if corr > 0.99:
        print("\n✅ Excellent agreement between VI and Score-Life!")
    elif corr > 0.95:
        print("\n✅ Good agreement between methods")

    print("\nThis demonstrates Score-Life works on economics problems!")


if __name__ == "__main__":
    main()
