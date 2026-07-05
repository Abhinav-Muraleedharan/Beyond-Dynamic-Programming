#!/usr/bin/env python
"""
Diagnose why Score-Life value function differs from VI.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def get_vi_value_function(gamma):
    """Get VI value function from results."""
    # For VI, V(X) can be estimated from the optimal policy
    # If threshold is T, then:
    # - For X < T: V(X) = expected cost following optimal policy
    # - For X >= T: V(X) = -100 + V(0)

    with open('results/bus_engine_definitive_comparison.json', 'r') as f:
        results = json.load(f)

    for result in results:
        if abs(result['gamma'] - gamma) < 0.01:
            threshold = result['VI']['threshold']
            avg_reward = result['VI']['reward']
            return threshold, avg_reward

    return None, None


def compute_sl_value_at_states(states, gamma, num_samples=100):
    """Compute Score-Life V(X) for specific states."""
    V_estimates = []

    for state in states:
        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=10, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Search for maximum
        l_values = np.linspace(0.01, 0.99, 30)
        scores = [score_func.compute_fractal(l) for l in l_values]

        V_estimates.append(max(scores))

    return np.array(V_estimates)


def main():
    gamma = 0.9

    print("=" * 70)
    print(f"DIAGNOSING VALUE FUNCTION DIFFERENCE (γ={gamma})")
    print("=" * 70)

    # Get VI info
    vi_threshold, vi_reward = get_vi_value_function(gamma)
    print(f"\nValue Iteration:")
    print(f"  Threshold: {vi_threshold:.0f} miles")
    print(f"  Average reward: {vi_reward:.2f}")

    # Compute Score-Life V(X) at various states
    test_states = np.array([0, 500, 1000, 1500, 2000, 2525, 3000, 4000, 5000, 7500, 10000])

    print(f"\nComputing Score-Life V(X) at {len(test_states)} states...")
    V_sl = compute_sl_value_at_states(test_states, gamma, num_samples=100)

    print(f"\nValue Function Comparison:")
    print(f"{'State':<10} {'V_SL(X)':<15}")
    print("-" * 25)
    for state, v in zip(test_states, V_sl):
        marker = " <-- VI threshold" if abs(state - vi_threshold) < 100 else ""
        print(f"{state:<10.0f} {v:<15.2f}{marker}")

    # Check if V is monotonic
    is_decreasing = all(V_sl[i] >= V_sl[i+1] for i in range(len(V_sl)-1))
    is_increasing = all(V_sl[i] <= V_sl[i+1] for i in range(len(V_sl)-1))

    print(f"\nValue Function Properties:")
    print(f"  Is monotonically decreasing (expected): {is_decreasing}")
    print(f"  Is monotonically increasing: {is_increasing}")
    print(f"  Neither (non-monotonic): {not is_decreasing and not is_increasing}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: V(X)
    ax1.plot(test_states, V_sl, 'bo-', linewidth=2, markersize=8, label='Score-Life V(X)')
    ax1.axvline(vi_threshold, color='green', linestyle='--', linewidth=2,
               label=f'VI threshold={vi_threshold:.0f}')
    ax1.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('V(X)', fontsize=12, fontweight='bold')
    ax1.set_title('Score-Life Value Function', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: V(X) derivative (discrete)
    dV = np.diff(V_sl) / np.diff(test_states)
    ax2.plot(test_states[:-1], dV, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(vi_threshold, color='green', linestyle='--', linewidth=2,
               label=f'VI threshold={vi_threshold:.0f}')
    ax2.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('dV/dX (slope)', fontsize=12, fontweight='bold')
    ax2.set_title('Value Function Slope', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'results/value_function_diagnosis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    if is_decreasing:
        print("✅ V(X) is monotonically decreasing (as expected)")
        print("   Higher mileage → more negative value → worse state")
    elif not is_decreasing and not is_increasing:
        print("⚠️  V(X) is NON-MONOTONIC!")
        print("   This is unexpected for the bus engine problem")
        print("   Possible causes:")
        print("   1. Monte Carlo sampling variance")
        print("   2. Score function not converging to true V")
        print("   3. Fundamental difference in what Score-Life optimizes")
    else:
        print("❌ V(X) is INCREASING (opposite of expected!)")
        print("   This suggests Score-Life is computing something different")

    print("\nIf V(X) is non-monotonic or increasing, the extracted policy")
    print("will not match VI because the underlying value function is wrong.")


if __name__ == "__main__":
    main()
