#!/usr/bin/env python
"""
Test Score-Life value function with HIGH Monte Carlo samples
to see if non-monotonicity is due to variance.
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


def test_state_initialization():
    """Verify that Score function actually starts from the correct state."""
    print("=" * 70)
    print("TESTING STATE INITIALIZATION")
    print("=" * 70)

    test_state = 2525.0
    gamma = 0.9

    env = BusEngineEnvironment()
    env.set_state(test_state)

    print(f"\nSet environment to state: {test_state}")
    print(f"Environment current state: {env.current_state()}")

    # Create SLP with this state
    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=10, j_max=5,
        num_samples=100,
        reference_state=np.array([test_state])
    )

    print(f"SLP reference_state: {slp.reference_state}")

    # Check what state is used when computing Score
    # The Score function should reset to reference_state
    print("\nVerifying Score computation starts from reference_state...")

    # Compute a score and check
    l_test = 0.5
    score = slp.S(l_test, np.array([test_state]))

    print(f"✓ Score computed for state {test_state}: {score:.2f}")
    print(f"Environment state after Score computation: {env.current_state()}")


def compute_value_with_varying_samples(state, gamma, sample_counts):
    """Compute V(state) with different Monte Carlo sample counts."""
    print(f"\n{'='*70}")
    print(f"Testing state {state} with varying sample counts")
    print(f"{'='*70}")

    results = []

    for num_samples in sample_counts:
        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=10, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Search for max
        l_values = np.linspace(0.01, 0.99, 50)  # Higher resolution
        scores = [score_func.compute_fractal(l) for l in l_values]

        V_estimate = max(scores)
        optimal_l = l_values[np.argmax(scores)]

        results.append({
            'num_samples': num_samples,
            'V': V_estimate,
            'l_star': optimal_l
        })

        print(f"  num_samples={num_samples:4d} -> V={V_estimate:8.2f}, l*={optimal_l:.4f}")

    return results


def test_monotonicity_with_high_samples(gamma, num_samples):
    """Test if value function is monotonic with high sample count."""
    print(f"\n{'='*70}")
    print(f"TESTING MONOTONICITY with {num_samples} samples")
    print(f"{'='*70}")

    test_states = np.array([0, 500, 1000, 1500, 2000, 2525, 3000, 4000, 5000, 7500, 10000])
    V_estimates = []

    for state in test_states:
        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=10, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Search for max with high resolution
        l_values = np.linspace(0.01, 0.99, 50)
        scores = [score_func.compute_fractal(l) for l in l_values]

        V_estimate = max(scores)
        V_estimates.append(V_estimate)

        print(f"  State {state:6.0f} -> V = {V_estimate:8.2f}")

    V_estimates = np.array(V_estimates)

    # Check monotonicity
    is_decreasing = all(V_estimates[i] >= V_estimates[i+1] for i in range(len(V_estimates)-1))

    print(f"\nMonotonicity check:")
    print(f"  Is monotonically decreasing: {is_decreasing}")

    if not is_decreasing:
        # Find where it violates
        for i in range(len(V_estimates)-1):
            if V_estimates[i] < V_estimates[i+1]:
                print(f"  ⚠️  Violation at state {test_states[i]:.0f} -> {test_states[i+1]:.0f}")
                print(f"      V({test_states[i]:.0f}) = {V_estimates[i]:.2f} < V({test_states[i+1]:.0f}) = {V_estimates[i+1]:.2f}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(test_states, V_estimates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('V(X)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Value Function (num_samples={num_samples})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Slope
    dV = np.diff(V_estimates) / np.diff(test_states)
    ax2.plot(test_states[:-1], dV, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('dV/dX', fontsize=12, fontweight='bold')
    ax2.set_title('Value Function Slope', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'results/value_function_samples{num_samples}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    return test_states, V_estimates, is_decreasing


def main():
    gamma = 0.9

    # Test 1: Verify state initialization
    test_state_initialization()

    # Test 2: Compare convergence with different sample counts
    print("\n" + "=" * 70)
    print("CONVERGENCE TEST: Effect of num_samples")
    print("=" * 70)

    sample_counts = [50, 100, 200, 500, 1000]
    test_state = 2525  # VI threshold for gamma=0.9

    results = compute_value_with_varying_samples(test_state, gamma, sample_counts)

    # Check stability
    V_values = [r['V'] for r in results]
    V_std = np.std(V_values)
    V_mean = np.mean(V_values)

    print(f"\nConvergence statistics:")
    print(f"  Mean V: {V_mean:.2f}")
    print(f"  Std V: {V_std:.2f}")
    print(f"  CV: {abs(V_std/V_mean)*100:.1f}%")

    if abs(V_std/V_mean) < 0.05:
        print("  ✓ Good convergence (CV < 5%)")
    elif abs(V_std/V_mean) < 0.10:
        print("  ⚠️  Moderate convergence (5% < CV < 10%)")
    else:
        print("  ❌ Poor convergence (CV > 10%)")

    # Test 3: Full monotonicity test with high samples
    print("\n" + "=" * 70)
    print("FULL MONOTONICITY TEST")
    print("=" * 70)

    # Test with progressively higher sample counts
    for num_samples in [100, 500, 1000]:
        states, V, is_monotonic = test_monotonicity_with_high_samples(gamma, num_samples)

        if is_monotonic:
            print(f"\n✅ SUCCESS with {num_samples} samples - Value function is monotonic!")
            break
        else:
            print(f"\n⚠️  Still non-monotonic with {num_samples} samples")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If monotonicity is achieved with high sample counts:
  → Non-monotonicity was due to Monte Carlo variance
  → Solution: Use higher num_samples for policy extraction

If monotonicity is NOT achieved even with 1000+ samples:
  → Fundamental issue with Score-Life formulation
  → May not be suitable for this problem structure
    """)


if __name__ == "__main__":
    main()
