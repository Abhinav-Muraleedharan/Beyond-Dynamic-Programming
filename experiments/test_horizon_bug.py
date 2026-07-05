#!/usr/bin/env python
"""
Test if the shift is due to finite horizon in Score-Life.

HYPOTHESIS: Score-Life uses N=10 bits for action sequence,
which means only ~10 steps of lookahead. This truncates the
infinite-horizon value, making it less negative.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def test_finite_horizon_effect():
    """Test how N (action sequence length) affects the value estimate."""

    print("=" * 70)
    print("TESTING FINITE HORIZON EFFECT")
    print("=" * 70)

    state = 0.0
    gamma = 0.9

    print(f"\nState: {state}")
    print(f"Gamma: {gamma}")
    print("\nTesting different horizon lengths (N):\n")

    N_values = [5, 10, 15, 20, 30, 50]

    results = []

    for N in N_values:
        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=500,  # Moderate samples for speed
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Find max
        l_values = np.linspace(0.01, 0.99, 20)
        scores = [score_func.compute_fractal(l) for l in l_values]
        V_estimate = max(scores)

        # Theoretical discount after N steps
        discount_after_N = gamma**N
        remaining_fraction = discount_after_N / (1 - gamma)  # Geometric series

        results.append({
            'N': N,
            'V': V_estimate,
            'gamma^N': discount_after_N,
            'remaining': remaining_fraction
        })

        print(f"  N={N:2d}: V={V_estimate:8.2f}, "
              f"γ^N={discount_after_N:.4f}, "
              f"remaining weight={remaining_fraction:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print(f"\nFor γ={gamma}:")
    print(f"  After 10 steps: γ^10 = {gamma**10:.4f} (65% of value captured)")
    print(f"  After 20 steps: γ^20 = {gamma**20:.4f} (88% of value captured)")
    print(f"  After 30 steps: γ^30 = {gamma**30:.4f} (96% of value captured)")
    print(f"  After 50 steps: γ^50 = {gamma**50:.4f} (99.5% of value captured)")

    # Check if V increases (becomes more negative) with N
    V_values = [r['V'] for r in results]
    is_decreasing = all(V_values[i] >= V_values[i+1] for i in range(len(V_values)-1))

    print(f"\nV(X) becomes more negative as N increases: {is_decreasing}")

    if is_decreasing:
        print("\n✅ CONFIRMED: The shift is due to finite horizon!")
        print(f"\nWith N=10, we only capture ~65% of infinite-horizon value.")
        print(f"This makes V less negative, causing the ~300 unit shift!")

        # Estimate what V should be at infinite horizon
        V_10 = results[1]['V']  # N=10
        V_50 = results[-1]['V']  # N=50 (approximately infinite)

        print(f"\nValue estimates:")
        print(f"  N=10:  V ≈ {V_10:.2f}")
        print(f"  N=50:  V ≈ {V_50:.2f} (closer to infinite horizon)")
        print(f"  Shift: {V_50 - V_10:.2f}")
    else:
        print("\n❌ Unexpected: V doesn't decrease monotonically with N")
        print("   There might be another bug...")

    print("\n" + "=" * 70)
    print("SOLUTION")
    print("=" * 70)
    print("""
To fix the shift:
1. Increase N to 30-50 for γ=0.9 to capture >95% of infinite-horizon value
2. For general γ, use N such that γ^N < 0.01 (captures >99% of value)
3. Trade-off: Larger N = more accurate but slower computation
    """)


if __name__ == "__main__":
    test_finite_horizon_effect()
