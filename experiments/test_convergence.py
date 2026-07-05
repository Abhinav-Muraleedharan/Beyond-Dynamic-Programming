#!/usr/bin/env python
"""
Test convergence of Score-Life value function with different resolutions.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def test_convergence_at_state(state, gamma=0.9, num_samples_list=[10, 50, 100], l_res_list=[10, 20, 50]):
    """Test how V(X) estimate changes with resolution."""

    print(f"\n{'='*70}")
    print(f"Testing convergence at state = {state} miles, γ = {gamma}")
    print(f"{'='*70}")

    results = []

    for num_samples in num_samples_list:
        for l_res in l_res_list:
            env = BusEngineEnvironment()
            slp = ScoreLifeProgramming(
                env, gamma=gamma, N=10, j_max=5,
                num_samples=num_samples,
                reference_state=np.array([0.0])
            )

            # Search for max using the SAME method as definitive comparison
            score_func = slp._compute_faber_schauder_coefficients()
            l_values = np.linspace(0.01, 0.99, l_res)
            scores = [score_func.compute_fractal(l) for l in l_values]

            max_score = max(scores)
            optimal_l = l_values[np.argmax(scores)]

            results.append({
                'num_samples': num_samples,
                'l_res': l_res,
                'V': max_score,
                'l_star': optimal_l
            })

            print(f"  samples={num_samples:3d}, l_res={l_res:2d} -> V={max_score:8.2f}, l*={optimal_l:.4f}")

    # Check stability
    V_values = [r['V'] for r in results]
    V_std = np.std(V_values)
    V_mean = np.mean(V_values)

    print(f"\nStability:")
    print(f"  V mean: {V_mean:.2f}")
    print(f"  V std: {V_std:.2f}")
    print(f"  Coefficient of variation: {abs(V_std/V_mean)*100:.1f}%")

    return results


def main():
    print("="*70)
    print("TESTING SCORE-LIFE CONVERGENCE")
    print("="*70)

    # Test at a few key states
    test_states = [0, 1000, 2525, 5000]  # 2525 is VI threshold for γ=0.9

    for state in test_states:
        test_convergence_at_state(
            state,
            gamma=0.9,
            num_samples_list=[20, 50, 100],
            l_res_list=[10, 20, 30]
        )

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("""
If V estimates are stable (low std), then Score-Life is converging.
If V estimates vary wildly, we need more samples or better search.

If even with high resolution the threshold differs from VI,
then Score-Life might be solving a fundamentally different problem
or there's an issue with how the Score function is computed.
    """)


if __name__ == "__main__":
    main()
