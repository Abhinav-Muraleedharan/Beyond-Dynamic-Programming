#!/usr/bin/env python
"""
Compare VI and Score-Life value functions with BOTH bugs fixed:
1. Truncation bug fixed (reset _current_step)
2. Using N=50 for proper infinite-horizon approximation
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


def compute_vi_value_function(gamma=0.9, n_states=30):
    """Compute VI value function."""
    from experiments.compare_value_functions import compute_vi_value_function
    return compute_vi_value_function(gamma, n_states)


def compute_sl_value_function_fixed(gamma=0.9, n_states=30, N=50, num_samples=500):
    """
    Compute Score-Life value function with FIXED parameters.

    N=50 for infinite-horizon approximation (γ^50 < 0.01)
    num_samples=500 for stable estimates
    """
    print(f"\nComputing Score-Life value function:")
    print(f"  N={N} (horizon length)")
    print(f"  num_samples={num_samples}")
    print(f"  n_states={n_states}\n")

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    V_sl = []

    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}")

        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()

        # Search for max
        l_values = np.linspace(0.01, 0.99, 30)
        scores = [score_func.compute_fractal(l) for l in l_values]

        V_sl.append(max(scores))

    print("  Done!\n")

    return states, np.array(V_sl)


def main():
    gamma = 0.9

    print("=" * 70)
    print(f"FINAL COMPARISON: VI vs SCORE-LIFE (BOTH BUGS FIXED)")
    print("=" * 70)

    # Compute VI
    print("\nComputing Value Iteration...")
    states_vi, V_vi, threshold_vi = compute_vi_value_function(gamma=gamma, n_states=30)

    # Compute Score-Life with N=50
    states_sl, V_sl = compute_sl_value_function_fixed(gamma=gamma, n_states=30, N=50, num_samples=500)

    # Import plotting function
    from experiments.compare_value_functions import create_comparison_plot

    create_comparison_plot(states_vi, V_vi, threshold_vi, states_sl, V_sl, gamma)

    # Print detailed comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nValue Iteration:")
    print(f"  V(0) = {V_vi[0]:.2f}")
    print(f"  V({states_vi[-1]:.0f}) = {V_vi[-1]:.2f}")
    print(f"  Range: {V_vi.max() - V_vi.min():.2f}")
    print(f"  Threshold: {threshold_vi:.0f} miles")

    print(f"\nScore-Life (N=50, num_samples=500):")
    print(f"  V(0) = {V_sl[0]:.2f}")
    print(f"  V({states_sl[-1]:.0f}) = {V_sl[-1]:.2f}")
    print(f"  Range: {V_sl.max() - V_sl.min():.2f}")

    # Correlation
    from scipy.interpolate import interp1d
    V_sl_interp = interp1d(states_sl, V_sl, kind='cubic')(states_vi)
    corr = np.corrcoef(V_vi, V_sl_interp)[0, 1]

    print(f"\nCorrelation: r = {corr:.4f}")

    # Compare magnitudes
    V_ratio = np.mean(V_sl) / np.mean(V_vi)
    print(f"Magnitude ratio (SL/VI): {V_ratio:.3f}")

    if corr > 0.95 and 0.9 < V_ratio < 1.1:
        print("\n✅ SUCCESS! Value functions match closely!")
    elif corr > 0.95:
        print(f"\n⚠️  Shape matches (r={corr:.3f}) but magnitude differs by {abs(1-V_ratio)*100:.1f}%")
    else:
        print(f"\n❌ Value functions still differ (r={corr:.3f})")


if __name__ == "__main__":
    main()
