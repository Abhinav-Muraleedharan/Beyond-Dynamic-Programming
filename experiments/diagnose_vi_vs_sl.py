#!/usr/bin/env python
"""
Diagnose why VI and Score-Life still don't match.

Check:
1. Is VI computing expected values correctly?
2. Is Score-Life's Faber-Schauder reconstruction accurate?
3. Do direct S() evaluations match Faber-Schauder?
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def test_vi_expected_value():
    """Test if VI expected value computation is correct."""

    print("=" * 70)
    print("TESTING VI EXPECTED VALUE COMPUTATION")
    print("=" * 70)

    state = 1000.0
    gamma = 0.9

    # VI's approach: Use expected next state
    p, q = 0.1, 0.3
    expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
    expected_next = state + expected_delta

    print(f"\nVI approach (using E[next state]):")
    print(f"  Current state: {state}")
    print(f"  E[delta]: {expected_delta}")
    print(f"  E[next state]: {expected_next}")
    print(f"  Cost on E[next]: -{0.01 * expected_next:.2f}")

    # Correct approach: Sample many next states and average
    np.random.seed(42)
    env = BusEngineEnvironment()

    sampled_costs = []
    sampled_next_states = []

    for _ in range(10000):
        env.set_state(state)
        next_state, reward, _, _, _ = env.step(0)  # keep action
        sampled_costs.append(reward)
        sampled_next_states.append(next_state[0])

    avg_cost = np.mean(sampled_costs)
    avg_next_state = np.mean(sampled_next_states)

    print(f"\nMonte Carlo sampling (10000 samples):")
    print(f"  Average next state: {avg_next_state:.2f}")
    print(f"  Average cost: {avg_cost:.2f}")
    print(f"  Expected cost on avg next: -{0.01 * avg_next_state:.2f}")

    print(f"\nComparison:")
    print(f"  E[next state]: VI={expected_next:.2f}, MC={avg_next_state:.2f}, "
          f"diff={abs(expected_next - avg_next_state):.2f}")
    print(f"  E[cost]: VI={-0.01*expected_next:.2f}, MC={avg_cost:.2f}, "
          f"diff={abs(-0.01*expected_next - avg_cost):.2f}")

    if abs(expected_next - avg_next_state) < 100:
        print("\n  ✓ VI expected value computation is reasonable")
    else:
        print("\n  ✗ VI expected value has significant error!")


def test_faber_schauder_reconstruction():
    """Test if Faber-Schauder reconstruction is accurate."""

    print("\n" + "=" * 70)
    print("TESTING FABER-SCHAUDER RECONSTRUCTION")
    print("=" * 70)

    state = 0.0
    gamma = 0.9
    N = 50

    env = BusEngineEnvironment()
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=500,
        reference_state=np.array([state])
    )

    # Compute Faber-Schauder coefficients
    score_func = slp._compute_faber_schauder_coefficients()

    # Test reconstruction accuracy at various l values
    test_l_values = np.linspace(0.0, 1.0, 11)

    print(f"\nTesting reconstruction at {len(test_l_values)} points:")
    print(f"{'l':<8} {'Direct S(l)':<15} {'Reconstructed':<15} {'Error':<12} {'% Error'}")
    print("-" * 70)

    errors = []
    for l in test_l_values:
        # Direct evaluation
        direct = slp.S(l, np.array([state]))

        # Faber-Schauder reconstruction
        reconstructed = score_func.compute_fractal(l)

        error = abs(direct - reconstructed)
        pct_error = (error / abs(direct)) * 100 if direct != 0 else 0

        errors.append(error)

        print(f"{l:<8.2f} {direct:<15.2f} {reconstructed:<15.2f} {error:<12.2f} {pct_error:.2f}%")

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"\nReconstruction accuracy:")
    print(f"  Average error: {avg_error:.2f}")
    print(f"  Maximum error: {max_error:.2f}")

    if max_error < 100:
        print("  ✓ Reconstruction is accurate")
    else:
        print("  ✗ Reconstruction has significant errors!")


def compare_direct_s_with_vi():
    """Compare direct S() evaluations with VI."""

    print("\n" + "=" * 70)
    print("COMPARING DIRECT S() WITH VI")
    print("=" * 70)

    gamma = 0.9
    test_states = [0, 1000, 2000, 5000, 10000]

    print(f"\nComparing V(state) estimates:")
    print(f"{'State':<10} {'VI V(state)':<15} {'SL max S(l,state)':<20} {'Ratio SL/VI'}")
    print("-" * 65)

    for state in test_states:
        # VI value (simplified single-step lookahead)
        p, q = 0.1, 0.3
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
        expected_next = state + expected_delta
        immediate_cost = -0.01 * expected_next

        # Very rough VI estimate (just one-step lookahead)
        vi_estimate = immediate_cost  # Simplified

        # Score-Life estimate
        env = BusEngineEnvironment()
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=50, j_max=5,
            num_samples=100,  # Fewer samples for speed
            reference_state=np.array([state])
        )

        score_func = slp._compute_faber_schauder_coefficients()
        l_values = np.linspace(0.01, 0.99, 20)
        scores = [score_func.compute_fractal(l) for l in l_values]
        sl_estimate = max(scores)

        ratio = sl_estimate / vi_estimate if vi_estimate != 0 else 0

        print(f"{state:<10.0f} {vi_estimate:<15.2f} {sl_estimate:<20.2f} {ratio:.3f}")


if __name__ == "__main__":
    test_vi_expected_value()
    test_faber_schauder_reconstruction()
    compare_direct_s_with_vi()
