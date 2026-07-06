#!/usr/bin/env python
"""
Compare INCORRECT VI (using V(E[next])) vs CORRECT VI (using E[V(next)]).

The fundamental issue: V is nonlinear, so V(E[X]) ≠ E[V(X)] (Jensen's inequality)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment


def vi_incorrect(gamma=0.9, n_states=30, tolerance=1e-6):
    """INCORRECT VI: Uses V(E[next_state]) instead of E[V(next_state)]."""

    print("=" * 80)
    print("INCORRECT VI: V(E[next_state])")
    print("=" * 80)

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Expected next state
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            expected_next = min(state + expected_delta, max_mileage)

            # Cost on expected next state
            operating_cost = -0.01 * expected_next

            # V at expected next state (INCORRECT!)
            next_idx = np.argmin(np.abs(states - expected_next))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next  # WRONG!
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        if np.max(np.abs(V_new - V)) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break
        V = V_new.copy()

    print(f"  V(0) = {V[0]:.6f}")
    print(f"  V(5000) = {V[np.argmin(np.abs(states - 5000))]:.6f}")
    return states, V


def vi_correct(gamma=0.9, n_states=30, tolerance=1e-6, n_samples=100):
    """CORRECT VI: Uses E[V(next_state)] via Monte Carlo sampling."""

    print("\n" + "=" * 80)
    print("CORRECT VI: E[V(next_state)] via Monte Carlo")
    print("=" * 80)

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    # Environment for sampling transitions
    env = BusEngineEnvironment(max_state=max_mileage)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Q(keep): Sample many next states and average V(next)
            V_next_samples = []
            cost_samples = []

            for _ in range(n_samples):
                env.set_state(state)
                next_state, reward, _, _, _ = env.step(0)  # keep action

                # Interpolate V at this sampled next state
                next_idx = np.argmin(np.abs(states - next_state[0]))
                V_next_samples.append(V[next_idx])
                cost_samples.append(reward)

            # Expected cost and expected V(next)
            expected_cost = np.mean(cost_samples)
            expected_V_next = np.mean(V_next_samples)

            Q_keep = expected_cost + gamma * expected_V_next  # CORRECT!

            # Q(replace): deterministic, so no sampling needed
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        if np.max(np.abs(V_new - V)) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break
        V = V_new.copy()

    print(f"  V(0) = {V[0]:.6f}")
    print(f"  V(5000) = {V[np.argmin(np.abs(states - 5000))]:.6f}")
    return states, V


def main():
    gamma = 0.9
    n_states = 30

    print("\nCOMPARING VI IMPLEMENTATIONS")
    print("="*80)
    print("\nBellman equation: V(s) = max_a [E[R(s,a,s')] + γ * E[V(s')]]")
    print("\nINCORRECT implementation:")
    print("  Q_keep = E[R] + γ * V(E[next_state])")
    print("           Uses V at the EXPECTED next state")
    print("\nCORRECT implementation:")
    print("  Q_keep = E[R] + γ * E[V(next_state)]")
    print("           Averages V over SAMPLED next states")
    print()

    # Run both
    states_incorrect, V_incorrect = vi_incorrect(gamma, n_states)
    states_correct, V_correct = vi_correct(gamma, n_states, n_samples=1000)

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    diff = V_correct - V_incorrect

    print(f"\nValue function differences:")
    print(f"  At state 0:")
    print(f"    Incorrect VI: {V_incorrect[0]:.6f}")
    print(f"    Correct VI:   {V_correct[0]:.6f}")
    print(f"    Difference:   {diff[0]:.6f}")

    idx_5000 = np.argmin(np.abs(states_correct - 5000))
    print(f"\n  At state 5000:")
    print(f"    Incorrect VI: {V_incorrect[idx_5000]:.6f}")
    print(f"    Correct VI:   {V_correct[idx_5000]:.6f}")
    print(f"    Difference:   {diff[idx_5000]:.6f}")

    print(f"\n  Mean difference: {np.mean(diff):.6f}")
    print(f"  Max |difference|: {np.max(np.abs(diff)):.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if np.abs(np.mean(diff)) > 1.0:
        print("\n⚠️  SIGNIFICANT DIFFERENCE!")
        print(f"   Mean difference: {np.mean(diff):.2f} units")
        print("\n   This explains the VI vs Score-Life mismatch!")
        print("   Score-Life uses the CORRECT Bellman equation (samples transitions)")
        print("   Our VI uses INCORRECT approximation (uses expected transition)")
        print("\n   Due to Jensen's inequality: V(E[X]) ≠ E[V(X)]")
    else:
        print("\n✅ Difference is small")
        print("   The approximation V(E[next]) ≈ E[V(next)] is reasonable")


if __name__ == "__main__":
    main()
