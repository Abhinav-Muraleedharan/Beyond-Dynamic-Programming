#!/usr/bin/env python
"""
Detailed trace to find the cost accumulation difference between VI and Score-Life.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def trace_vi_value_computation(state=0.0, gamma=0.9, num_steps=5):
    """Trace VI value computation step by step."""

    print("=" * 80)
    print(f"TRACING VI VALUE COMPUTATION FOR STATE={state}")
    print("=" * 80)

    print("\nVI computes V(X) iteratively using Bellman equation:")
    print("  V_{k+1}(X) = max_a [R(X,a) + γ * V_k(next_state)]")

    # Simulate a few iterations
    states = np.linspace(0, 10000, 100)
    V = np.zeros(len(states))

    print(f"\nIteration 1 (base case, V=0 everywhere):")

    for iteration in range(num_steps):
        V_new = np.zeros(len(states))

        for i, s in enumerate(states):
            # Q(s, keep)
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(s + expected_delta, 10000)

            operating_cost = -0.01 * next_state

            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        V = V_new.copy()

        if iteration < 3:
            state_idx = np.argmin(np.abs(states - state))
            print(f"\nIteration {iteration + 1}:")
            print(f"  V({state}) = {V[state_idx]:.2f}")
            if state == 0:
                expected_next = 0 + expected_delta
                print(f"    Q_keep = -0.01 * {expected_next:.0f} + {gamma} * V({expected_next:.0f})")
                print(f"           = -{0.01 * expected_next:.2f} + {gamma} * {V[np.argmin(np.abs(states - expected_next))]:.2f}")
                print(f"           = {-0.01 * expected_next + gamma * V[np.argmin(np.abs(states - expected_next))]:.2f}")

    state_idx = np.argmin(np.abs(states - state))
    return V[state_idx]


def trace_score_life_computation(state=0.0, gamma=0.9, N=5, num_samples=100):
    """Trace Score-Life computation step by step."""

    print("\n" + "=" * 80)
    print(f"TRACING SCORE-LIFE COMPUTATION FOR STATE={state}")
    print("=" * 80)

    print("\nScore-Life computes S(l, X) by simulating N-step trajectories:")
    print("  S(l, X) = E[Σ_{i=0}^{N-1} γ^i * R_i]")

    # Use all "keep" actions (l=0)
    np.random.seed(42)
    env = BusEngineEnvironment()

    print(f"\nSimulating ONE trajectory with all 'keep' actions:")
    print(f"  Starting state: {state}")

    env.set_state(state)
    R = 0

    print(f"\n  Initial: R = 0")

    for i in range(N):
        action = 0  # keep
        state_before = env.current_state()
        next_state, reward, done, truncated, info = env.step(action)

        contribution = (gamma ** i) * reward
        R += contribution

        print(f"\n  Step {i}:")
        print(f"    State before: {state_before:.2f}")
        print(f"    Action: keep")
        print(f"    Next state: {next_state[0]:.2f}")
        print(f"    Reward: {reward:.2f}")
        print(f"    Discount: γ^{i} = {gamma**i:.4f}")
        print(f"    Contribution: {gamma**i:.4f} * {reward:.2f} = {contribution:.2f}")
        print(f"    Cumulative R: {R:.2f}")

        if done or truncated:
            print(f"    Episode ended!")
            break

    print(f"\n  Final R = {R:.2f}")

    return R


def check_initial_cost_hypothesis(state=0.0):
    """Check if there's an extra initial cost somewhere."""

    print("\n" + "=" * 80)
    print("CHECKING INITIAL COST HYPOTHESIS")
    print("=" * 80)

    # Check if environment returns a reward when setting state
    env = BusEngineEnvironment()

    print(f"\n1. Does set_state() cause any cost?")
    print(f"   Before: env.state = {env.state}")
    env.set_state(state)
    print(f"   After set_state({state}): env.state = {env.state}")
    print(f"   No reward/cost received from set_state()")

    print(f"\n2. Does Score-Life add initial cost before the loop?")
    print(f"   Looking at score_life_programming.py lines 224-250...")
    print(f"   Line 225: R = 0  (no initial cost)")
    print(f"   Line 226: self.env.set_state(X)  (just sets state)")
    print(f"   Line 228-246: for loop accumulating rewards")
    print(f"   Line 247: avg_R = avg_R + R")
    print(f"   → No initial cost added!")

    print(f"\n3. Does VI assume a cost for being in initial state?")
    print(f"   Looking at VI Bellman equation:")
    print(f"   V(X) = max_a [R(X,a,X') + γ*V(X')]")
    print(f"   The reward R(X,a,X') is the transition reward, not state cost")
    print(f"   → No initial state cost in VI either!")

    print(f"\n4. Could the issue be in the environment step() function?")

    env.set_state(state)
    print(f"\n   Starting at state={state}")
    next_state, reward, done, truncated, info = env.step(0)  # keep

    print(f"   After step(keep):")
    print(f"     Next state: {next_state[0]:.2f}")
    print(f"     Reward: {reward:.2f}")
    print(f"     Cost: {info['cost']:.2f}")
    print(f"     Cost formula: 0.01 * next_state = 0.01 * {next_state[0]:.2f} = {0.01 * next_state[0]:.2f}")
    print(f"     Reward = -cost = {reward:.2f}")
    print(f"   → Reward is based on NEXT state, not current state ✓")


def compute_theoretical_difference(gamma=0.9, N=50):
    """Compute theoretical difference between infinite and finite horizon."""

    print("\n" + "=" * 80)
    print("THEORETICAL DIFFERENCE: INFINITE vs FINITE HORIZON")
    print("=" * 80)

    # Assume steady-state: engine stays around some average mileage
    # Average cost per step
    avg_mileage = 7000  # rough estimate
    avg_cost_per_step = -0.01 * avg_mileage  # -70

    print(f"\nAssumptions:")
    print(f"  Average mileage in steady state: {avg_mileage}")
    print(f"  Average cost per step: {avg_cost_per_step:.2f}")

    # VI infinite horizon value
    vi_infinite = avg_cost_per_step / (1 - gamma)
    print(f"\nVI (infinite horizon):")
    print(f"  V ≈ cost/(1-γ) = {avg_cost_per_step:.2f} / {1-gamma}")
    print(f"    = {vi_infinite:.2f}")

    # Score-Life finite horizon value
    sl_finite = avg_cost_per_step * (1 - gamma**N) / (1 - gamma)
    print(f"\nScore-Life (N={N} horizon):")
    print(f"  S ≈ cost * (1-γ^N)/(1-γ)")
    print(f"    = {avg_cost_per_step:.2f} * (1-{gamma**N:.6f}) / {1-gamma}")
    print(f"    = {avg_cost_per_step:.2f} * {(1 - gamma**N):.6f} / {1-gamma}")
    print(f"    = {sl_finite:.2f}")

    # Difference
    diff = sl_finite - vi_infinite
    missing_tail = vi_infinite - sl_finite

    print(f"\nDifference:")
    print(f"  S - V = {sl_finite:.2f} - {vi_infinite:.2f} = {diff:.2f}")
    print(f"  Missing tail = {missing_tail:.2f}")

    print(f"\nBUT: Observed difference is ~105 units!")
    print(f"     This theoretical difference is only ~{abs(diff):.1f} units")
    print(f"     → Must be something else!")


def main():
    state = 0.0
    gamma = 0.9

    # Trace VI
    vi_value = trace_vi_value_computation(state, gamma, num_steps=5)

    # Trace Score-Life
    sl_value = trace_score_life_computation(state, gamma, N=5, num_samples=1)

    # Check initial cost hypothesis
    check_initial_cost_hypothesis(state)

    # Theoretical difference
    compute_theoretical_difference(gamma, N=50)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFor state={state}, γ={gamma}:")
    print(f"  VI value (5 iterations): {vi_value:.2f}")
    print(f"  Score-Life (5 steps, 1 sample): {sl_value:.2f}")
    print(f"\nNeed to investigate further to find the ~105 unit shift!")


if __name__ == "__main__":
    main()
