#!/usr/bin/env python
"""
Debug the discount factor application in Score function.

Check if gamma is being applied correctly to each reward.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def manual_score_computation(state, gamma, N, l_value, num_samples=10):
    """
    Manually compute the Score function to verify the formula.
    """
    env = BusEngineEnvironment()
    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=1,  # Single sample for debugging
        reference_state=np.array([state])
    )

    # Get action sequence
    action_sequence = slp._real_to_action_sequence_base(l_value, N, 2)
    print(f"\nAction sequence for l={l_value}: {action_sequence}")
    print(f"Length: {len(action_sequence)}")

    # Manual computation
    env.set_state(state)
    print(f"\nInitial state: {env.current_state()}")

    manual_R = 0
    print(f"\nStep-by-step computation:")
    print(f"{'Step':<6} {'Action':<8} {'Reward':<10} {'Discount':<10} {'Term':<15} {'Cumulative':<15}")
    print("-" * 75)

    for i in range(len(action_sequence) - 1):
        action = int(action_sequence[i + 1])  # Note the i+1 indexing!

        result = env.step(action)
        if len(result) == 5:
            next_state, reward, done, truncated, _ = result
        else:
            next_state, reward, done, truncated = result

        discount = gamma ** i
        term = discount * reward
        manual_R += term

        print(f"{i:<6} {action:<8} {reward:<10.2f} {discount:<10.4f} {term:<15.2f} {manual_R:<15.2f}")

        if done or truncated:
            print("  (Episode ended)")
            break

    print(f"\nFinal manual score: {manual_R:.2f}")

    # Now compute using SLP's S function
    slp_score = slp.S(l_value, np.array([state]))
    print(f"SLP S() function:   {slp_score:.2f}")
    print(f"Difference:         {abs(manual_R - slp_score):.2f}")

    return manual_R, slp_score


def check_discount_formula():
    """Check if the discount formula is correct."""

    print("=" * 70)
    print("DEBUGGING DISCOUNT FACTOR APPLICATION")
    print("=" * 70)

    state = 0.0
    gamma = 0.9
    N = 10
    l_value = 0.5

    manual_score, slp_score = manual_score_computation(state, gamma, N, l_value)

    print("\n" + "=" * 70)
    print("EXPECTED vs ACTUAL")
    print("=" * 70)

    print(f"""
For infinite-horizon value function:
  V(s) = r_0 + γ*r_1 + γ²*r_2 + γ³*r_3 + ...

Where:
  - r_0 is reward from FIRST action at state s
  - Discounted by γ^0 = 1 (no discount for immediate reward)
  - r_1 discounted by γ^1
  - r_2 discounted by γ^2
  - etc.

Current implementation:
  for i in range(N-1):
      action = action_sequence[i+1]  # ← Takes from i+1
      reward = step(action)
      R = gamma^i * reward + R

  When i=0: Takes action_sequence[1], multiplies by γ^0 = 1 ✓
  When i=1: Takes action_sequence[2], multiplies by γ^1 ✓

This LOOKS correct...
    """)

    # Check if the issue is in Faber-Schauder reconstruction
    print("\n" + "=" * 70)
    print("CHECKING FABER-SCHAUDER RECONSTRUCTION")
    print("=" * 70)

    env = BusEngineEnvironment()
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=100,
        reference_state=np.array([state])
    )

    # Direct S(l) evaluations
    test_l_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\nDirect S(l) evaluations:")
    for l in test_l_values:
        score = slp.S(l, np.array([state]))
        print(f"  S({l:.2f}) = {score:8.2f}")

    # Faber-Schauder reconstruction
    print(f"\nFaber-Schauder reconstruction:")
    score_func = slp._compute_faber_schauder_coefficients()

    print(f"  a_0 = {score_func.a_0:.2f}")
    print(f"  a_1 = {score_func.a_1:.2f}")

    for l in test_l_values:
        reconstructed = score_func.compute_fractal(l)
        direct = slp.S(l, np.array([state]))
        error = abs(reconstructed - direct)
        print(f"  S({l:.2f}): Direct={direct:8.2f}, Reconstructed={reconstructed:8.2f}, Error={error:8.2f}")

    print("\n" + "=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print("""
If reconstruction error is large:
  → Bug in Faber-Schauder coefficient computation or reconstruction

If reconstruction matches but values oscillate with N:
  → Bug in how action sequences are generated/indexed for different N

If direct S(l) also oscillates with N:
  → Bug in the core S() function (discount or reward accumulation)
    """)


if __name__ == "__main__":
    check_discount_formula()
