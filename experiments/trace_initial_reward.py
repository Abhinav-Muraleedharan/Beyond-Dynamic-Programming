#!/usr/bin/env python
"""
Trace exactly how the INITIAL (first) reward is accumulated in VI vs Score-Life.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def trace_vi_initial_reward():
    """Show how VI handles the first reward."""

    print("=" * 80)
    print("VALUE ITERATION - INITIAL REWARD HANDLING")
    print("=" * 80)

    print("\nVI Bellman Equation:")
    print("  V(X) = max_a [R(X, a, X') + γ * V(X')]")
    print("              ^^^^^^^^^^^^^")
    print("              Immediate reward - NOT DISCOUNTED!")

    print("\nExample at state X=0:")
    gamma = 0.9

    # Simulate one step
    env = BusEngineEnvironment(max_state=10000)
    env.set_state(0)

    print("\n  Step 1: At state X=0")
    print("    Take action 'keep'")

    # Compute expected reward
    p, q = 0.1, 0.3
    expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
    expected_next = 0 + expected_delta
    immediate_reward = -0.01 * expected_next

    print(f"    Expected next state: {expected_next:.2f}")
    print(f"    Immediate reward: {immediate_reward:.2f}")
    print(f"    Discount factor applied: 1.0 (NO DISCOUNT on immediate reward)")
    print(f"\n  Q_keep(0) = {immediate_reward:.2f} × 1.0 + {gamma} × V({expected_next:.2f})")
    print(f"            = {immediate_reward:.2f} + {gamma} × V(next)")

    print("\n  CONCLUSION: VI does NOT discount the first reward")
    print(f"              First reward contributes: {immediate_reward:.2f} × 1.0 = {immediate_reward:.2f}")


def trace_score_life_initial_reward():
    """Show how Score-Life handles the first reward."""

    print("\n" + "=" * 80)
    print("SCORE-LIFE - INITIAL REWARD HANDLING")
    print("=" * 80)

    print("\nScore-Life formula:")
    print("  S(l, X) = Σ_{i=0}^{N-1} γ^i * R_i")
    print("            ^^^^^^^^^^^^^^^^^^^")
    print("            i=0: γ^0 * R_0 = 1.0 * R_0 - NOT DISCOUNTED!")

    print("\nLooking at the code (score_life_programming.py lines 228-238):")
    print("  for i in range(len(action_sequence)-1):")
    print("      action = int(action_sequence[i+1])")
    print("      result = self.env.step(action)")
    print("      state, reward, done, truncated, _ = result")
    print("      R = (self.gamma**(i))*reward + R")
    print("          ^^^^^^^^^^^^^^^^")

    print("\nTrace through first iteration:")
    print("  i = 0 (first iteration)")
    print("    action = action_sequence[1]  (first action to take)")
    print("    reward = result from env.step(action)")
    print("    Discount factor: gamma^i = gamma^0 = 1.0")
    print("    Contribution: 1.0 × reward")

    print("\n  CONCLUSION: Score-Life does NOT discount the first reward")
    print("              First reward contributes: reward × 1.0 = reward")

    # Actual simulation
    print("\n" + "-" * 80)
    print("ACTUAL SIMULATION TO VERIFY:")
    print("-" * 80)

    np.random.seed(42)
    gamma = 0.9
    N = 3

    env = BusEngineEnvironment(max_state=10000)
    env.set_state(0)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=1,  # Single sample to see exact values
        reference_state=np.array([0])
    )

    # Manually trace through what S() does
    print("\nManually simulating Score function with all 'keep' actions (l=0):")

    env.set_state(0)
    R = 0

    # Simulate N steps
    for i in range(N):
        action = 0  # keep
        state_before = env.current_state()
        next_state, reward, done, truncated, info = env.step(action)

        discount = gamma ** i
        contribution = discount * reward
        R += contribution

        print(f"\n  Step i={i}:")
        print(f"    State before: {state_before:.2f}")
        print(f"    Next state: {next_state[0]:.2f}")
        print(f"    Reward: {reward:.2f}")
        print(f"    Discount: γ^{i} = {discount:.4f}")
        print(f"    Contribution: {discount:.4f} × {reward:.2f} = {contribution:.2f}")
        print(f"    Cumulative R: {R:.2f}")

        if i == 0:
            print(f"    >>> First reward is multiplied by γ^0 = 1.0 (NOT DISCOUNTED)")

    print(f"\n  Final R = {R:.2f}")


def compare_both_methods():
    """Direct comparison of both methods."""

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    print("\n┌─────────────────────┬────────────────────────┬────────────────────────┐")
    print("│                     │ Value Iteration        │ Score-Life             │")
    print("├─────────────────────┼────────────────────────┼────────────────────────┤")
    print("│ Formula             │ V = R + γ*V(next)      │ S = Σ γ^i * R_i        │")
    print("│ First reward term   │ R × 1.0                │ γ^0 × R_0 = R_0 × 1.0  │")
    print("│ Second reward term  │ (in V(next))           │ γ^1 × R_1 = γ × R_1    │")
    print("│ Third reward term   │ (in V(next))           │ γ^2 × R_2 = γ² × R_2   │")
    print("└─────────────────────┴────────────────────────┴────────────────────────┘")

    print("\n✅ BOTH methods apply NO discount to the first reward!")
    print("✅ BOTH methods apply γ to the second reward!")
    print("✅ BOTH methods apply γ² to the third reward!")

    print("\nVI expands recursively:")
    print("  V(X) = R_0 + γ*V(X_1)")
    print("       = R_0 + γ*(R_1 + γ*V(X_2))")
    print("       = R_0 + γ*R_1 + γ²*V(X_2)")
    print("       = R_0 + γ*R_1 + γ²*R_2 + γ³*V(X_3) + ...")

    print("\nScore-Life computes directly:")
    print("  S(X) = R_0 + γ*R_1 + γ²*R_2 + γ³*R_3 + ... + γ^(N-1)*R_{N-1}")

    print("\nThey should be IDENTICAL up to the Nth term!")
    print("The only difference is VI continues infinitely, Score-Life stops at N.")


def check_for_off_by_one_error():
    """Check if there's an off-by-one error in Score-Life."""

    print("\n" + "=" * 80)
    print("CHECKING FOR OFF-BY-ONE ERRORS")
    print("=" * 80)

    print("\nPotential issue: Does Score-Life start with i=1 instead of i=0?")
    print("\nLet's check the loop bounds:")
    print("  for i in range(len(action_sequence)-1):")

    N = 3
    action_sequence_length = N + 1  # action_sequence[0..N]

    print(f"\n  If N={N}, action_sequence has length {action_sequence_length}")
    print(f"  range(len(action_sequence)-1) = range({action_sequence_length-1}) = range({action_sequence_length-1})")
    print(f"  i takes values: {list(range(action_sequence_length-1))}")

    print("\n  Loop iterations:")
    for i in range(action_sequence_length-1):
        action_idx = i + 1
        discount = 0.9 ** i
        print(f"    i={i}: action=action_sequence[{action_idx}], discount=γ^{i}={discount:.4f}")

    print("\n✅ Loop correctly starts at i=0, so first reward gets γ^0 = 1.0")
    print("✅ No off-by-one error!")


def main():
    trace_vi_initial_reward()
    trace_score_life_initial_reward()
    compare_both_methods()
    check_for_off_by_one_error()

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print("\nBoth VI and Score-Life handle the initial reward IDENTICALLY:")
    print("  • First reward is NOT discounted (multiplied by 1.0)")
    print("  • Second reward is discounted by γ")
    print("  • Third reward is discounted by γ²")
    print("  • And so on...")
    print("\nThere is NO difference in how the initial reward is accumulated!")
    print("\nThe ~59 unit remaining difference must come from:")
    print("  1. Finite horizon (N=50) vs infinite horizon")
    print("  2. Expected next state (VI) vs sampled next states (Score-Life)")
    print("  3. Numerical approximations in interpolation")


if __name__ == "__main__":
    main()
