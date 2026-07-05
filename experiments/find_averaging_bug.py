#!/usr/bin/env python
"""
Find the bug causing S() to return less negative values than expected.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def trace_S_function_bug():
    """Trace through S() function execution to find the bug."""

    print("=" * 70)
    print("TRACING S() FUNCTION BUG")
    print("=" * 70)

    state = 0.0
    gamma = 0.9
    N = 10
    l_value = 0.5

    # Create SLP with num_samples=1 for deterministic testing
    env = BusEngineEnvironment()
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=1,  # Single sample!
        reference_state=np.array([state])
    )

    # Get the action sequence
    action_sequence = slp._real_to_action_sequence_base(l_value, N, 2)
    print(f"\nAction sequence: {action_sequence}")
    print(f"Length: {len(action_sequence)}")
    print(f"Loop will run: {len(action_sequence) - 1} iterations")

    # Manually execute what S() should do
    print("\n" + "=" * 70)
    print("MANUAL EXECUTION OF S() LOGIC")
    print("=" * 70)

    env.set_state(state)
    print(f"Initial state: {env.current_state()}")

    avg_R = 0

    # Outer loop: num_samples
    for j in range(slp.num_samples):
        R = 0
        env.set_state(state)
        print(f"\nSample {j}:")

        # Inner loop: action sequence
        for i in range(len(action_sequence) - 1):
            action = int(action_sequence[i + 1])
            result = env.step(action)

            if len(result) == 5:
                next_state, reward, done, truncated, _ = result
            else:
                next_state, reward, done, truncated = result

            # This is the accumulation formula from the code
            R = (gamma ** i) * reward + R

            print(f"  Step {i}: action={action}, reward={reward:.2f}, "
                  f"γ^{i}={gamma**i:.4f}, term={gamma**i * reward:.2f}, R={R:.2f}")

            if done or truncated:
                print(f"  Episode terminated")
                break

        avg_R = avg_R + R
        print(f"  Sample {j} total R: {R:.2f}")

    avg_R = avg_R / slp.num_samples
    print(f"\nFinal avg_R: {avg_R:.2f}")

    # Now call actual S() function
    print("\n" + "=" * 70)
    print("ACTUAL S() FUNCTION CALL")
    print("=" * 70)

    # Reset environment
    env2 = BusEngineEnvironment()
    slp2 = ScoreLifeProgramming(
        env2, gamma=gamma, N=N, j_max=5,
        num_samples=1,
        reference_state=np.array([state])
    )

    actual_score = slp2.S(l_value, np.array([state]))
    print(f"S({l_value}, {state}) = {actual_score:.2f}")

    print(f"\nExpected (manual): {avg_R:.2f}")
    print(f"Actual (S()):      {actual_score:.2f}")
    print(f"Difference:        {abs(avg_R - actual_score):.2f}")

    if abs(avg_R - actual_score) < 0.01:
        print("\n✅ Manual and S() match - bug is elsewhere")
    else:
        print("\n❌ Manual and S() don't match - bug IS in S() function")
        print("\nPossible causes:")
        print("  1. num_samples parameter not being used correctly")
        print("  2. Random seed causing different trajectories")
        print("  3. State not being reset properly between samples")
        print("  4. Reward accumulation formula incorrect")

    # Check with multiple samples to see variance
    print("\n" + "=" * 70)
    print("TESTING WITH MULTIPLE SAMPLES")
    print("=" * 70)

    for num_samp in [1, 10, 100, 1000]:
        env3 = BusEngineEnvironment()
        slp3 = ScoreLifeProgramming(
            env3, gamma=gamma, N=N, j_max=5,
            num_samples=num_samp,
            reference_state=np.array([state])
        )

        score = slp3.S(l_value, np.array([state]))
        print(f"  num_samples={num_samp:4d}: S={score:8.2f}")


if __name__ == "__main__":
    trace_S_function_bug()
