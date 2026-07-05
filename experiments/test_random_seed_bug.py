#!/usr/bin/env python
"""
Test if random seeding fixes the averaging bug.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def test_with_seed_control():
    """Test S() function with controlled random seed."""

    print("=" * 70)
    print("TESTING WITH RANDOM SEED CONTROL")
    print("=" * 70)

    state = 0.0
    gamma = 0.9
    N = 10
    l_value = 0.5

    print(f"\nComparing S() with and without seed control:\n")

    # Test 1: Without seed control (current behavior)
    print("WITHOUT seed control:")
    for num_samp in [1, 10, 100, 1000]:
        env = BusEngineEnvironment()
        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samp,
            reference_state=np.array([state])
        )
        score = slp.S(l_value, np.array([state]))
        print(f"  num_samples={num_samp:4d}: S={score:8.2f}")

    # Test 2: WITH seed control
    print("\nWITH seed control (set before each call):")
    for num_samp in [1, 10, 100, 1000]:
        np.random.seed(42)  # Fix the seed!
        env = BusEngineEnvironment()
        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samp,
            reference_state=np.array([state])
        )
        score = slp.S(l_value, np.array([state]))
        print(f"  num_samples={num_samp:4d}: S={score:8.2f}")

    # Test 3: Same seed, multiple calls
    print("\nSame seed, multiple S() calls:")
    np.random.seed(42)
    env = BusEngineEnvironment()
    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=100,
        reference_state=np.array([state])
    )

    for i in range(5):
        np.random.seed(42)  # Reset seed each time
        score = slp.S(l_value, np.array([state]))
        print(f"  Call {i+1}: S={score:8.2f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    print("""
If WITH seed control shows stable values across num_samples:
  ✅ The drift is due to random seed affecting different runs

If even WITH seed control values change with num_samples:
  ❌ There's a deeper bug in the averaging logic

Expected behavior:
  - Same seed + same num_samples = same result
  - More samples should REDUCE variance, not change mean
    """)


if __name__ == "__main__":
    test_with_seed_control()
