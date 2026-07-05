#!/usr/bin/env python
"""
Analyze Score-Life Action Sequences

Understand what action sequences are encoded by different l* values
and how they differ from stationary policies.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def decode_action_sequence(l_value, N=10, M=2):
    """Decode what action sequence is represented by l."""
    env = BusEngineEnvironment()
    slp = ScoreLifeProgramming(env, gamma=0.9, N=N, j_max=5, num_samples=100, reference_state=np.array([0.0]))
    action_seq = slp._real_to_action_sequence_base(l_value, N, M)
    return action_seq


def main():
    print("=" * 70)
    print("ANALYZING SCORE-LIFE ACTION SEQUENCES")
    print("=" * 70)

    # Load results
    with open('results/bus_engine_definitive_comparison.json', 'r') as f:
        results = json.load(f)

    # Analyze γ=0.9
    gamma_data = [r for r in results if abs(r['gamma'] - 0.9) < 0.01][0]
    sl_analysis = gamma_data['SL']['analysis']

    print(f"\nγ = 0.9")
    print("=" * 70)
    print(f"\n{'State':<10} {'optimal_l*':<12} {'Action Sequence':<20} {'Interpretation'}")
    print("-" * 70)

    for item in sl_analysis:
        state = item['state']
        l_star = item['optimal_l']

        # Decode action sequence
        action_seq = decode_action_sequence(l_star, N=10, M=2)

        # Interpret the sequence
        if action_seq.startswith('.0'):
            num_zeros = len(action_seq) - len(action_seq.lstrip('.0'))
            interpretation = f"Keep for {num_zeros} steps, then..."
        elif action_seq.startswith('.1'):
            interpretation = "Replace immediately"
        else:
            num_zeros = 0
            for char in action_seq[1:]:  # Skip the '.'
                if char == '0':
                    num_zeros += 1
                else:
                    break
            interpretation = f"Keep for {num_zeros} steps, then mixed"

        print(f"{state:<10.0f} {l_star:<12.4f} {action_seq:<20} {interpretation}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
Score-Life Programming encodes ACTION SEQUENCES (time-dependent policies),
not STATIONARY POLICIES (state-dependent policies).

- VI finds: π(state) = action  (same action for same state every time)
- SL finds: π(t) = action      (action depends on time step, not state)

This is why l* varies across states - each starting state needs a different
time-sequence of actions to be optimal.

For threshold policies (bus engine problem), VI says:
  "Replace when mileage > X" (stationary rule)

Score-Life would encode:
  "Replace at time step T" (time-based rule)

These are fundamentally different representations!
    """)

    print("\n" + "=" * 70)
    print("TESTING HYPOTHESIS")
    print("=" * 70)

    # Test if lower l* values correspond to "replace earlier in sequence"
    print("\nSorted by l* (should show pattern if hypothesis is correct):")
    sorted_items = sorted(sl_analysis, key=lambda x: x['optimal_l'])

    print(f"\n{'l*':<8} {'State':<10} {'Action Sequence':<20}")
    print("-" * 40)
    for item in sorted_items:
        l_star = item['optimal_l']
        state = item['state']
        action_seq = decode_action_sequence(l_star, N=10, M=2)
        print(f"{l_star:<8.4f} {state:<10.0f} {action_seq:<20}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("""
Score-Life Programming and Value Iteration solve different formulations:

1. VALUE ITERATION:
   - Finds stationary policy π(s): same action for same state
   - Optimal for Markov Decision Processes
   - Policy: "Replace when mileage ≥ threshold"

2. SCORE-LIFE PROGRAMMING:
   - Finds time-based action sequences
   - Optimizes action schedule from each starting state
   - Policy: "At step 1 do A, at step 2 do B, ..."

For the bus engine problem, VI's formulation is more natural because
the decision should depend on current mileage (state), not on time elapsed.

However, Score-Life's approach might be useful for:
- Finite-horizon problems
- Problems where actions should change over time
- Warm-starting policies that evolve with time

To compare them fairly, we'd need to:
1. Convert SL's action sequences to stationary policies, OR
2. Evaluate both using the same metric (e.g., empirical reward)
    """)


if __name__ == "__main__":
    main()
