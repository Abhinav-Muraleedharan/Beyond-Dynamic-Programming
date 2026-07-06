#!/usr/bin/env python
"""
Measure precise computational efficiency of VI vs Score-Life.

Compares:
1. Time complexity (wall-clock time)
2. Sample complexity (number of environment transitions)
3. Memory complexity (storage requirements)
4. Scalability (how they scale with state space size)
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


class ComputationCounter:
    """Track computational operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.env_steps = 0
        self.value_lookups = 0
        self.iterations = 0

    def step(self):
        self.env_steps += 1

    def lookup(self):
        self.value_lookups += 1

    def iteration(self):
        self.iterations += 1


def measure_vi_complexity(n_states, gamma=0.9, max_mileage=10000,
                         transition_samples=1000, tolerance=1e-6):
    """Measure VI computational complexity."""

    print(f"\n{'='*80}")
    print(f"VALUE ITERATION: n_states={n_states}")
    print(f"{'='*80}")

    counter = ComputationCounter()
    states = np.linspace(0, max_mileage, n_states)

    # Pre-compute transitions
    print("Pre-computing transitions...")
    start_precompute = time.time()

    transitions = {}
    env = BusEngineEnvironment(max_state=max_mileage)

    for i, state in enumerate(states):
        next_states = []
        rewards = []
        for _ in range(transition_samples):
            env.set_state(state)
            next_state, reward, _, _, _ = env.step(0)
            next_states.append(next_state[0])
            rewards.append(reward)
            counter.step()

        transitions[i] = {
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }

    precompute_time = time.time() - start_precompute
    precompute_samples = counter.env_steps

    print(f"  Precompute: {precompute_time:.2f}s, {precompute_samples:,} samples")

    # Run VI
    print("Running VI...")
    start_vi = time.time()

    V = np.zeros(n_states)

    for iteration in range(1000):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Use cached transitions
            next_states = transitions[i]['next_states']
            rewards = transitions[i]['rewards']

            V_next_samples = []
            for ns in next_states:
                next_idx = np.argmin(np.abs(states - ns))
                V_next_samples.append(V[next_idx])
                counter.lookup()

            E_reward = np.mean(rewards)
            E_V_next = np.mean(V_next_samples)

            Q_keep = E_reward + gamma * E_V_next
            Q_replace = -100 + gamma * V[0]
            V_new[i] = max(Q_keep, Q_replace)

        counter.iteration()

        if np.max(np.abs(V_new - V)) < tolerance:
            break

        V = V_new.copy()

    vi_time = time.time() - start_vi
    total_time = precompute_time + vi_time

    # Memory usage
    memory_transitions = sum(len(t['next_states']) + len(t['rewards'])
                            for t in transitions.values()) * 8  # bytes (float64)
    memory_value_function = n_states * 8
    total_memory = memory_transitions + memory_value_function

    print(f"  VI: {vi_time:.2f}s, {counter.iterations} iterations")
    print(f"  Total: {total_time:.2f}s")
    print(f"  Environment steps: {counter.env_steps:,}")
    print(f"  Value lookups: {counter.value_lookups:,}")
    print(f"  Memory: {total_memory/1024:.1f} KB")

    return {
        'total_time': total_time,
        'precompute_time': precompute_time,
        'vi_time': vi_time,
        'env_steps': counter.env_steps,
        'value_lookups': counter.value_lookups,
        'iterations': counter.iterations,
        'memory_bytes': total_memory,
        'value_function': V
    }


def measure_score_life_complexity(n_states, gamma=0.9, N=50, num_samples=1000,
                                  n_l_points=30, max_mileage=10000):
    """Measure Score-Life computational complexity."""

    print(f"\n{'='*80}")
    print(f"SCORE-LIFE: n_states={n_states}")
    print(f"{'='*80}")

    counter = ComputationCounter()
    states = np.linspace(0, max_mileage, n_states)
    V_sl = np.zeros(n_states)

    start_time = time.time()

    # For each state
    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{n_states}")

        env = BusEngineEnvironment(max_state=max_mileage)
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        # Grid search over l
        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = []

        for l in l_values:
            # Each S() call does num_samples * N environment steps
            score = slp.S(l, np.array([state]))
            scores.append(score)
            counter.env_steps += num_samples * N  # Approximate

        V_sl[i] = max(scores)

    total_time = time.time() - start_time

    # Memory usage (Faber-Schauder coefficients)
    # j_max levels, each level has 2^j coefficients
    j_max = 5
    num_coefficients = sum(2**j for j in range(j_max)) + 2  # a_0, a_1, plus coefficients
    memory_fractal = num_coefficients * 8 * n_states  # bytes per state
    total_memory = memory_fractal

    print(f"  Total: {total_time:.2f}s")
    print(f"  Environment steps: {counter.env_steps:,}")
    print(f"  Memory (Faber-Schauder): {total_memory/1024:.1f} KB")

    return {
        'total_time': total_time,
        'env_steps': counter.env_steps,
        'memory_bytes': total_memory,
        'value_function': V_sl
    }


def compare_scalability(state_counts, gamma=0.9):
    """Compare how VI and Score-Life scale with state space size."""

    print(f"\n{'='*80}")
    print("SCALABILITY ANALYSIS")
    print(f"{'='*80}")

    vi_times = []
    sl_times = []
    vi_samples = []
    sl_samples = []
    vi_memory = []
    sl_memory = []

    for n_states in state_counts:
        print(f"\n### Testing with n_states = {n_states} ###")

        # VI
        vi_result = measure_vi_complexity(
            n_states=n_states,
            gamma=gamma,
            transition_samples=500,  # Reduced for speed
            tolerance=1e-4  # Looser for speed
        )
        vi_times.append(vi_result['total_time'])
        vi_samples.append(vi_result['env_steps'])
        vi_memory.append(vi_result['memory_bytes'])

        # Score-Life
        sl_result = measure_score_life_complexity(
            n_states=n_states,
            gamma=gamma,
            N=20,  # Reduced for speed
            num_samples=200,  # Reduced for speed
            n_l_points=10  # Reduced for speed
        )
        sl_times.append(sl_result['total_time'])
        sl_samples.append(sl_result['env_steps'])
        sl_memory.append(sl_result['memory_bytes'])

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time complexity
    ax1 = axes[0, 0]
    ax1.plot(state_counts, vi_times, 'b-o', linewidth=2.5, markersize=8, label='VI')
    ax1.plot(state_counts, sl_times, 'r-s', linewidth=2.5, markersize=8, label='Score-Life')
    ax1.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Complexity', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Sample complexity
    ax2 = axes[0, 1]
    ax2.plot(state_counts, [s/1000 for s in vi_samples], 'b-o', linewidth=2.5,
            markersize=8, label='VI')
    ax2.plot(state_counts, [s/1000 for s in sl_samples], 'r-s', linewidth=2.5,
            markersize=8, label='Score-Life')
    ax2.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Environment Steps (×1000)', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Complexity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Memory complexity
    ax3 = axes[1, 0]
    ax3.plot(state_counts, [m/1024 for m in vi_memory], 'b-o', linewidth=2.5,
            markersize=8, label='VI')
    ax3.plot(state_counts, [m/1024 for m in sl_memory], 'r-s', linewidth=2.5,
            markersize=8, label='Score-Life')
    ax3.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Memory (KB)', fontsize=12, fontweight='bold')
    ax3.set_title('Memory Complexity', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Efficiency ratio
    ax4 = axes[1, 1]
    time_ratio = [sl/vi for sl, vi in zip(sl_times, vi_times)]
    sample_ratio = [sl/vi for sl, vi in zip(sl_samples, vi_samples)]

    ax4.plot(state_counts, time_ratio, 'g-o', linewidth=2.5, markersize=8, label='Time ratio')
    ax4.plot(state_counts, sample_ratio, 'm-s', linewidth=2.5, markersize=8, label='Sample ratio')
    ax4.axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Equal')
    ax4.set_xlabel('Number of States', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Ratio (SL / VI)', fontsize=12, fontweight='bold')
    ax4.set_title('Efficiency Ratio (SL/VI)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/computational_efficiency_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/computational_efficiency_comparison.png")

    return {
        'state_counts': state_counts,
        'vi_times': vi_times,
        'sl_times': sl_times,
        'vi_samples': vi_samples,
        'sl_samples': sl_samples,
        'vi_memory': vi_memory,
        'sl_memory': sl_memory
    }


def print_complexity_summary(results):
    """Print detailed complexity analysis."""

    print("\n" + "="*80)
    print("COMPUTATIONAL COMPLEXITY SUMMARY")
    print("="*80)

    state_counts = results['state_counts']

    print("\nTime Complexity:")
    print(f"{'States':<10} {'VI (s)':<12} {'Score-Life (s)':<15} {'Ratio (SL/VI)':<15}")
    print("-" * 55)
    for i, n in enumerate(state_counts):
        ratio = results['sl_times'][i] / results['vi_times'][i]
        print(f"{n:<10} {results['vi_times'][i]:<12.2f} {results['sl_times'][i]:<15.2f} {ratio:<15.2f}")

    print("\nSample Complexity:")
    print(f"{'States':<10} {'VI':<15} {'Score-Life':<15} {'Ratio (SL/VI)':<15}")
    print("-" * 60)
    for i, n in enumerate(state_counts):
        ratio = results['sl_samples'][i] / results['vi_samples'][i]
        print(f"{n:<10} {results['vi_samples'][i]:<15,} {results['sl_samples'][i]:<15,} {ratio:<15.2f}")

    print("\nMemory Complexity:")
    print(f"{'States':<10} {'VI (KB)':<15} {'Score-Life (KB)':<18} {'Ratio (SL/VI)':<15}")
    print("-" * 60)
    for i, n in enumerate(state_counts):
        vi_kb = results['vi_memory'][i] / 1024
        sl_kb = results['sl_memory'][i] / 1024
        ratio = sl_kb / vi_kb if vi_kb > 0 else 0
        print(f"{n:<10} {vi_kb:<15.2f} {sl_kb:<18.2f} {ratio:<15.2f}")

    # Theoretical complexity
    print("\n" + "="*80)
    print("THEORETICAL COMPLEXITY")
    print("="*80)

    print("\nValue Iteration:")
    print("  Time:   O(K * n_states * transition_samples)")
    print("          where K = convergence iterations")
    print("  Memory: O(n_states * transition_samples)")
    print("  Scales: Linear in n_states (for fixed K)")

    print("\nScore-Life:")
    print("  Time:   O(n_states * n_l_points * num_samples * N)")
    print("          where N = horizon, n_l_points = l-grid density")
    print("  Memory: O(n_states * 2^j_max)")
    print("          Faber-Schauder representation")
    print("  Scales: Linear in n_states")

    print("\nKey Differences:")
    print("  • VI requires K iterations to converge (K ≈ 100-200)")
    print("  • Score-Life is single-pass (no iterations)")
    print("  • VI memory grows with transition_samples")
    print("  • Score-Life memory grows logarithmically (2^j_max)")


def main():
    # Test with different state space sizes
    state_counts = [10, 20, 30]

    print("="*80)
    print("COMPUTATIONAL EFFICIENCY MEASUREMENT")
    print("="*80)
    print("\nComparing Value Iteration vs Score-Life Programming")
    print("Measuring: Time, Samples, Memory")
    print()

    results = compare_scalability(state_counts, gamma=0.9)
    print_complexity_summary(results)


if __name__ == "__main__":
    main()
