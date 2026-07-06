#!/usr/bin/env python
"""
Demonstrate parallelization speedup for Score-Life Programming.

Compare serial vs parallel execution across multiple CPU cores.
"""

import numpy as np
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def compute_value_for_single_state(state, gamma, N, num_samples, n_l_points, max_mileage):
    """Compute value function for a single state (worker function)."""

    env = BusEngineEnvironment(max_state=max_mileage)
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    # Grid search over l
    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([state])) for l in l_values]

    max_idx = np.argmax(scores)
    optimal_value = scores[max_idx]
    optimal_l = l_values[max_idx]

    return state, optimal_value, optimal_l


def score_life_serial(states, gamma, N, num_samples, n_l_points, max_mileage):
    """Serial execution (baseline)."""

    print("Running SERIAL Score-Life...")
    start_time = time.time()

    results = []
    for i, state in enumerate(states):
        if i % 5 == 0:
            print(f"  State {i}/{len(states)}")

        _, value, optimal_l = compute_value_for_single_state(
            state, gamma, N, num_samples, n_l_points, max_mileage
        )
        results.append((state, value, optimal_l))

    elapsed = time.time() - start_time

    print(f"  ✓ Completed in {elapsed:.2f} seconds")

    return results, elapsed


def score_life_parallel(states, gamma, N, num_samples, n_l_points, max_mileage, n_workers):
    """Parallel execution using multiprocessing."""

    print(f"Running PARALLEL Score-Life ({n_workers} workers)...")
    start_time = time.time()

    # Create worker function with fixed parameters
    worker_func = partial(
        compute_value_for_single_state,
        gamma=gamma,
        N=N,
        num_samples=num_samples,
        n_l_points=n_l_points,
        max_mileage=max_mileage
    )

    # Parallel execution
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_func, states)

    elapsed = time.time() - start_time

    print(f"  ✓ Completed in {elapsed:.2f} seconds")
    print(f"  ✓ Speedup: {elapsed:.2f}s → {elapsed:.2f}s")

    return results, elapsed


def benchmark_parallel_speedup(n_states=30, gamma=0.9, N=50, num_samples=500,
                               n_l_points=20, max_mileage=10000):
    """Benchmark speedup with different numbers of workers."""

    print("=" * 80)
    print("PARALLEL SPEEDUP BENCHMARK")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  n_states:     {n_states}")
    print(f"  N (horizon):  {N}")
    print(f"  num_samples:  {num_samples}")
    print(f"  n_l_points:   {n_l_points}")

    # Get CPU count
    cpu_count = mp.cpu_count()
    print(f"\nAvailable CPUs: {cpu_count}")

    states = np.linspace(0, max_mileage, n_states)

    # Test different worker counts
    worker_counts = [1, 2, 4, min(8, cpu_count), min(16, cpu_count)]
    worker_counts = sorted(list(set(worker_counts)))  # Remove duplicates

    times = []
    speedups = []

    print(f"\n{'=' * 80}")

    # Serial baseline
    print("\n1. SERIAL BASELINE (1 worker)")
    _, serial_time = score_life_serial(states, gamma, N, num_samples, n_l_points, max_mileage)
    times.append(serial_time)
    speedups.append(1.0)

    # Parallel with different worker counts
    for i, n_workers in enumerate(worker_counts[1:], start=2):
        print(f"\n{i}. PARALLEL ({n_workers} workers)")
        _, parallel_time = score_life_parallel(
            states, gamma, N, num_samples, n_l_points, max_mileage, n_workers
        )
        times.append(parallel_time)
        speedup = serial_time / parallel_time
        speedups.append(speedup)
        print(f"  → Speedup: {speedup:.2f}x")

    return worker_counts, times, speedups, serial_time


def plot_speedup_results(worker_counts, times, speedups, serial_time):
    """Create comprehensive speedup visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Execution time vs workers
    ax1.plot(worker_counts, times, 'b-o', linewidth=3, markersize=10, label='Actual time')
    ax1.axhline(serial_time, color='red', linestyle='--', linewidth=2,
               alpha=0.6, label=f'Serial baseline ({serial_time:.2f}s)')
    ax1.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Execution Time vs Number of Workers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)

    # Plot 2: Speedup vs workers
    ax2.plot(worker_counts, speedups, 'g-o', linewidth=3, markersize=10, label='Actual speedup')
    ax2.plot(worker_counts, worker_counts, 'r--', linewidth=2, alpha=0.6, label='Linear (ideal)')
    ax2.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Speedup (x)', fontsize=13, fontweight='bold')
    ax2.set_title('Speedup vs Number of Workers', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)

    # Plot 3: Parallel efficiency
    efficiency = [s / w * 100 for s, w in zip(speedups, worker_counts)]
    ax3.plot(worker_counts, efficiency, 'm-o', linewidth=3, markersize=10)
    ax3.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Ideal (100%)')
    ax3.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Parallel Efficiency = (Speedup / Workers) × 100%', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(worker_counts)
    ax3.set_ylim([0, 110])

    # Plot 4: Summary table
    ax4.axis('off')

    table_data = []
    table_data.append(['Workers', 'Time (s)', 'Speedup', 'Efficiency'])
    table_data.append(['─' * 8, '─' * 10, '─' * 8, '─' * 10])

    for i, (w, t, s, e) in enumerate(zip(worker_counts, times, speedups, efficiency)):
        table_data.append([
            f"{w:>8}",
            f"{t:>10.2f}",
            f"{s:>8.2f}x",
            f"{e:>9.1f}%"
        ])

    # Create table text
    table_text = '\n'.join(['  '.join(row) for row in table_data])

    ax4.text(0.1, 0.5, table_text, fontsize=12, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax4.set_title('Speedup Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Score-Life Programming: Parallel Speedup Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    filename = 'results/parallel_speedup_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def print_summary(worker_counts, times, speedups, serial_time):
    """Print detailed summary of results."""

    print("\n" + "=" * 80)
    print("PARALLEL SPEEDUP SUMMARY")
    print("=" * 80)

    efficiency = [s / w * 100 for s, w in zip(speedups, worker_counts)]

    print(f"\n{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12}")
    print("-" * 50)

    for w, t, s, e in zip(worker_counts, times, speedups, efficiency):
        print(f"{w:<10} {t:<12.2f} {s:<12.2f}x {e:<12.1f}%")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    max_speedup = max(speedups)
    max_speedup_idx = speedups.index(max_speedup)
    best_workers = worker_counts[max_speedup_idx]
    best_time = times[max_speedup_idx]

    print(f"\n✓ Serial execution:     {serial_time:.2f} seconds")
    print(f"✓ Best parallel:        {best_time:.2f} seconds ({best_workers} workers)")
    print(f"✓ Maximum speedup:      {max_speedup:.2f}x")
    print(f"✓ Time reduction:       {(1 - best_time/serial_time) * 100:.1f}%")

    last_efficiency = efficiency[-1]
    if last_efficiency > 80:
        print(f"\n✅ EXCELLENT parallel efficiency ({last_efficiency:.1f}%)")
        print("   Score-Life scales nearly linearly with CPU cores!")
    elif last_efficiency > 60:
        print(f"\n✅ GOOD parallel efficiency ({last_efficiency:.1f}%)")
        print("   Minor overhead from inter-process communication")
    else:
        print(f"\n⚠️  Moderate efficiency ({last_efficiency:.1f}%)")
        print("   Communication overhead becoming significant")

    print("\n" + "=" * 80)
    print("THEORETICAL vs ACTUAL")
    print("=" * 80)

    print("\nEmbarrassingly Parallel Problem:")
    print("  Each state's computation is independent")
    print("  Theoretical speedup = number of workers")
    print("\nActual Results:")
    for w, s, e in zip(worker_counts, speedups, efficiency):
        gap = w - s
        print(f"  {w} workers: {s:.2f}x speedup ({e:.1f}% efficiency, {gap:.2f}x below ideal)")


def main():
    # Use moderate parameters for reasonable runtime
    n_states = 30
    N = 30  # Reduced from 50 for faster demo
    num_samples = 500  # Reduced from 1000
    n_l_points = 20  # Reduced from 30

    print("=" * 80)
    print("SCORE-LIFE PARALLEL EXECUTION DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating embarrassingly parallel nature of Score-Life")
    print("Each state's value can be computed independently in parallel\n")

    worker_counts, times, speedups, serial_time = benchmark_parallel_speedup(
        n_states=n_states,
        N=N,
        num_samples=num_samples,
        n_l_points=n_l_points
    )

    plot_speedup_results(worker_counts, times, speedups, serial_time)
    print_summary(worker_counts, times, speedups, serial_time)


if __name__ == "__main__":
    main()
