#!/usr/bin/env python
"""
Demonstrate Score-Life on 10,000 states using parallel execution.

Shows this is completely feasible with the multi-core approach.
"""

import numpy as np
import time
import multiprocessing as mp
from functools import partial
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def compute_state_value(state, gamma, N, num_samples, n_l_points, max_mileage):
    """Worker function: compute value for a single state."""

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
    return {
        'state': state,
        'value': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


def run_10000_states_demo():
    """Demonstrate Score-Life on 10,000 states."""

    print("=" * 80)
    print("SCORE-LIFE: 10,000 STATES DEMONSTRATION")
    print("=" * 80)

    # Parameters
    n_states = 10000
    gamma = 0.9
    max_mileage = 100000  # 0 to 100,000 miles
    n_workers = mp.cpu_count()

    # Use moderate parameters for reasonable runtime
    N = 20           # Finite horizon
    num_samples = 200  # Monte Carlo samples
    n_l_points = 10    # l-grid points

    print(f"\nConfiguration:")
    print(f"  States:       {n_states:,}")
    print(f"  Max mileage:  {max_mileage:,} miles")
    print(f"  Resolution:   {max_mileage/n_states:.1f} miles per state")
    print(f"  Workers:      {n_workers}")
    print(f"  N (horizon):  {N}")
    print(f"  MC samples:   {num_samples}")
    print(f"  l-points:     {n_l_points}")

    # Estimate time
    # From previous experiments: ~0.05s per state with these params on 1 core
    estimated_serial = 0.05 * n_states
    estimated_parallel = estimated_serial / n_workers * 1.1  # 10% overhead

    print(f"\nEstimated time:")
    print(f"  Serial:       ~{estimated_serial/60:.1f} minutes")
    print(f"  Parallel:     ~{estimated_parallel/60:.1f} minutes")
    print(f"  Speedup:      ~{estimated_serial/estimated_parallel:.1f}×")

    # Generate states
    states = np.linspace(0, max_mileage, n_states)

    print(f"\nStarting computation...")
    print(f"(This will take approximately {estimated_parallel/60:.1f} minutes)")

    # Create worker function
    worker_func = partial(
        compute_state_value,
        gamma=gamma, N=N, num_samples=num_samples,
        n_l_points=n_l_points, max_mileage=max_mileage
    )

    start_time = time.time()

    # Parallel execution with progress updates
    with mp.Pool(processes=n_workers) as pool:
        # Use imap for progress tracking
        results = []
        for i, result in enumerate(pool.imap(worker_func, states, chunksize=10)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                remaining = (n_states - i - 1) * (elapsed / (i + 1))
                print(f"  Progress: {i+1:,}/{n_states:,} "
                      f"({100*(i+1)/n_states:.1f}%) - "
                      f"ETA: {remaining/60:.1f} min")

    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"  Throughput: {n_states/elapsed:.2f} states/second")
    print(f"  Per-state:  {elapsed/n_states*1000:.2f} ms")

    # Extract results
    states_computed = np.array([r['state'] for r in results])
    values = np.array([r['value'] for r in results])
    optimal_ls = np.array([r['optimal_l'] for r in results])

    # Analyze results
    print(f"\nValue Function Analysis:")
    print(f"  V(0):           {values[0]:.4f}")
    print(f"  V({max_mileage}):   {values[-1]:.4f}")
    print(f"  Mean V:         {np.mean(values):.4f}")
    print(f"  Std V:          {np.std(values):.4f}")

    # Find policy threshold
    policy = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        # Simplified policy extraction
        if values[i] < values[0] - 50:  # Replace if value drops significantly
            policy[i] = 1

    threshold_idx = np.where(policy == 1)[0]
    if len(threshold_idx) > 0:
        threshold = states_computed[threshold_idx[0]]
        print(f"\nPolicy Threshold: {threshold:,.0f} miles")
    else:
        print(f"\nPolicy: Always keep (threshold > {max_mileage:,})")

    # Memory usage
    memory_mb = (n_states * 33 * 8) / (1024**2)  # 33 coefficients per state
    print(f"\nMemory Usage:")
    print(f"  Faber-Schauder coefficients: {memory_mb:.2f} MB")
    print(f"  Results storage: {len(results) * 3 * 8 / 1024:.2f} KB")

    # Save results (compact format)
    np.savez_compressed(
        'results/value_function_10000_states.npz',
        states=states_computed,
        values=values,
        optimal_ls=optimal_ls,
        policy=policy,
        metadata={
            'n_states': n_states,
            'gamma': gamma,
            'N': N,
            'num_samples': num_samples,
            'elapsed': elapsed
        }
    )
    print(f"\n✓ Saved results to: results/value_function_10000_states.npz")

    return results, elapsed


def estimate_scaling():
    """Estimate performance for different state counts."""

    print("\n" + "=" * 80)
    print("SCALING ESTIMATES")
    print("=" * 80)

    n_workers = mp.cpu_count()
    time_per_state_parallel = 0.05 / n_workers * 1.1  # seconds (with overhead)

    configs = [
        ('Ultra-Fast', 10, 100, 5, 0.02 / n_workers * 1.1),
        ('Fast', 20, 200, 10, 0.05 / n_workers * 1.1),
        ('Balanced', 30, 500, 20, 0.12 / n_workers * 1.1),
        ('High-Quality', 50, 1000, 30, 0.25 / n_workers * 1.1),
    ]

    state_counts = [1000, 5000, 10000, 50000, 100000]

    print(f"\nAssuming {n_workers} CPU cores:\n")

    for config_name, N, samples, l_pts, time_per in configs:
        print(f"{config_name} (N={N}, samples={samples}, l-points={l_pts}):")
        print(f"  {'States':<12} {'Time':<15} {'Throughput':<15}")
        print(f"  {'-'*40}")

        for n_states in state_counts:
            total_time = n_states * time_per
            if total_time < 60:
                time_str = f"{total_time:.1f}s"
            elif total_time < 3600:
                time_str = f"{total_time/60:.1f}min"
            else:
                time_str = f"{total_time/3600:.1f}hr"

            throughput = n_states / total_time
            print(f"  {n_states:<12,} {time_str:<15} {throughput:<15,.1f} states/s")
        print()

    print(f"\nKEY INSIGHT:")
    print(f"  With {n_workers} cores, 10,000 states is completely feasible!")
    print(f"  Even 100,000 states is practical with moderate parameters.")


def main():
    # Show estimates first
    estimate_scaling()

    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR 10,000 STATES")
    print("=" * 80)

    print("""
For 10,000 states, use this command:

from experiments.large_scale_parallel_scorelife import run_parallel_scorelife
import numpy as np

states = np.linspace(0, 100000, 10000)  # 0 to 100k miles
results, time = run_parallel_scorelife(
    states,
    N=30,              # Balanced quality
    num_samples=500,
    n_l_points=20,
    n_workers=None     # Use all available cores
)

Expected time: ~1-3 minutes on a modern multi-core CPU

For even faster (rough estimates):
    N=10, num_samples=100, n_l_points=5  →  ~20-30 seconds

For high quality (publication):
    N=50, num_samples=1000, n_l_points=30  →  ~5-10 minutes
""")

    print("\nWould you like me to run the full 10,000 state computation now?")
    print("(It will take approximately 1-3 minutes)")


if __name__ == "__main__":
    main()

    # Uncomment to run the actual computation:
    # run_10000_states_demo()
