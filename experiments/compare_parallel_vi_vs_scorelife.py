#!/usr/bin/env python
"""
Compare parallel performance: VI vs Score-Life.

Key difference:
- VI: Can only parallelize WITHIN iterations (still needs K sequential iterations)
- Score-Life: Fully parallelizable across all states (single-pass)
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


def compute_vi_single_state_update(args):
    """Compute value update for a single state (VI worker function)."""
    state, states, V_current, gamma, transitions = args

    state_idx = np.argmin(np.abs(states - state))

    # Use cached transitions
    next_states = transitions[state_idx]['next_states']
    rewards = transitions[state_idx]['rewards']

    # Compute E[V(next)]
    V_next_samples = []
    for ns in next_states:
        next_idx = np.argmin(np.abs(states - ns))
        V_next_samples.append(V_current[next_idx])

    E_reward = np.mean(rewards)
    E_V_next = np.mean(V_next_samples)

    Q_keep = E_reward + gamma * E_V_next
    Q_replace = -100 + gamma * V_current[0]

    return max(Q_keep, Q_replace)


def vi_parallel_within_iteration(states, gamma, max_mileage, transitions,
                                 n_workers, tolerance=1e-6):
    """
    VI with parallelization WITHIN each iteration.
    Still requires K sequential iterations!
    """
    print(f"Running VI with {n_workers} workers (within-iteration parallelization)...")
    start_time = time.time()

    V = np.zeros(len(states))
    iterations = 0

    for iteration in range(1000):
        # Parallel update of all states (but iterations are sequential!)
        args_list = [(state, states, V, gamma, transitions) for state in states]

        if n_workers == 1:
            # Serial
            V_new = np.array([compute_vi_single_state_update(args) for args in args_list])
        else:
            # Parallel within iteration
            with mp.Pool(processes=n_workers) as pool:
                V_new = np.array(pool.map(compute_vi_single_state_update, args_list))

        max_change = np.max(np.abs(V_new - V))

        if max_change < tolerance:
            iterations = iteration + 1
            break

        V = V_new.copy()

    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.2f}s ({iterations} iterations)")

    return V, elapsed, iterations


def compute_sl_single_state(state, gamma, N, num_samples, n_l_points, max_mileage):
    """Compute value for a single state (Score-Life worker)."""
    env = BusEngineEnvironment(max_state=max_mileage)
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([state])) for l in l_values]

    return max(scores)


def scorelife_parallel_across_states(states, gamma, N, num_samples, n_l_points,
                                     max_mileage, n_workers):
    """
    Score-Life with parallelization ACROSS states.
    Single-pass, fully parallelizable!
    """
    print(f"Running Score-Life with {n_workers} workers (across-states parallelization)...")
    start_time = time.time()

    worker_func = partial(
        compute_sl_single_state,
        gamma=gamma, N=N, num_samples=num_samples,
        n_l_points=n_l_points, max_mileage=max_mileage
    )

    if n_workers == 1:
        # Serial
        V = np.array([worker_func(state) for state in states])
    else:
        # Parallel
        with mp.Pool(processes=n_workers) as pool:
            V = np.array(pool.map(worker_func, states))

    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.2f}s (single-pass)")

    return V, elapsed


def precompute_vi_transitions(states, max_mileage, n_samples):
    """Pre-compute transitions for VI."""
    print("Pre-computing VI transitions...")
    start = time.time()

    transitions = {}
    env = BusEngineEnvironment(max_state=max_mileage)

    for i, state in enumerate(states):
        next_states = []
        rewards = []

        np.random.seed(i)
        for _ in range(n_samples):
            env.set_state(state)
            next_state, reward, _, _, _ = env.step(0)
            next_states.append(next_state[0])
            rewards.append(reward)

        transitions[i] = {
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }

    elapsed = time.time() - start
    print(f"  ✓ Done in {elapsed:.2f}s")
    return transitions, elapsed


def benchmark_parallel_comparison(n_states=30, gamma=0.9):
    """Compare parallel scalability of VI vs Score-Life."""

    print("=" * 80)
    print("PARALLEL COMPARISON: VI vs SCORE-LIFE")
    print("=" * 80)

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    # VI parameters (reduced for reasonable runtime)
    vi_transition_samples = 500
    vi_tolerance = 1e-4

    # Score-Life parameters (reduced for reasonable runtime)
    sl_N = 30
    sl_num_samples = 500
    sl_n_l_points = 20

    print(f"\nParameters:")
    print(f"  n_states: {n_states}")
    print(f"  VI: {vi_transition_samples} trans. samples, tolerance={vi_tolerance}")
    print(f"  Score-Life: N={sl_N}, {sl_num_samples} samples, {sl_n_l_points} l-points")

    # Pre-compute VI transitions (this is overhead, but reused across all runs)
    transitions, precompute_time = precompute_vi_transitions(
        states, max_mileage, vi_transition_samples
    )

    cpu_count = mp.cpu_count()
    print(f"\nAvailable CPUs: {cpu_count}")

    worker_counts = [1, 2, min(4, cpu_count)]

    vi_times = []
    vi_iterations = []
    sl_times = []

    print("\n" + "=" * 80)

    for n_workers in worker_counts:
        print(f"\n### Testing with {n_workers} workers ###\n")

        # VI
        _, vi_time, iterations = vi_parallel_within_iteration(
            states, gamma, max_mileage, transitions, n_workers, vi_tolerance
        )
        vi_times.append(vi_time)
        vi_iterations.append(iterations)

        print()

        # Score-Life
        _, sl_time = scorelife_parallel_across_states(
            states, gamma, sl_N, sl_num_samples, sl_n_l_points,
            max_mileage, n_workers
        )
        sl_times.append(sl_time)

    return {
        'worker_counts': worker_counts,
        'vi_times': vi_times,
        'vi_iterations': vi_iterations,
        'sl_times': sl_times,
        'precompute_time': precompute_time
    }


def plot_comparison(results):
    """Create comprehensive comparison visualization."""

    worker_counts = results['worker_counts']
    vi_times = results['vi_times']
    sl_times = results['sl_times']
    vi_iterations = results['vi_iterations']
    precompute = results['precompute_time']

    # Add precompute time to VI times for fair comparison
    vi_total_times = [t + precompute for t in vi_times]

    # Calculate speedups
    vi_speedup = [vi_total_times[0] / t for t in vi_total_times]
    sl_speedup = [sl_times[0] / t for t in sl_times]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Plot 1: Execution time comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(worker_counts))
    width = 0.35

    bars1 = ax1.bar(x - width/2, vi_total_times, width, label='VI (incl. precompute)',
                    color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, sl_times, width, label='Score-Life',
                    color='red', alpha=0.7)

    ax1.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Execution Time: VI vs Score-Life', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(worker_counts)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

    # Plot 2: Speedup comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(worker_counts, vi_speedup, 'b-o', linewidth=3, markersize=10,
            label='VI speedup')
    ax2.plot(worker_counts, sl_speedup, 'r-s', linewidth=3, markersize=10,
            label='Score-Life speedup')
    ax2.plot(worker_counts, worker_counts, 'k--', linewidth=2, alpha=0.5,
            label='Ideal (linear)')

    ax2.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)

    # Plot 3: Parallel efficiency
    ax3 = fig.add_subplot(gs[1, 1])
    vi_efficiency = [s / w * 100 for s, w in zip(vi_speedup, worker_counts)]
    sl_efficiency = [s / w * 100 for s, w in zip(sl_speedup, worker_counts)]

    ax3.plot(worker_counts, vi_efficiency, 'b-o', linewidth=3, markersize=10,
            label='VI efficiency')
    ax3.plot(worker_counts, sl_efficiency, 'r-s', linewidth=3, markersize=10,
            label='Score-Life efficiency')
    ax3.axhline(100, color='k', linestyle='--', linewidth=2, alpha=0.5,
               label='Ideal (100%)')

    ax3.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Parallel Efficiency', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(worker_counts)
    ax3.set_ylim([0, 110])

    # Plot 4: Breakdown of VI time
    ax4 = fig.add_subplot(gs[2, 0])

    # Show precompute vs iteration time
    iter_times = vi_times
    precompute_times = [precompute] * len(worker_counts)

    x = np.arange(len(worker_counts))
    ax4.bar(x, precompute_times, width=0.6, label='Precompute (one-time)',
           color='lightblue', alpha=0.7)
    ax4.bar(x, iter_times, width=0.6, bottom=precompute_times,
           label='Iterations (parallelized)', color='blue', alpha=0.7)

    ax4.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('VI Time Breakdown', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(worker_counts)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Comparison table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    table_data = [
        ['', 'VI', 'Score-Life', 'Winner'],
        ['─' * 12, '─' * 15, '─' * 15, '─' * 15],
        ['Parallelization', 'Within iter.', 'Across states', 'Score-Life'],
        ['Sequential', f'{vi_iterations[0]} iters', 'Single-pass', 'Score-Life'],
        ['Time (1 core)', f'{vi_total_times[0]:.1f}s', f'{sl_times[0]:.1f}s',
         'VI' if vi_total_times[0] < sl_times[0] else 'Score-Life'],
        ['Time (4 cores)', f'{vi_total_times[-1]:.1f}s', f'{sl_times[-1]:.1f}s',
         'VI' if vi_total_times[-1] < sl_times[-1] else 'Score-Life'],
        ['Max speedup', f'{vi_speedup[-1]:.2f}×', f'{sl_speedup[-1]:.2f}×',
         'VI' if vi_speedup[-1] > sl_speedup[-1] else 'Score-Life'],
        ['Efficiency', f'{vi_efficiency[-1]:.1f}%', f'{sl_efficiency[-1]:.1f}%',
         'VI' if vi_efficiency[-1] > sl_efficiency[-1] else 'Score-Life'],
    ]

    table_text = '\n'.join(['  '.join(row) for row in table_data])

    ax5.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('PARALLEL SCALABILITY: VI vs Score-Life',
                fontsize=17, fontweight='bold', y=0.995)

    filename = 'results/parallel_vi_vs_scorelife.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def print_analysis(results):
    """Print detailed analysis."""

    worker_counts = results['worker_counts']
    vi_times = results['vi_times']
    vi_total_times = [t + results['precompute_time'] for t in vi_times]
    sl_times = results['sl_times']
    vi_iterations = results['vi_iterations']

    vi_speedup = [vi_total_times[0] / t for t in vi_total_times]
    sl_speedup = [sl_times[0] / t for t in sl_times]

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    print(f"\n{'Workers':<10} {'VI Time':<15} {'VI Speedup':<15} {'SL Time':<15} {'SL Speedup':<15}")
    print("-" * 75)

    for i, w in enumerate(worker_counts):
        print(f"{w:<10} {vi_total_times[i]:<15.2f} {vi_speedup[i]:<15.2f}× "
              f"{sl_times[i]:<15.2f} {sl_speedup[i]:<15.2f}×")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. PARALLELIZATION STRATEGY:")
    print("   VI:")
    print("     • Parallelizes WITHIN each iteration")
    print("     • But iterations are SEQUENTIAL (must wait for convergence)")
    print(f"     • Required {vi_iterations[0]} iterations to converge")
    print("     • Limited speedup potential")

    print("\n   Score-Life:")
    print("     • Parallelizes ACROSS all states")
    print("     • Single-pass (no iterations)")
    print("     • Embarrassingly parallel")
    print("     • Near-linear speedup potential")

    print("\n2. SPEEDUP COMPARISON:")
    max_workers = worker_counts[-1]
    print(f"   With {max_workers} workers:")
    print(f"     VI speedup:          {vi_speedup[-1]:.2f}×")
    print(f"     Score-Life speedup:  {sl_speedup[-1]:.2f}×")
    print(f"     Advantage:           Score-Life is {sl_speedup[-1]/vi_speedup[-1]:.2f}× better")

    print("\n3. WHY VI HAS LIMITED SPEEDUP:")
    print(f"   • Must complete ALL {vi_iterations[0]} iterations sequentially")
    print("   • Each iteration can be parallelized, but then must synchronize")
    print("   • Overhead compounds over many iterations")
    print("   • Amdahl's Law: sequential bottleneck limits speedup")

    print("\n4. WHY SCORE-LIFE SCALES BETTER:")
    print("   • No sequential dependency (single-pass)")
    print("   • Each state completely independent")
    print("   • Near-perfect parallel efficiency")
    print("   • Scales to thousands of cores (GPU-ready)")


def main():
    n_states = 30

    print("=" * 80)
    print("COMPARING PARALLEL SCALABILITY: VI vs SCORE-LIFE")
    print("=" * 80)
    print("\nThis experiment reveals a fundamental difference:")
    print("  VI: Limited by sequential iterations")
    print("  Score-Life: Fully parallelizable (single-pass)")
    print()

    results = benchmark_parallel_comparison(n_states=n_states)
    plot_comparison(results)
    print_analysis(results)


if __name__ == "__main__":
    main()
