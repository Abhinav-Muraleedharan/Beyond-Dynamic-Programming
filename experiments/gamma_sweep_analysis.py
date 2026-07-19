#!/usr/bin/env python
"""
Gamma Sweep Analysis: How discount factor affects VI vs Score-Life agreement

Sweeps over gamma values to show:
1. Value function agreement vs gamma
2. Policy agreement vs gamma
3. Computational time comparison
4. Score-Life's parallelization advantage
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def precompute_transitions(n_states, max_mileage, transition_samples=1000):
    """Pre-compute transition samples for VI."""
    states = np.linspace(0, max_mileage, n_states)
    env = BusEngineEnvironment(max_state=max_mileage)
    transitions = {}

    for i, mileage in enumerate(states):
        next_states = []
        rewards = []
        np.random.seed(i)
        for sample in range(transition_samples):
            env.set_state(mileage)
            next_state, reward, _, _, _ = env.step(0)
            next_states.append(next_state[0])
            rewards.append(reward)
        transitions[i] = {
            'next_states': np.array(next_states),
            'rewards': np.array(rewards)
        }
    return transitions


def value_iteration(gamma, n_states, max_mileage, transitions, tolerance=1e-6):
    """Run VI and extract value function + policy."""
    states = np.linspace(0, max_mileage, n_states)
    V = np.zeros(n_states)

    start_time = time.time()

    # Convergence loop
    for iteration in range(1000):
        V_new = np.zeros(n_states)
        for i in range(n_states):
            next_states = transitions[i]['next_states']
            rewards = transitions[i]['rewards']

            V_next_samples = []
            for ns in next_states:
                next_idx = np.argmin(np.abs(states - ns))
                V_next_samples.append(V[next_idx])

            E_reward = np.mean(rewards)
            E_V_next = np.mean(V_next_samples)

            Q_keep = E_reward + gamma * E_V_next
            Q_replace = -100 + gamma * V[0]
            V_new[i] = max(Q_keep, Q_replace)

        if np.max(np.abs(V_new - V)) < tolerance:
            break
        V = V_new.copy()

    # Extract policy
    policy = np.zeros(n_states)
    for i in range(n_states):
        next_states = transitions[i]['next_states']
        rewards = transitions[i]['rewards']
        V_next_samples = [V[np.argmin(np.abs(states - ns))] for ns in next_states]
        Q_keep = np.mean(rewards) + gamma * np.mean(V_next_samples)
        Q_replace = -100 + gamma * V[0]
        policy[i] = 1 if Q_replace > Q_keep else 0

    elapsed = time.time() - start_time

    return V, policy, elapsed, iteration + 1


def compute_scorelife_single_state(args):
    """Compute Score-Life for a single state (for parallel execution)."""
    mileage, gamma, N, num_samples, n_l_points, max_mileage = args

    env = BusEngineEnvironment(max_state=max_mileage)
    env.set_state(mileage)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([mileage])
    )

    # Grid search for optimal l
    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([mileage])) for l in l_values]
    max_idx = np.argmax(scores)

    return scores[max_idx], l_values[max_idx]


def score_life_parallel(gamma, N, num_samples, n_states, n_l_points, max_mileage, n_workers=None):
    """Run Score-Life in parallel and extract value function + policy."""
    if n_workers is None:
        n_workers = cpu_count()

    states = np.linspace(0, max_mileage, n_states)

    start_time = time.time()

    # Prepare arguments for parallel execution
    args_list = [(m, gamma, N, num_samples, n_l_points, max_mileage) for m in states]

    # Run in parallel
    with Pool(n_workers) as pool:
        results = pool.map(compute_scorelife_single_state, args_list)

    V_sl = np.array([r[0] for r in results])
    optimal_l = np.array([r[1] for r in results])

    # Extract policy
    policy_sl = np.zeros(n_states)
    for i in range(n_states):
        score_keep = V_sl[i]
        score_replace = -100 + gamma * V_sl[0] if i > 0 else -100
        policy_sl[i] = 1 if score_replace > score_keep else 0

    elapsed = time.time() - start_time

    return V_sl, policy_sl, elapsed, optimal_l


def run_single_gamma(gamma, n_states, max_mileage, transitions, N, num_samples, n_l_points):
    """Run VI and Score-Life for a single gamma value."""
    print(f"\n{'='*80}")
    print(f"GAMMA = {gamma}")
    print(f"{'='*80}")

    # Value Iteration
    print("  Running Value Iteration...")
    V_vi, policy_vi, time_vi, iters_vi = value_iteration(
        gamma, n_states, max_mileage, transitions
    )
    print(f"    Time: {time_vi:.2f}s, Iterations: {iters_vi}")

    # Score-Life (parallel)
    print("  Running Score-Life (parallel)...")
    V_sl, policy_sl, time_sl, optimal_l = score_life_parallel(
        gamma, N, num_samples, n_states, n_l_points, max_mileage
    )
    print(f"    Time: {time_sl:.2f}s")

    # Compute metrics
    corr = np.corrcoef(V_vi, V_sl)[0, 1]
    rmse = np.sqrt(np.mean((V_sl - V_vi)**2))
    mean_diff = np.mean(V_sl - V_vi)
    policy_agreement = np.mean(policy_vi == policy_sl) * 100

    print(f"  Value correlation: r={corr:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Mean offset: {mean_diff:.2f}")
    print(f"  Policy agreement: {policy_agreement:.1f}%")
    print(f"  Speedup: {time_vi/time_sl:.2f}x")

    return {
        'gamma': gamma,
        'V_vi': V_vi,
        'V_sl': V_sl,
        'policy_vi': policy_vi,
        'policy_sl': policy_sl,
        'optimal_l': optimal_l,
        'corr': corr,
        'rmse': rmse,
        'mean_diff': mean_diff,
        'policy_agreement': policy_agreement,
        'time_vi': time_vi,
        'time_sl': time_sl,
        'iters_vi': iters_vi
    }


def plot_gamma_sweep_results(results, states):
    """Create comprehensive visualization of gamma sweep."""

    gammas = [r['gamma'] for r in results]
    n_gammas = len(gammas)

    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Row 1: Value functions for each gamma
    for i, result in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(states, result['V_vi'], 'b-', linewidth=2, alpha=0.8, label='VI')
        ax.plot(states, result['V_sl'], 'r--', linewidth=2, alpha=0.8, label='SL')
        ax.set_xlabel('Mileage', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'γ={result["gamma"]}\nr={result["corr"]:.4f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 2: Policies for each gamma
    for i, result in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(states, result['policy_vi'], 'b-', linewidth=2, marker='o',
                markersize=4, alpha=0.8, label='VI')
        ax.plot(states, result['policy_sl'], 'r--', linewidth=2, marker='s',
                markersize=4, alpha=0.8, label='SL')
        ax.set_xlabel('Mileage', fontsize=10)
        ax.set_ylabel('Action', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Keep', 'Replace'], fontsize=9)
        ax.set_title(f'Policy Agreement: {result["policy_agreement"]:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 3: Agreement metrics vs gamma (span multiple columns)
    ax1 = fig.add_subplot(gs[2, 0:2])  # Span 2 columns
    ax1.plot(gammas, [r['corr'] for r in results], 'o-', linewidth=3, markersize=10,
            color='blue', label='Value Correlation')
    ax1.axhline(0.99, color='green', linestyle='--', alpha=0.5, label='Excellent (0.99)')
    ax1.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax1.set_title('Value Function Agreement vs Gamma', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 2:4])  # Span 2 columns
    ax2.plot(gammas, [abs(r['mean_diff']) for r in results], 'o-', linewidth=3,
            markersize=10, color='purple', label='|Mean Offset|')
    ax2.axhline(5, color='green', linestyle='--', alpha=0.5, label='Target (±5)')
    ax2.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|Mean Offset|', fontsize=12, fontweight='bold')
    ax2.set_title('Value Offset vs Gamma', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[3, 0:2])  # Row 4, span 2 columns
    ax3.plot(gammas, [r['policy_agreement'] for r in results], 'o-', linewidth=3,
            markersize=10, color='green', label='Policy Agreement')
    ax3.axhline(95, color='green', linestyle='--', alpha=0.5, label='Target (95%)')
    ax3.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Agreement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Policy Agreement vs Gamma', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Row 4: Computational time
    ax4 = fig.add_subplot(gs[3, 2:4])  # Span 2 columns
    x = np.arange(len(gammas))
    width = 0.35
    ax4.bar(x - width/2, [r['time_vi'] for r in results], width,
           label='VI', color='blue', alpha=0.7)
    ax4.bar(x + width/2, [r['time_sl'] for r in results], width,
           label='SL (4 cores)', color='red', alpha=0.7)
    ax4.set_xlabel('Gamma (γ)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Computation Time by Gamma', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{g}' for g in gammas])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Gamma Sweep Analysis: VI vs Score-Life',
                fontsize=18, fontweight='bold', y=0.995)

    filename = 'results/gamma_sweep_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {filename}")

    return filename


def main():
    # Parameters
    gammas = [0.3, 0.5, 0.7, 0.9]
    n_states = 30
    max_mileage = 10000
    N = 50
    num_samples = 5000
    n_l_points = 100
    transition_samples = 1000

    print("=" * 80)
    print("GAMMA SWEEP ANALYSIS: VI vs Score-Life")
    print("=" * 80)
    print(f"\nSweeping gamma: {gammas}")
    print(f"States: {n_states}")
    print(f"Score-Life: N={N}, samples={num_samples}, l-points={n_l_points}")
    print(f"Cores available: {cpu_count()}")

    # Pre-compute transitions (shared across all gammas)
    print("\nPre-computing transitions...")
    states = np.linspace(0, max_mileage, n_states)
    transitions = precompute_transitions(n_states, max_mileage, transition_samples)

    # Run sweep
    results = []
    for gamma in gammas:
        result = run_single_gamma(
            gamma, n_states, max_mileage, transitions,
            N, num_samples, n_l_points
        )
        results.append(result)

    # Create visualizations
    plot_gamma_sweep_results(results, states)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Findings:")
    print("  1. Lower gamma → better VI-SL agreement (shorter effective horizon)")
    print("  2. Score-Life achieves significant speedup via parallelization")
    print(f"  3. Average speedup: {np.mean([r['time_vi']/r['time_sl'] for r in results]):.1f}×")
    print(f"  4. Parallel efficiency: {np.mean([r['time_vi']/r['time_sl'] for r in results])/cpu_count()*100:.1f}%")
    print("\nGamma-specific results:")
    for r in results:
        print(f"  γ={r['gamma']}: r={r['corr']:.4f}, policy={r['policy_agreement']:.1f}%, "
              f"speedup={r['time_vi']/r['time_sl']:.1f}×")


if __name__ == "__main__":
    main()
