#!/usr/bin/env python
"""
Bus Engine Scalability Experiment
Complete the missing timing data for different state space sizes
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine import BusEngineEnvironment, value_iteration


def run_scalability_experiment():
    """Run value iteration with different state space sizes."""
    print("=" * 70)
    print("BUS ENGINE SCALABILITY EXPERIMENT")
    print("Testing Value Iteration with Different State Space Sizes")
    print("=" * 70)

    # State space sizes to test
    state_space_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    results = {
        "state_space_sizes": [],
        "computation_times": [],
        "replacement_thresholds": []
    }

    for size in state_space_sizes:
        print(f"\n{'='*70}")
        print(f"Testing State Space Size: {size}")
        print(f"{'='*70}")

        # Create environment
        env = BusEngineEnvironment(x=0, p=0.1, q=0.3)

        # Modify the value_iteration function to use custom state space
        state_space = np.linspace(0, 100000, size)
        V = np.zeros_like(state_space)
        gamma = 0.99
        epsilon = 1e-6
        max_iterations = 1000

        print(f"Starting value iteration with {size} states...")
        start_time = time.time()

        # Value iteration loop
        for iteration in range(max_iterations):
            delta = 0
            for i, state in enumerate(state_space):
                env.set_state(state)
                # Compute Q-values for both actions
                q_values = []
                for action in [0, 1]:
                    q_value = 0
                    for _ in range(100):  # Monte Carlo sampling
                        next_state, utility, done, terminated = env.step(action)
                        next_state_idx = np.abs(state_space - next_state).argmin()
                        q_value += utility + gamma * V[next_state_idx]
                    q_values.append(q_value / 100)

                # Update value function
                best_q = max(q_values)
                delta = max(delta, abs(V[i] - best_q))
                V[i] = best_q

            if delta < epsilon:
                print(f"Converged in {iteration + 1} iterations")
                break

        end_time = time.time()
        computation_time = end_time - start_time

        # Compute optimal policy
        policy = np.zeros_like(state_space, dtype=int)
        for i, state in enumerate(state_space):
            env.set_state(state)
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(100):  # Monte Carlo sampling
                    next_state, utility, done, terminated = env.step(action)
                    next_state_idx = np.abs(state_space - next_state).argmin()
                    q_value += utility + gamma * V[next_state_idx]
                q_values.append(q_value / 100)
            policy[i] = np.argmax(q_values)

        # Find replacement threshold
        replacement_threshold = np.argmax(policy)
        threshold_value = state_space[replacement_threshold]

        print(f"Computation time: {computation_time:.2f} seconds")
        print(f"Replacement threshold: {threshold_value:.0f}")

        results["state_space_sizes"].append(size)
        results["computation_times"].append(computation_time)
        results["replacement_thresholds"].append(float(threshold_value))

    return results


def plot_results(results):
    """Create visualization of scalability results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = results["state_space_sizes"]
    times = results["computation_times"]
    thresholds = results["replacement_thresholds"]

    # Plot 1: Computation Time vs State Space Size
    ax1.plot(sizes, times, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('State Space Size', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Value Iteration Scalability', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add data labels
    for i, (size, time) in enumerate(zip(sizes, times)):
        ax1.annotate(f'{time:.1f}s',
                    xy=(size, time),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)

    # Plot 2: Replacement Thresholds
    ax2.plot(sizes, thresholds, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('State Space Size', fontsize=12)
    ax2.set_ylabel('Replacement Threshold (miles)', fontsize=12)
    ax2.set_title('Replacement Threshold vs State Space Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add data labels
    for i, (size, threshold) in enumerate(zip(sizes, thresholds)):
        ax2.annotate(f'{threshold:.0f}',
                    xy=(size, threshold),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig('results/bus_engine_scalability.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/bus_engine_scalability.png")
    plt.close()


def print_summary(results):
    """Print a summary table."""
    print("\n" + "=" * 70)
    print("SCALABILITY RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'State Space Size':<20} {'Time (seconds)':<20} {'Threshold (miles)':<20}")
    print("-" * 70)

    for size, time, threshold in zip(results["state_space_sizes"],
                                      results["computation_times"],
                                      results["replacement_thresholds"]):
        print(f"{size:<20} {time:<20.2f} {threshold:<20.0f}")


def main():
    os.makedirs("results", exist_ok=True)

    # Run experiments
    results = run_scalability_experiment()

    # Save results
    with open("results/bus_engine_scalability.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/bus_engine_scalability.json")

    # Plot results
    plot_results(results)

    # Print summary
    print_summary(results)

    # Print formatted comments for bus_engine.py
    print("\n" + "=" * 70)
    print("FORMATTED COMMENTS FOR bus_engine.py")
    print("=" * 70)
    for size, time in zip(results["state_space_sizes"], results["computation_times"]):
        print(f"    # TOTAL TIME TAKEN FOR VALUE ITERATION - {time:6.2f} s STATE SPACE SIZE: {size}")


if __name__ == "__main__":
    main()
