#!/usr/bin/env python
"""
Verify that Value Iteration is truly converging.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment


def run_vi_with_diagnostics(gamma=0.9, n_states=30, max_iterations=1000, tolerance=1e-6):
    """Run VI with detailed convergence diagnostics."""

    print("=" * 80)
    print("VALUE ITERATION CONVERGENCE DIAGNOSTICS")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  gamma: {gamma}")
    print(f"  n_states: {n_states}")
    print(f"  max_iterations: {max_iterations}")
    print(f"  tolerance: {tolerance}")

    max_mileage = 10000
    states = np.linspace(0, max_mileage, n_states)

    # Track convergence metrics
    max_changes = []
    mean_changes = []
    V_history = []

    # Run value iteration
    V = np.zeros(n_states)
    V_history.append(V.copy())

    converged = False
    final_iteration = 0

    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)

        for i, state in enumerate(states):
            # Q(state, keep)
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(state + expected_delta, max_mileage)

            operating_cost = -0.01 * next_state

            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        # Compute change metrics
        change = np.abs(V_new - V)
        max_change = np.max(change)
        mean_change = np.mean(change)

        max_changes.append(max_change)
        mean_changes.append(mean_change)
        V_history.append(V_new.copy())

        # Print progress every 50 iterations
        if iteration % 50 == 0 or iteration < 10:
            print(f"\nIteration {iteration + 1}:")
            print(f"  Max change: {max_change:.10f}")
            print(f"  Mean change: {mean_change:.10f}")
            print(f"  V(0): {V_new[0]:.6f}")
            print(f"  V(10000): {V_new[-1]:.6f}")

        # Check convergence
        if max_change < tolerance:
            print(f"\n{'='*80}")
            print(f"CONVERGED in {iteration + 1} iterations!")
            print(f"{'='*80}")
            print(f"  Final max change: {max_change:.12f}")
            print(f"  Final mean change: {mean_change:.12f}")
            print(f"  Tolerance: {tolerance:.12f}")
            print(f"  Max change / tolerance: {max_change / tolerance:.4f}")
            converged = True
            final_iteration = iteration + 1
            break

        V = V_new.copy()

    if not converged:
        print(f"\n{'='*80}")
        print(f"WARNING: Did NOT converge after {max_iterations} iterations!")
        print(f"{'='*80}")
        print(f"  Final max change: {max_changes[-1]:.12f}")
        print(f"  Tolerance: {tolerance:.12f}")
        final_iteration = max_iterations

    # Final value function
    print(f"\nFinal Value Function:")
    print(f"  V(0): {V_new[0]:.6f}")
    print(f"  V(5000): {V_new[np.argmin(np.abs(states - 5000))]:.6f}")
    print(f"  V(10000): {V_new[-1]:.6f}")
    print(f"  Range: {V_new.max() - V_new.min():.6f}")

    # Test if running more iterations would change anything
    print(f"\n{'='*80}")
    print("TESTING: What if we run 100 MORE iterations?")
    print(f"{'='*80}")

    V_before = V_new.copy()

    for extra_iter in range(100):
        V_extra = np.zeros(n_states)

        for i, state in enumerate(states):
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(state + expected_delta, max_mileage)
            operating_cost = -0.01 * next_state
            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V_new[next_idx]
            Q_keep = operating_cost + gamma * V_next
            Q_replace = -100 + gamma * V_new[0]
            V_extra[i] = max(Q_keep, Q_replace)

        V_new = V_extra.copy()

    V_after = V_new.copy()
    extra_change = np.max(np.abs(V_after - V_before))

    print(f"\nAfter 100 extra iterations:")
    print(f"  Max change from 'converged' solution: {extra_change:.12f}")
    print(f"  V(0) before: {V_before[0]:.6f}")
    print(f"  V(0) after:  {V_after[0]:.6f}")
    print(f"  Difference:  {V_after[0] - V_before[0]:.12f}")

    if extra_change < tolerance:
        print(f"\n  ✅ VI has truly converged (no change beyond tolerance)")
    else:
        print(f"\n  ⚠️  WARNING: VI may not have fully converged!")

    # Plot convergence
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Max change over iterations (log scale)
    ax1.semilogy(range(1, len(max_changes) + 1), max_changes, 'b-', linewidth=2)
    ax1.axhline(tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance={tolerance}')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max |V_new - V| (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('VI Convergence: Max Change per Iteration', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean change over iterations
    ax2.semilogy(range(1, len(mean_changes) + 1), mean_changes, 'g-', linewidth=2)
    ax2.axhline(tolerance, color='red', linestyle='--', linewidth=2, label=f'Tolerance={tolerance}')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean |V_new - V| (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('VI Convergence: Mean Change per Iteration', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Value function evolution
    sample_iters = [0, 5, 10, 20, 50, 100, final_iteration - 1]
    sample_iters = [i for i in sample_iters if i < len(V_history)]

    for i in sample_iters:
        alpha = 0.3 + 0.7 * (i / max(sample_iters))
        ax3.plot(states, V_history[i], linewidth=2, alpha=alpha, label=f'Iter {i}')

    ax3.set_xlabel('Mileage (miles)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('V(state)', fontsize=12, fontweight='bold')
    ax3.set_title('Value Function Evolution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence rate
    if len(max_changes) > 1:
        convergence_rate = [max_changes[i] / max_changes[i-1] if max_changes[i-1] > 0 else 0
                           for i in range(1, len(max_changes))]
        ax4.plot(range(2, len(max_changes) + 1), convergence_rate, 'purple', linewidth=2)
        ax4.axhline(gamma, color='orange', linestyle='--', linewidth=2,
                   label=f'γ={gamma} (expected rate)', alpha=0.7)
        ax4.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax4.set_ylabel('max_change[i] / max_change[i-1]', fontsize=12, fontweight='bold')
        ax4.set_title('Convergence Rate (should approach γ)', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, min(1.5, max(convergence_rate) * 1.1)])

    plt.tight_layout()
    plt.savefig('results/vi_convergence_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/vi_convergence_diagnostics.png")

    return V_new, converged, final_iteration, max_changes


def main():
    # Test with default parameters
    V, converged, iterations, max_changes = run_vi_with_diagnostics(
        gamma=0.9,
        n_states=30,
        max_iterations=1000,
        tolerance=1e-6
    )

    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if converged:
        print(f"\n✅ Value Iteration CONVERGED properly in {iterations} iterations")
        print(f"   Final max change: {max_changes[-1]:.12f} < tolerance: {1e-6:.12f}")
    else:
        print(f"\n❌ Value Iteration did NOT converge")
        print(f"   You may need to increase max_iterations or check for bugs")


if __name__ == "__main__":
    main()
