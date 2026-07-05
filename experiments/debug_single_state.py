#!/usr/bin/env python
"""
Debug Faber-Schauder reconstruction for a SINGLE state.

Focus on state=0 to understand why reconstruction fails.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def debug_single_state(state=0.0, gamma=0.9, N=50, j_max=5, num_samples=500):
    """Detailed analysis of Score function and reconstruction for ONE state."""

    print("=" * 70)
    print(f"DEBUGGING SINGLE STATE: {state}")
    print("=" * 70)
    print(f"Parameters: γ={gamma}, N={N}, j_max={j_max}, num_samples={num_samples}\n")

    env = BusEngineEnvironment()
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=j_max,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    # Compute Faber-Schauder coefficients
    print("Computing Faber-Schauder coefficients...")
    score_func = slp._compute_faber_schauder_coefficients()

    print(f"\nCoefficients:")
    print(f"  a_0 (S(0)): {score_func.compute_fractal(0.0):.2f}")
    print(f"  a_1 (S(1)-S(0)): {score_func.compute_fractal(1.0) - score_func.compute_fractal(0.0):.2f}")
    print(f"  j_max: {j_max}")

    # Sample the Score function densely
    print(f"\nSampling Score function at 50 points...")
    l_values = np.linspace(0.0, 1.0, 50)

    direct_scores = []
    reconstructed_scores = []

    for i, l in enumerate(l_values):
        if i % 10 == 0:
            print(f"  Progress: {i}/50")
        direct = slp.S(l, np.array([state]))
        reconstructed = score_func.compute_fractal(l)

        direct_scores.append(direct)
        reconstructed_scores.append(reconstructed)

    direct_scores = np.array(direct_scores)
    reconstructed_scores = np.array(reconstructed_scores)

    # Calculate errors
    errors = np.abs(direct_scores - reconstructed_scores)
    max_error_idx = np.argmax(errors)

    print(f"\nReconstruction Analysis:")
    print(f"  Average error: {np.mean(errors):.2f}")
    print(f"  Max error: {errors[max_error_idx]:.2f} at l={l_values[max_error_idx]:.3f}")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f}")

    # Find optimal l values
    direct_optimal_idx = np.argmax(direct_scores)
    recon_optimal_idx = np.argmax(reconstructed_scores)

    print(f"\nOptimal l:")
    print(f"  Direct: l*={l_values[direct_optimal_idx]:.3f}, S={direct_scores[direct_optimal_idx]:.2f}")
    print(f"  Reconstructed: l*={l_values[recon_optimal_idx]:.3f}, S={reconstructed_scores[recon_optimal_idx]:.2f}")

    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Score functions
    ax1.plot(l_values, direct_scores, 'b-', linewidth=2, label='Direct S(l)', marker='o', markersize=3)
    ax1.plot(l_values, reconstructed_scores, 'r--', linewidth=2, label='Faber-Schauder', marker='s', markersize=3)
    ax1.axvline(l_values[direct_optimal_idx], color='blue', linestyle=':', alpha=0.5, label=f'Direct l*={l_values[direct_optimal_idx]:.2f}')
    ax1.axvline(l_values[recon_optimal_idx], color='red', linestyle=':', alpha=0.5, label=f'Recon l*={l_values[recon_optimal_idx]:.2f}')
    ax1.set_xlabel('l (life parameter)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score S(l, X)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Score Function at state={state} (γ={gamma}, N={N}, j_max={j_max})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Absolute error
    ax2.plot(l_values, errors, 'r-', linewidth=2)
    ax2.axhline(np.mean(errors), color='orange', linestyle='--', label=f'Mean={np.mean(errors):.0f}')
    ax2.scatter([l_values[max_error_idx]], [errors[max_error_idx]], s=100, c='red', marker='*', zorder=5, label=f'Max={errors[max_error_idx]:.0f}')
    ax2.set_xlabel('l', fontsize=11, fontweight='bold')
    ax2.set_ylabel('|Direct - Reconstructed|', fontsize=11, fontweight='bold')
    ax2.set_title('Reconstruction Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Percent error
    pct_errors = (errors / np.abs(direct_scores)) * 100
    pct_errors = np.clip(pct_errors, 0, 500)  # Clip for visualization

    ax3.plot(l_values, pct_errors, 'm-', linewidth=2)
    ax3.axhline(100, color='red', linestyle='--', alpha=0.5, label='100% error')
    ax3.set_xlabel('l', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Percent Error (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Percent Reconstruction Error (clipped at 500%)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'results/faber_schauder_debug_state{int(state)}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if np.mean(errors) < 100:
        print("✓ Reconstruction is reasonably accurate (avg error < 100)")
    elif np.mean(errors) < 500:
        print("⚠️  Reconstruction has moderate errors (100 < avg error < 500)")
        print("   j_max might be too small - try increasing it")
    else:
        print("❌ Reconstruction has LARGE errors (avg error > 500)")
        print("   Faber-Schauder approximation is failing!")
        print("   Possible causes:")
        print("   1. j_max too small (need more basis functions)")
        print("   2. Score function is not smooth enough for this approximation")
        print("   3. Bug in Faber-Schauder coefficient computation")


if __name__ == "__main__":
    debug_single_state(state=0.0, gamma=0.9, N=50, j_max=5, num_samples=500)
