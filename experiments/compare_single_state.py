#!/usr/bin/env python
"""
Compare VI value estimate with Score function for a SINGLE state.

Pick state=0, get its VI value, plot S(l,0) and horizontal line for VI.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def compute_vi_value_single_state(state, gamma=0.9, max_iterations=1000):
    """Compute VI value function for a single state."""

    print(f"Computing VI value for state={state}...")

    # Create a state grid
    max_mileage = 10000
    n_states = 100
    states = np.linspace(0, max_mileage, n_states)

    # Run value iteration
    V = np.zeros(n_states)
    tolerance = 1e-6

    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)

        for i, s in enumerate(states):
            # Q(s, keep)
            p, q = 0.1, 0.3
            expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500
            next_state = min(s + expected_delta, max_mileage)

            # Cost on NEXT state
            operating_cost = -0.01 * next_state

            # Interpolate V at next_state
            next_idx = np.argmin(np.abs(states - next_state))
            V_next = V[next_idx]

            Q_keep = operating_cost + gamma * V_next

            # Q(s, replace)
            Q_replace = -100 + gamma * V[0]

            V_new[i] = max(Q_keep, Q_replace)

        if np.max(np.abs(V_new - V)) < tolerance:
            print(f"  Converged in {iteration + 1} iterations")
            break

        V = V_new.copy()

    # Get value for target state
    state_idx = np.argmin(np.abs(states - state))
    vi_value = V[state_idx]

    print(f"  VI V({state}) = {vi_value:.2f}")

    return vi_value


def compute_score_function(state, gamma=0.9, N=50, num_samples=1000, n_points=100):
    """Compute Score function S(l, state) for many l values using DIRECT evaluation."""

    print(f"\nComputing Score function for state={state}...")
    print(f"  N={N}, num_samples={num_samples}, evaluating at {n_points} points")

    # CRITICAL: Match VI's max_mileage cap
    env = BusEngineEnvironment(max_state=10000)
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    # Sample l values densely
    l_values = np.linspace(0.0, 1.0, n_points)
    scores = []

    print(f"  Evaluating S(l, {state})...")
    for i, l in enumerate(l_values):
        if i % 20 == 0:
            print(f"    Progress: {i}/{n_points}")

        # Direct evaluation (NO Faber-Schauder)
        score = slp.S(l, np.array([state]))
        scores.append(score)

    scores = np.array(scores)

    # Find maximum
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]
    optimal_l = l_values[max_idx]

    print(f"  Direct S(l, {state}) maximum:")
    print(f"    l* = {optimal_l:.3f}")
    print(f"    max S = {max_score:.2f}")

    return l_values, scores, optimal_l, max_score


def compute_monte_carlo_samples(state, l_values_to_test, gamma=0.9, N=50, num_samples=1000, n_runs=20):
    """Run Score function multiple times at various l values to show Monte Carlo variance."""

    print(f"\nComputing Monte Carlo variance at {len(l_values_to_test)} l values...")
    print(f"  Running {n_runs} independent trials per l value...")

    # CRITICAL: Match VI's max_mileage cap
    env = BusEngineEnvironment(max_state=10000)

    mc_results = {}
    for l_val in l_values_to_test:
        samples = []
        for run in range(n_runs):
            env.set_state(state)
            slp = ScoreLifeProgramming(
                env, gamma=gamma, N=N, j_max=5,
                num_samples=num_samples,
                reference_state=np.array([state])
            )
            score = slp.S(l_val, np.array([state]))
            samples.append(score)

        mc_results[l_val] = np.array(samples)
        print(f"  l={l_val:.2f}: mean={np.mean(samples):.2f}, std={np.std(samples):.2f}")

    return mc_results


def compute_faber_schauder_reconstruction(state, gamma=0.9, N=50, num_samples=1000):
    """Compute Faber-Schauder reconstruction for comparison."""

    print(f"\nComputing Faber-Schauder reconstruction...")

    # CRITICAL: Match VI's max_mileage cap
    env = BusEngineEnvironment(max_state=10000)
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    score_func = slp._compute_faber_schauder_coefficients()

    # Sample reconstruction
    l_values = np.linspace(0.0, 1.0, 100)
    reconstructed = [score_func.compute_fractal(l) for l in l_values]

    return l_values, np.array(reconstructed)


def plot_comparison(state, l_values, scores, optimal_l, max_score, vi_value, gamma, N, mc_results, fs_l_values, fs_scores):
    """Create comprehensive comparison plot with Monte Carlo variance."""

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Plot 1: Full view - Direct vs Faber-Schauder vs VI
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(l_values, scores, 'b-', linewidth=3, label='Direct S(l, X) [Monte Carlo]', marker='o', markersize=4, alpha=0.8)
    ax1.plot(fs_l_values, fs_scores, 'purple', linewidth=2.5, linestyle='--', label='Faber-Schauder Reconstruction', alpha=0.7)
    ax1.axhline(vi_value, color='red', linestyle='--', linewidth=3,
               label=f'VI V(X) = {vi_value:.2f}', alpha=0.8)
    ax1.axvline(optimal_l, color='green', linestyle=':', linewidth=2,
               label=f'Direct l* = {optimal_l:.3f}', alpha=0.7)
    ax1.scatter([optimal_l], [max_score], s=200, c='green', marker='*',
               zorder=5, edgecolors='black', linewidth=2, label=f'Direct max S = {max_score:.2f}')

    ax1.set_xlabel('l (life parameter)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score S(l, X)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Score Function vs VI Value Estimate (state={state}, γ={gamma}, N={N})',
                 fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Zoomed around maximum
    ax2 = fig.add_subplot(gs[1, 0])
    margin = 0.2
    l_min_zoom = max(0, optimal_l - margin)
    l_max_zoom = min(1, optimal_l + margin)

    zoom_mask = (l_values >= l_min_zoom) & (l_values <= l_max_zoom)
    fs_zoom_mask = (fs_l_values >= l_min_zoom) & (fs_l_values <= l_max_zoom)

    ax2.plot(l_values[zoom_mask], scores[zoom_mask], 'b-', linewidth=3,
            label='Direct S(l, X)', marker='o', markersize=7, alpha=0.8)
    ax2.plot(fs_l_values[fs_zoom_mask], fs_scores[fs_zoom_mask], 'purple', linewidth=2.5,
            linestyle='--', label='Faber-Schauder', alpha=0.7)
    ax2.axhline(vi_value, color='red', linestyle='--', linewidth=3,
               label=f'VI V(X) = {vi_value:.2f}', alpha=0.8)
    ax2.axvline(optimal_l, color='green', linestyle=':', linewidth=2,
               label=f'l* = {optimal_l:.3f}', alpha=0.7)
    ax2.scatter([optimal_l], [max_score], s=250, c='green', marker='*',
               zorder=5, edgecolors='black', linewidth=2, label=f'max S = {max_score:.2f}')

    ax2.set_xlabel('l (life parameter)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score S(l, X)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Zoomed View Around Optimum', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Monte Carlo variance - box plot
    ax3 = fig.add_subplot(gs[1, 1])
    l_test_values = sorted(mc_results.keys())
    mc_data = [mc_results[l] for l in l_test_values]

    bp = ax3.boxplot(mc_data, positions=range(len(l_test_values)), widths=0.6,
                     patch_artist=True, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax3.set_xticks(range(len(l_test_values)))
    ax3.set_xticklabels([f'{l:.2f}' for l in l_test_values], fontsize=9)
    ax3.set_xlabel('l value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score S(l, X)', fontsize=12, fontweight='bold')
    ax3.set_title('Monte Carlo Variance (20 runs per l)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Raw Monte Carlo samples at optimal l
    ax4 = fig.add_subplot(gs[2, 0])
    optimal_l_key = min(l_test_values, key=lambda x: abs(x - optimal_l))
    optimal_samples = mc_results[optimal_l_key]

    ax4.plot(range(len(optimal_samples)), optimal_samples, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax4.axhline(np.mean(optimal_samples), color='green', linestyle='--', linewidth=2.5,
               label=f'Mean = {np.mean(optimal_samples):.2f}', alpha=0.8)
    ax4.axhline(vi_value, color='red', linestyle='--', linewidth=2.5,
               label=f'VI = {vi_value:.2f}', alpha=0.8)
    ax4.fill_between(range(len(optimal_samples)),
                     np.mean(optimal_samples) - np.std(optimal_samples),
                     np.mean(optimal_samples) + np.std(optimal_samples),
                     color='green', alpha=0.2, label=f'±1 std = {np.std(optimal_samples):.2f}')

    ax4.set_xlabel('Trial run', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score S(l, X)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Raw Monte Carlo Samples at l={optimal_l_key:.2f}', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Error between Direct and Faber-Schauder
    ax5 = fig.add_subplot(gs[2, 1])
    # Interpolate Faber-Schauder to match direct l_values
    fs_interp = interp1d(fs_l_values, fs_scores, kind='cubic', fill_value='extrapolate')
    fs_at_direct = fs_interp(l_values)

    error = np.abs(scores - fs_at_direct)
    pct_error = (error / np.abs(scores)) * 100

    ax5.plot(l_values, error, 'r-', linewidth=2.5, marker='o', markersize=4)
    ax5.axhline(np.mean(error), color='orange', linestyle='--', linewidth=2,
               label=f'Mean error = {np.mean(error):.2f}')
    ax5.set_xlabel('l', fontsize=12, fontweight='bold')
    ax5.set_ylabel('|Direct - Faber-Schauder|', fontsize=12, fontweight='bold')
    ax5.set_title('Faber-Schauder Reconstruction Error', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Monte Carlo std vs l
    ax6 = fig.add_subplot(gs[3, 0])
    stds = [np.std(mc_results[l]) for l in l_test_values]
    means = [np.mean(mc_results[l]) for l in l_test_values]

    ax6.plot(l_test_values, stds, 'mo-', linewidth=2.5, markersize=7, label='Standard deviation')
    ax6.set_xlabel('l', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax6.set_title('Monte Carlo Variance vs l', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Coefficient of variation
    ax7 = fig.add_subplot(gs[3, 1])
    cv = [(np.std(mc_results[l]) / abs(np.mean(mc_results[l]))) * 100 for l in l_test_values]

    ax7.plot(l_test_values, cv, 'co-', linewidth=2.5, markersize=7, label='CV = (std/mean) × 100%')
    ax7.set_xlabel('l', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax7.set_title('Relative Monte Carlo Variance', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    plt.suptitle(f'Complete Analysis: State={int(state)} miles (γ={gamma}, N={N})',
                fontsize=17, fontweight='bold', y=0.998)

    filename = f'results/score_vs_vi_state{int(state)}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")

    # Detailed analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)

    # VI vs Score-Life comparison
    difference = abs(max_score - vi_value)
    pct_diff = (difference / abs(vi_value)) * 100 if vi_value != 0 else 0

    print(f"\n1. VALUE FUNCTION COMPARISON:")
    print(f"   VI V({int(state)}):           {vi_value:.2f}")
    print(f"   Score-Life max S (Direct):    {max_score:.2f} at l*={optimal_l:.3f}")
    print(f"   Difference:                   {difference:.2f} ({pct_diff:.1f}%)")

    # Monte Carlo variance
    optimal_l_key = min(mc_results.keys(), key=lambda x: abs(x - optimal_l))
    optimal_samples = mc_results[optimal_l_key]
    mc_mean = np.mean(optimal_samples)
    mc_std = np.std(optimal_samples)
    mc_cv = (mc_std / abs(mc_mean)) * 100

    print(f"\n2. MONTE CARLO STATISTICS (at l={optimal_l_key:.2f}):")
    print(f"   Mean:                {mc_mean:.2f}")
    print(f"   Std Dev:             {mc_std:.2f}")
    print(f"   Coeff. of Variation: {mc_cv:.2f}%")
    print(f"   95% CI:              [{mc_mean - 1.96*mc_std:.2f}, {mc_mean + 1.96*mc_std:.2f}]")

    # Faber-Schauder reconstruction error
    fs_optimal_idx = np.argmax(fs_scores)
    fs_optimal_l = fs_l_values[fs_optimal_idx]
    fs_max_score = fs_scores[fs_optimal_idx]

    avg_fs_error = np.mean(error)
    max_fs_error = np.max(error)

    print(f"\n3. FABER-SCHAUDER RECONSTRUCTION:")
    print(f"   FS max S:            {fs_max_score:.2f} at l*={fs_optimal_l:.3f}")
    print(f"   Average error:       {avg_fs_error:.2f}")
    print(f"   Maximum error:       {max_fs_error:.2f}")
    print(f"   Optimal l mismatch:  {abs(fs_optimal_l - optimal_l):.3f}")

    # Overall verdict
    print(f"\n4. VERDICT:")
    if pct_diff < 10:
        print(f"   ✅ MATCH! Score-Life and VI agree within {pct_diff:.1f}%")
        print(f"      Monte Carlo variance is {mc_cv:.2f}% - well controlled")
    elif pct_diff < 50:
        print(f"   ⚠️  Moderate mismatch ({pct_diff:.1f}%)")
        print(f"      Potential causes:")
        if mc_cv > 5:
            print(f"      - High Monte Carlo variance ({mc_cv:.2f}%) - increase num_samples")
        if N < 100:
            print(f"      - N={N} may be too small for infinite-horizon approximation")
        if avg_fs_error > 100:
            print(f"      - Faber-Schauder reconstruction has large errors (avg={avg_fs_error:.0f})")
    else:
        print(f"   ❌ Large mismatch ({pct_diff:.1f}%)")
        print(f"      This suggests a fundamental difference in what's being computed")
        if avg_fs_error > 200:
            print(f"      - Faber-Schauder reconstruction is severely inaccurate!")
            print(f"      - Try increasing j_max or using direct optimization")

    print("\n" + "=" * 70)


def main():
    """Main execution."""

    # Pick an INTERMEDIATE state (not 0 or max)
    state = 5000.0
    gamma = 0.9
    N = 50
    num_samples = 1000

    print("=" * 70)
    print(f"SINGLE STATE COMPARISON: state={state} miles")
    print("=" * 70)

    # Get VI value
    vi_value = compute_vi_value_single_state(state, gamma=gamma)

    # Get Score function via direct evaluation
    l_values, scores, optimal_l, max_score = compute_score_function(
        state, gamma=gamma, N=N, num_samples=num_samples, n_points=50
    )

    # Get Monte Carlo variance at several l values
    l_values_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]
    if optimal_l not in l_values_to_test:
        # Add optimal_l to the test set
        l_values_to_test.append(optimal_l)
        l_values_to_test = sorted(l_values_to_test)

    mc_results = compute_monte_carlo_samples(
        state, l_values_to_test, gamma=gamma, N=N, num_samples=num_samples, n_runs=20
    )

    # Get Faber-Schauder reconstruction
    fs_l_values, fs_scores = compute_faber_schauder_reconstruction(
        state, gamma=gamma, N=N, num_samples=num_samples
    )

    # Plot comprehensive comparison
    plot_comparison(state, l_values, scores, optimal_l, max_score, vi_value,
                   gamma, N, mc_results, fs_l_values, fs_scores)


if __name__ == "__main__":
    main()
