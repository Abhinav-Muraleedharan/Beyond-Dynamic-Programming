#!/usr/bin/env python
"""
Extract Policy from Existing Score-Life Results

Uses the max_score values from the definitive comparison as V(X) estimates
and extracts the threshold policy.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')


def extract_policy_from_score_results(gamma_value):
    """Extract policy from existing Score-Life results for a given gamma."""

    # Load results
    with open('results/bus_engine_definitive_comparison.json', 'r') as f:
        results = json.load(f)

    # Find results for this gamma
    target_results = None
    for result in results:
        if abs(result['gamma'] - gamma_value) < 0.01:
            target_results = result
            break

    if target_results is None:
        print(f"No results found for γ={gamma_value}")
        return

    # Extract Score-Life data
    sl_data = target_results['SL']['analysis']
    states = np.array([item['state'] for item in sl_data])
    V_estimates = np.array([item['max_score'] for item in sl_data])

    print(f"\nγ = {gamma_value}")
    print("=" * 70)
    print("\nValue Function Estimates from Score-Life:")
    print(f"{'State (miles)':<15} {'V(X)':<15}")
    print("-" * 30)
    for state, v in zip(states, V_estimates):
        print(f"{state:<15.0f} {v:<15.2f}")

    # Interpolate V(X)
    V_interp = interp1d(states, V_estimates, kind='cubic', fill_value='extrapolate')

    # Create dense state grid
    state_grid = np.linspace(0, 10000, 1000)
    V_grid = V_interp(state_grid)

    # Extract policy by comparing Q-values
    # Q(X, replace) = -100 + γ * V(0)
    # Q(X, keep) = -cost(X) + γ * E[V(X')]

    Q_replace = -100 + gamma_value * V_interp(0)

    Q_keep_values = []
    policy = []

    # Environment parameters
    p, q = 0.1, 0.3
    replacement_cost = 100
    operating_cost_rate = 0.01

    for state in state_grid:
        # Operating cost
        operating_cost = -operating_cost_rate * state

        # Expected next state (approximate with mean of transition distribution)
        # E[X'] = X + E[ΔX]
        # E[ΔX] = p*500 + q*2000 + (1-p-q)*6500
        expected_delta = p * 500 + q * 2000 + (1 - p - q) * 6500

        # Clip next state to max
        next_state = min(state + expected_delta, states[-1])

        # Q(X, keep)
        Q_keep = operating_cost + gamma_value * V_interp(next_state)
        Q_keep_values.append(Q_keep)

        # Policy: replace if Q(replace) > Q(keep)
        action = 1 if Q_replace > Q_keep else 0
        policy.append(action)

    policy = np.array(policy)
    Q_keep_values = np.array(Q_keep_values)

    # Find threshold (first state where we switch to replace)
    replace_indices = np.where(policy == 1)[0]
    if len(replace_indices) > 0:
        threshold_sl = state_grid[replace_indices[0]]
    else:
        threshold_sl = state_grid[-1]

    # Get VI threshold for comparison
    vi_threshold = target_results['VI']['threshold']
    vi_reward = target_results['VI']['reward']

    print(f"\n{'Method':<25} {'Threshold (miles)':<20} {'Avg Reward':<15}")
    print("-" * 60)
    print(f"{'Value Iteration':<25} {vi_threshold:<20.0f} {vi_reward:<15.2f}")
    print(f"{'Score-Life (extracted)':<25} {threshold_sl:<20.0f} {'N/A':<15}")
    print(f"{'Difference':<25} {abs(vi_threshold - threshold_sl):<20.0f} {abs(vi_threshold - threshold_sl)/vi_threshold*100:.1f}%")

    # Create visualization
    create_visualization(state_grid, V_grid, policy, Q_keep_values, Q_replace,
                        threshold_sl, vi_threshold, gamma_value, states, V_estimates)

    return threshold_sl, vi_threshold


def create_visualization(states, V, policy, Q_keep, Q_replace_value,
                         threshold_sl, threshold_vi, gamma, sampled_states, sampled_V):
    """Create visualization comparing Score-Life and VI policies."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Value Function with sampled points
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(states, V, 'b-', linewidth=2, label='V(X) interpolated', alpha=0.7)
    ax1.scatter(sampled_states, sampled_V, c='red', s=100, zorder=5,
               label='Sampled states', edgecolors='black', linewidth=1.5)
    ax1.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2,
               label=f'Score-Life threshold={threshold_sl:.0f} mi')
    ax1.axvline(threshold_vi, color='green', linestyle='--', linewidth=2,
               label=f'VI threshold={threshold_vi:.0f} mi')
    ax1.set_xlabel('Mileage (miles)', fontsize=12)
    ax1.set_ylabel('Value Function V(X)', fontsize=12)
    ax1.set_title(f'Value Function from Score-Life (γ={gamma})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Policy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between([0, threshold_sl], 0, 1, alpha=0.3, color='blue', label='Keep (SL)')
    ax2.fill_between([threshold_sl, states[-1]], 0, 1, alpha=0.3, color='orange', label='Replace (SL)')
    ax2.axvline(threshold_vi, color='green', linestyle='--', linewidth=3,
               label=f'VI={threshold_vi:.0f}', alpha=0.7)
    ax2.set_xlim([0, 10000])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Mileage (miles)', fontsize=12)
    ax2.set_title('Policy Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Q-values
    ax3 = fig.add_subplot(gs[1, :2])
    Q_replace_array = np.full_like(states, Q_replace_value)
    ax3.plot(states, Q_keep, 'b-', linewidth=2, label='Q(X, keep)')
    ax3.plot(states, Q_replace_array, 'r--', linewidth=2, label='Q(X, replace)')
    ax3.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(threshold_vi, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Mileage (miles)', fontsize=12)
    ax3.set_ylabel('Q-value', fontsize=12)
    ax3.set_title('Q-values: Keep vs Replace', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Advantage function
    ax4 = fig.add_subplot(gs[1, 2])
    advantage = Q_keep - Q_replace_array
    ax4.plot(states, advantage, 'm-', linewidth=2)
    ax4.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax4.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'SL={threshold_sl:.0f}')
    ax4.axvline(threshold_vi, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'VI={threshold_vi:.0f}')
    ax4.fill_between(states, 0, advantage, where=(advantage > 0),
                    alpha=0.3, color='blue', label='Keep better')
    ax4.fill_between(states, 0, advantage, where=(advantage <= 0),
                    alpha=0.3, color='red', label='Replace better')
    ax4.set_xlabel('Mileage (miles)', fontsize=12)
    ax4.set_ylabel('Advantage', fontsize=12)
    ax4.set_title('A(X) = Q(keep) - Q(replace)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Policy extracted
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(states, policy, 'g-', linewidth=3)
    ax5.axvline(threshold_sl, color='orange', linestyle='--', linewidth=2,
               label=f'Score-Life={threshold_sl:.0f} mi')
    ax5.axvline(threshold_vi, color='green', linestyle='--', linewidth=2,
               label=f'VI={threshold_vi:.0f} mi')
    ax5.set_xlabel('Mileage (miles)', fontsize=12)
    ax5.set_ylabel('Action (0=Keep, 1=Replace)', fontsize=12)
    ax5.set_title('Extracted Policy from Score-Life', fontsize=13, fontweight='bold')
    ax5.set_ylim([-0.1, 1.1])
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary stats
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    diff = abs(threshold_sl - threshold_vi)
    pct_diff = diff / threshold_vi * 100

    summary_text = f"""
    SUMMARY (γ={gamma})
    {'='*30}

    Value Iteration:
      Threshold: {threshold_vi:.0f} miles

    Score-Life (extracted):
      Threshold: {threshold_sl:.0f} miles

    Difference:
      Absolute: {diff:.0f} miles
      Relative: {pct_diff:.1f}%

    Conclusion:
    {'Policies match closely!' if pct_diff < 10 else 'Policies differ significantly'}
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Policy Extraction from Score-Life Programming (γ={gamma})',
                fontsize=15, fontweight='bold', y=0.995)

    filename = f'results/policy_extraction_gamma{gamma:.2f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("EXTRACTING POLICIES FROM SCORE-LIFE RESULTS")
    print("=" * 70)

    # Test all gamma values
    gammas = [0.6, 0.8, 0.9, 0.95, 0.99]

    results_summary = []

    for gamma in gammas:
        threshold_sl, threshold_vi = extract_policy_from_score_results(gamma)
        results_summary.append({
            'gamma': gamma,
            'threshold_sl': threshold_sl,
            'threshold_vi': threshold_vi,
            'difference': abs(threshold_sl - threshold_vi),
            'pct_difference': abs(threshold_sl - threshold_vi) / threshold_vi * 100
        })

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: SCORE-LIFE vs VALUE ITERATION")
    print("=" * 70)
    print(f"\n{'γ':<8} {'VI Threshold':<15} {'SL Threshold':<15} {'Diff (mi)':<12} {'Diff (%)':<10}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['gamma']:<8.2f} {r['threshold_vi']:<15.0f} {r['threshold_sl']:<15.0f} "
              f"{r['difference']:<12.0f} {r['pct_difference']:<10.1f}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    avg_pct_diff = np.mean([r['pct_difference'] for r in results_summary])
    print(f"Average difference: {avg_pct_diff:.1f}%")

    if avg_pct_diff < 10:
        print("✅ Score-Life and Value Iteration produce SIMILAR policies!")
    elif avg_pct_diff < 25:
        print("⚠️  Score-Life and Value Iteration produce SOMEWHAT DIFFERENT policies")
    else:
        print("❌ Score-Life and Value Iteration produce SIGNIFICANTLY DIFFERENT policies")

    print("\nPolicies have been extracted and visualized in results/ directory.")


if __name__ == "__main__":
    main()
