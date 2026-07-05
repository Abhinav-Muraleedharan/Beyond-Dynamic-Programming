#!/usr/bin/env python
"""
Score-Life Function Analysis Around Replacement Thresholds

Compare Score-Life functions before and after:
1. Value Iteration threshold (~2,525 miles)
2. Score-Life Programming threshold (~5,556 miles)

Understand what changes at the threshold and why policies differ.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def analyze_threshold_region(states_around_threshold, threshold_name, gamma=0.60, N=16, j_max=6, num_samples=200):
    """Analyze Score-Life functions around a threshold."""

    print(f"\n{'='*70}")
    print(f"Analyzing {threshold_name}")
    print(f"{'='*70}")

    l_values = np.linspace(0, 1, 300)
    score_functions = []

    for state in states_around_threshold:
        print(f"  Computing Score-Life for {state:.0f} miles...")
        env = BusEngineEnvironment(x=state, p=0.1, q=0.3)
        slp = ScoreLifeProgramming(env, gamma, N, j_max, num_samples, state)
        score_func = slp._compute_faber_schauder_coefficients()

        # Evaluate on grid
        scores = np.array([score_func.compute_fractal(l) for l in l_values])
        score_functions.append(scores)

    return l_values, np.array(score_functions)


def plot_threshold_comparison(l_values, states, scores_before, scores_after, threshold_value, threshold_name):
    """Create detailed comparison plot for one threshold."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Color schemes
    colors_before = plt.cm.Blues(np.linspace(0.3, 0.9, len(scores_before)))
    colors_after = plt.cm.Reds(np.linspace(0.3, 0.9, len(scores_after)))

    # 1. Score-Life functions BEFORE threshold
    ax1 = fig.add_subplot(gs[0, 0])
    for i, score in enumerate(scores_before):
        ax1.plot(l_values, score, linewidth=2.5, color=colors_before[i],
                label=f'{states[i]:.0f} mi', alpha=0.8)
    ax1.set_xlabel('Life Parameter (l)', fontsize=11)
    ax1.set_ylabel('Score S(l, x)', fontsize=11)
    ax1.set_title(f'BEFORE Threshold\n(Keep Running Region)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

    # 2. Score-Life functions AFTER threshold
    ax2 = fig.add_subplot(gs[0, 1])
    for i, score in enumerate(scores_after):
        ax2.plot(l_values, score, linewidth=2.5, color=colors_after[i],
                label=f'{states[len(scores_before) + i]:.0f} mi', alpha=0.8)
    ax2.set_xlabel('Life Parameter (l)', fontsize=11)
    ax2.set_ylabel('Score S(l, x)', fontsize=11)
    ax2.set_title(f'AFTER Threshold\n(Replace Region)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

    # 3. Overlay comparison
    ax3 = fig.add_subplot(gs[0, 2])
    for i, score in enumerate(scores_before):
        ax3.plot(l_values, score, linewidth=2, color=colors_before[i],
                alpha=0.6, linestyle='-')
    for i, score in enumerate(scores_after):
        ax3.plot(l_values, score, linewidth=2, color=colors_after[i],
                alpha=0.6, linestyle='--')
    ax3.set_xlabel('Life Parameter (l)', fontsize=11)
    ax3.set_ylabel('Score S(l, x)', fontsize=11)
    ax3.set_title('Overlay: Blue=Before, Red=After', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)

    # 4. Optimal life parameter progression
    ax4 = fig.add_subplot(gs[1, 0])
    optimal_l = []
    max_scores = []
    for score in np.vstack([scores_before, scores_after]):
        max_idx = np.argmax(score)
        optimal_l.append(l_values[max_idx])
        max_scores.append(score[max_idx])

    ax4.plot(states, optimal_l, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax4.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold_value:.0f} mi')
    ax4.set_xlabel('Engine Mileage (miles)', fontsize=11)
    ax4.set_ylabel('Optimal l*', fontsize=11)
    ax4.set_title('Optimal Life Parameter vs State', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    # 5. Maximum score progression
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(states, max_scores, 's-', linewidth=2.5, markersize=10, color='green')
    ax5.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2,
               label=f'Threshold: {threshold_value:.0f} mi')
    ax5.set_xlabel('Engine Mileage (miles)', fontsize=11)
    ax5.set_ylabel('Maximum Score', fontsize=11)
    ax5.set_title('Maximum Score vs State', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 6. Score difference (after - before at same l)
    ax6 = fig.add_subplot(gs[1, 2])
    # Compare first of "after" with last of "before"
    if len(scores_before) > 0 and len(scores_after) > 0:
        score_diff = scores_after[0] - scores_before[-1]
        ax6.plot(l_values, score_diff, linewidth=2.5, color='darkviolet')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.fill_between(l_values, 0, score_diff, where=(score_diff > 0),
                        color='red', alpha=0.3, label='After > Before')
        ax6.fill_between(l_values, 0, score_diff, where=(score_diff < 0),
                        color='blue', alpha=0.3, label='Before > After')
        ax6.set_xlabel('Life Parameter (l)', fontsize=11)
        ax6.set_ylabel('Score Difference', fontsize=11)
        ax6.set_title(f'Δ Score: {states[len(scores_before)]:.0f}mi - {states[len(scores_before)-1]:.0f}mi',
                     fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)

    # 7. Score gradient before threshold
    ax7 = fig.add_subplot(gs[2, 0])
    dl = l_values[1] - l_values[0]
    for i, score in enumerate(scores_before):
        gradient = np.gradient(score, dl)
        ax7.plot(l_values, gradient, linewidth=2, color=colors_before[i],
                label=f'{states[i]:.0f} mi', alpha=0.7)
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Life Parameter (l)', fontsize=11)
    ax7.set_ylabel('∂S/∂l', fontsize=11)
    ax7.set_title('Score Gradient - BEFORE Threshold', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9, loc='best')
    ax7.grid(True, alpha=0.3)

    # 8. Score gradient after threshold
    ax8 = fig.add_subplot(gs[2, 1])
    for i, score in enumerate(scores_after):
        gradient = np.gradient(score, dl)
        ax8.plot(l_values, gradient, linewidth=2, color=colors_after[i],
                label=f'{states[len(scores_before) + i]:.0f} mi', alpha=0.7)
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Life Parameter (l)', fontsize=11)
    ax8.set_ylabel('∂S/∂l', fontsize=11)
    ax8.set_title('Score Gradient - AFTER Threshold', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9, loc='best')
    ax8.grid(True, alpha=0.3)

    # 9. Heatmap: Score vs Life vs State
    ax9 = fig.add_subplot(gs[2, 2])
    all_scores = np.vstack([scores_before, scores_after])
    im = ax9.imshow(all_scores, aspect='auto', cmap='RdYlBu_r',
                   origin='lower', interpolation='bilinear',
                   extent=[l_values[0], l_values[-1], states[0], states[-1]])
    ax9.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2)
    ax9.set_xlabel('Life Parameter (l)', fontsize=11)
    ax9.set_ylabel('Engine Mileage (miles)', fontsize=11)
    ax9.set_title('Score Heatmap Across Threshold', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax9)
    cbar.set_label('Score', fontsize=10)

    plt.suptitle(f'Score-Life Function Analysis: {threshold_name}\n' +
                 f'Threshold at {threshold_value:.0f} miles',
                 fontsize=16, fontweight='bold', y=0.995)

    filename = f"results/bus_engine_threshold_{threshold_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()


def analyze_action_values_at_threshold(states, threshold_value, threshold_name):
    """Analyze what makes the policy switch at the threshold."""

    print(f"\n{'='*70}")
    print(f"Action Value Analysis: {threshold_name}")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Parameters
    gamma = 0.99  # Use VI's gamma for fair comparison
    replacement_cost = 100
    operating_cost_rate = 0.01
    p = 0.1
    q = 0.3

    keep_values = []
    replace_values = []
    q_differences = []

    for state in states:
        # Expected cost of KEEP (simplified)
        expected_next_mileage = state + (
            p * 500 +  # E[Uniform(0,1000)]
            q * 2000 + # E[Uniform(1000,3000)]
            (1-p-q) * 6500  # E[Uniform(3000,10000)]
        )
        keep_cost = operating_cost_rate * expected_next_mileage

        # Cost of REPLACE
        replace_cost = replacement_cost

        keep_values.append(-keep_cost)
        replace_values.append(-replace_cost)
        q_differences.append(-keep_cost - (-replace_cost))

    # Plot 1: Q-values
    ax = axes[0, 0]
    ax.plot(states, keep_values, 'b-o', linewidth=2.5, markersize=8, label='Q(s, Keep)')
    ax.plot(states, replace_values, 'r-s', linewidth=2.5, markersize=8, label='Q(s, Replace)')
    ax.axvline(x=threshold_value, color='green', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold_value:.0f} mi')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Q-value (immediate cost)', fontsize=12)
    ax.set_title('Action Values: Keep vs Replace', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Q-value difference
    ax = axes[0, 1]
    ax.plot(states, q_differences, 'purple', linewidth=2.5, marker='o', markersize=8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=threshold_value, color='green', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold_value:.0f} mi')
    ax.fill_between(states, 0, q_differences, where=(np.array(q_differences) > 0),
                    color='blue', alpha=0.3, label='Keep Better')
    ax.fill_between(states, 0, q_differences, where=(np.array(q_differences) <= 0),
                    color='red', alpha=0.3, label='Replace Better')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Q(Keep) - Q(Replace)', fontsize=12)
    ax.set_title('Q-Value Difference (Positive = Keep Better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Expected costs breakdown
    ax = axes[1, 0]
    expected_mileages = [state + (p * 500 + q * 2000 + (1-p-q) * 6500) for state in states]
    operating_costs = [operating_cost_rate * m for m in expected_mileages]

    ax.plot(states, operating_costs, 'b-o', linewidth=2.5, markersize=8,
           label='Expected Operating Cost (Keep)')
    ax.axhline(y=replacement_cost, color='red', linestyle='-', linewidth=2.5,
              label=f'Replacement Cost: ${replacement_cost}')
    ax.axvline(x=threshold_value, color='green', linestyle='--', linewidth=2,
              label=f'Threshold: {threshold_value:.0f} mi')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Cost Comparison: Operating vs Replacement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 4: Policy recommendation
    ax = axes[1, 1]
    policy = [0 if q > 0 else 1 for q in q_differences]
    ax.step(states, policy, where='post', linewidth=3, color='darkgreen')
    ax.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2,
              label=f'Actual Threshold: {threshold_value:.0f} mi')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Keep Running', 'Replace'], fontsize=11)
    ax.set_title('Implied Policy (From Immediate Costs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Action Value Analysis: {threshold_name}\n' +
                 f'Why does policy switch at {threshold_value:.0f} miles?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f"results/bus_engine_action_values_{threshold_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    print("=" * 70)
    print("SCORE-LIFE FUNCTION THRESHOLD ANALYSIS")
    print("Understanding Policy Changes at Thresholds")
    print("=" * 70)

    # Define thresholds from our previous experiments
    vi_threshold = 2525  # Value Iteration threshold
    sl_threshold = 5556  # Score-Life threshold

    # States to analyze around each threshold
    # VI threshold analysis: states from 1500 to 3500 miles
    vi_states_before = [1500, 1800, 2000, 2200, 2400]
    vi_states_after = [2600, 2800, 3000, 3200, 3500]
    vi_all_states = vi_states_before + vi_states_after

    # SL threshold analysis: states from 4000 to 7000 miles
    sl_states_before = [4000, 4500, 5000, 5300, 5500]
    sl_states_after = [5600, 5800, 6000, 6500, 7000]
    sl_all_states = sl_states_before + sl_states_after

    # Parameters for Score-Life computation
    gamma = 0.60
    N = 16
    j_max = 6
    num_samples = 200

    # ========== Value Iteration Threshold Analysis ==========
    print("\n" + "=" * 70)
    print("ANALYZING VALUE ITERATION THRESHOLD (~2,525 miles)")
    print("=" * 70)

    l_vals_vi, scores_vi = analyze_threshold_region(
        vi_all_states, "Value Iteration Threshold", gamma, N, j_max, num_samples
    )

    plot_threshold_comparison(
        l_vals_vi, vi_all_states,
        scores_vi[:len(vi_states_before)],
        scores_vi[len(vi_states_before):],
        vi_threshold,
        "Value Iteration Threshold"
    )

    analyze_action_values_at_threshold(vi_all_states, vi_threshold, "VI Threshold")

    # ========== Score-Life Threshold Analysis ==========
    print("\n" + "=" * 70)
    print("ANALYZING SCORE-LIFE THRESHOLD (~5,556 miles)")
    print("=" * 70)

    l_vals_sl, scores_sl = analyze_threshold_region(
        sl_all_states, "Score-Life Threshold", gamma, N, j_max, num_samples
    )

    plot_threshold_comparison(
        l_vals_sl, sl_all_states,
        scores_sl[:len(sl_states_before)],
        scores_sl[len(sl_states_before):],
        sl_threshold,
        "Score-Life Threshold"
    )

    analyze_action_values_at_threshold(sl_all_states, sl_threshold, "SL Threshold")

    # ========== Summary Analysis ==========
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)

    print(f"\nValue Iteration Threshold: {vi_threshold} miles")
    print(f"  - States analyzed: {min(vi_all_states):.0f} to {max(vi_all_states):.0f} miles")
    print(f"  - Before: {vi_states_before}")
    print(f"  - After: {vi_states_after}")

    print(f"\nScore-Life Threshold: {sl_threshold} miles")
    print(f"  - States analyzed: {min(sl_all_states):.0f} to {max(sl_all_states):.0f} miles")
    print(f"  - Before: {sl_states_before}")
    print(f"  - After: {sl_states_after}")

    print(f"\nThreshold Difference: {sl_threshold - vi_threshold} miles ({(sl_threshold/vi_threshold - 1)*100:.1f}% higher)")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. bus_engine_threshold_value_iteration_threshold.png")
    print("  2. bus_engine_action_values_vi_threshold.png")
    print("  3. bus_engine_threshold_score-life_threshold.png")
    print("  4. bus_engine_action_values_sl_threshold.png")
    print("\nThese show how Score-Life functions change before/after each threshold.")


if __name__ == "__main__":
    main()
