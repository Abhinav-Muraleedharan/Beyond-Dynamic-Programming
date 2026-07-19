#!/usr/bin/env python
"""Quick plot from gamma sweep results already computed."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Results from the completed gamma sweep
results = [
    {'gamma': 0.3, 'corr': 0.9987, 'mean_diff': -0.28, 'policy_agreement': 100.0, 'time_vi': 1.40, 'time_sl': 453.46},
    {'gamma': 0.5, 'corr': 0.9973, 'mean_diff': -1.90, 'policy_agreement': 96.7, 'time_vi': 2.35, 'time_sl': 453.43},
    {'gamma': 0.7, 'corr': 0.9960, 'mean_diff': -5.87, 'policy_agreement': 86.7, 'time_vi': 4.07, 'time_sl': 448.68},
    {'gamma': 0.9, 'corr': 0.9931, 'mean_diff': -15.77, 'policy_agreement': 60.0, 'time_vi': 13.87, 'time_sl': 446.58},
]

gammas = [r['gamma'] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation vs gamma
ax1 = axes[0, 0]
ax1.plot(gammas, [r['corr'] for r in results], 'o-', linewidth=3, markersize=12,
        color='blue', label='Value Correlation')
ax1.axhline(0.99, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (0.99)')
ax1.set_xlabel('Discount Factor (γ)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Correlation Coefficient', fontsize=13, fontweight='bold')
ax1.set_title('Value Function Agreement vs Gamma', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.99, 1.0)

# Plot 2: Offset vs gamma
ax2 = axes[0, 1]
ax2.plot(gammas, [abs(r['mean_diff']) for r in results], 'o-', linewidth=3,
        markersize=12, color='purple', label='|Mean Offset|')
ax2.axhline(5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (±5)')
ax2.set_xlabel('Discount Factor (γ)', fontsize=13, fontweight='bold')
ax2.set_ylabel('|Mean Offset| (units)', fontsize=13, fontweight='bold')
ax2.set_title('Value Function Offset vs Gamma', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Policy agreement vs gamma
ax3 = axes[1, 0]
ax3.plot(gammas, [r['policy_agreement'] for r in results], 'o-', linewidth=3,
        markersize=12, color='green', label='Policy Agreement')
ax3.axhline(95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (95%)')
ax3.set_xlabel('Discount Factor (γ)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Agreement (%)', fontsize=13, fontweight='bold')
ax3.set_title('Policy Agreement vs Gamma', fontsize=14, fontweight='bold')
ax3.set_ylim(50, 105)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Computation time comparison
ax4 = axes[1, 1]
x = np.arange(len(gammas))
width = 0.35
bars1 = ax4.bar(x - width/2, [r['time_vi'] for r in results], width,
               label='Value Iteration', color='blue', alpha=0.7)
bars2 = ax4.bar(x + width/2, [r['time_sl'] for r in results], width,
               label='Score-Life (4 cores)', color='red', alpha=0.7)

# Add time labels on bars
for i, r in enumerate(results):
    ax4.text(i - width/2, r['time_vi'] + 5, f"{r['time_vi']:.1f}s",
            ha='center', va='bottom', fontsize=9)
    ax4.text(i + width/2, r['time_sl'] + 5, f"{r['time_sl']/60:.1f}m",
            ha='center', va='bottom', fontsize=9)

ax4.set_xlabel('Discount Factor (γ)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax4.set_title('Computation Time by Gamma (30 states)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{g}' for g in gammas])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('Gamma Sweep: How Discount Factor Affects VI vs Score-Life Agreement',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

filename = 'results/gamma_sweep_quick.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {filename}")

# Print summary
print("\n" + "=" * 80)
print("GAMMA SWEEP RESULTS SUMMARY")
print("=" * 80)
print("\nKey Finding: Lower gamma → Better agreement (shorter effective horizon)")
print("\n{:<10} {:<12} {:<15} {:<18} {:<10}".format("Gamma", "Correlation", "Offset", "Policy Agree", "VI Time"))
print("-" * 80)
for r in results:
    print("{:<10} {:<12.4f} {:<15.2f} {:<18.1f}% {:<10.2f}s".format(
        r['gamma'], r['corr'], abs(r['mean_diff']), r['policy_agreement'], r['time_vi']))

print("\nInterpretation:")
print("  γ=0.3: Nearly perfect (r=0.999, 100% policy, offset=0.3)")
print("  γ=0.5: Excellent (r=0.997, 96.7% policy, offset=1.9) ← RECOMMENDED")
print("  γ=0.7: Good (r=0.996, 86.7% policy, offset=5.9)")
print("  γ=0.9: Moderate (r=0.993, 60% policy, offset=15.8)")
print("\nConclusion: Use gamma ≤ 0.5 for ±5 error target")
