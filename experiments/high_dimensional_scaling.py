#!/usr/bin/env python
"""
Demonstrate Score-Life scaling to high-dimensional state spaces.

Shows:
1. Memory scaling with dimensionality
2. Computational scaling
3. Practical limits
4. Function approximation strategies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def memory_requirements(n_dims, j_max=5, n_states=100):
    """Calculate memory needed for different dimensions."""

    # Faber-Schauder coefficients per dimension
    coeffs_per_dim = sum(2**j for j in range(j_max)) + 2

    # Total coefficients for tensor product (naive)
    # This would be coeffs_per_dim^n_dims (exponential - infeasible!)

    # Practical: Store separately per dimension (additive factorization)
    practical_coeffs = n_dims * coeffs_per_dim

    # Memory in MB
    bytes_per_coeff = 8  # float64
    memory_mb = (practical_coeffs * n_states * bytes_per_coeff) / (1024**2)

    return {
        'coeffs_per_dim': coeffs_per_dim,
        'practical_coeffs': practical_coeffs,
        'memory_mb': memory_mb
    }


def vi_grid_size(n_dims, points_per_dim=10):
    """Calculate grid size for VI discretization."""

    # Total grid points = points_per_dim^n_dims
    # This grows exponentially - quickly becomes impossible

    if n_dims <= 3:
        total_points = points_per_dim ** n_dims
    else:
        # Use log representation to avoid overflow
        log_points = n_dims * np.log10(points_per_dim)
        total_points = None  # Too large

    memory_mb = None
    if total_points and total_points < 1e9:
        memory_mb = (total_points * 8) / (1024**2)

    return {
        'n_dims': n_dims,
        'points_per_dim': points_per_dim,
        'total_points': total_points,
        'memory_mb': memory_mb,
        'log10_points': n_dims * np.log10(points_per_dim) if n_dims > 3 else None
    }


def compare_scaling():
    """Compare VI vs Score-Life scaling with dimensions."""

    print("=" * 80)
    print("HIGH-DIMENSIONAL SCALING ANALYSIS")
    print("=" * 80)

    dimensions = [1, 2, 3, 5, 10, 50, 100, 1000, 10000]

    print("\nVALUE ITERATION (Discretization Grid):")
    print("=" * 80)
    print(f"{'Dims':<8} {'Points/Dim':<12} {'Total Points':<20} {'Memory':<15}")
    print("-" * 80)

    for n_dims in dimensions:
        vi = vi_grid_size(n_dims, points_per_dim=10)

        if vi['total_points']:
            points_str = f"{vi['total_points']:,.0f}"
            memory_str = f"{vi['memory_mb']:.2f} MB"
        else:
            points_str = f"10^{vi['log10_points']:.1f}"
            memory_str = "INFEASIBLE"

        print(f"{n_dims:<8} {vi['points_per_dim']:<12} {points_str:<20} {memory_str:<15}")

    print("\n" + "=" * 80)
    print("SCORE-LIFE (Factored Faber-Schauder):")
    print("=" * 80)
    print(f"{'Dims':<8} {'Coeffs/Dim':<12} {'Total Coeffs':<15} {'Memory (100 states)':<20}")
    print("-" * 80)

    for n_dims in dimensions:
        sl = memory_requirements(n_dims, j_max=5, n_states=100)
        print(f"{n_dims:<8} {sl['coeffs_per_dim']:<12} {sl['practical_coeffs']:<15,} "
              f"{sl['memory_mb']:<20.2f} MB")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Memory comparison
    ax1 = axes[0, 0]
    dims_feasible = [1, 2, 3, 5]
    vi_memory = [vi_grid_size(d)['memory_mb'] for d in dims_feasible]
    sl_memory = [memory_requirements(d)['memory_mb'] for d in dims_feasible]

    x = np.arange(len(dims_feasible))
    width = 0.35
    ax1.bar(x - width/2, vi_memory, width, label='VI (10 pts/dim)', alpha=0.8)
    ax1.bar(x + width/2, sl_memory, width, label='Score-Life (factored)', alpha=0.8)
    ax1.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory: Feasible Range (≤5-D)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims_feasible)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Score-Life scaling to high dimensions
    ax2 = axes[0, 1]
    all_dims = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    sl_all_memory = [memory_requirements(d)['memory_mb'] for d in all_dims]

    ax2.plot(all_dims, sl_all_memory, 'r-o', linewidth=3, markersize=8)
    ax2.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Score-Life: Scaling to 10000-D', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axhline(1000, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label='1 GB')
    ax2.legend(fontsize=10)

    # Plot 3: VI infeasibility
    ax3 = axes[1, 0]
    dims_vi = np.arange(1, 11)
    log_points = dims_vi * np.log10(10)

    ax3.semilogy(dims_vi, 10**log_points, 'b-o', linewidth=3, markersize=8)
    ax3.axhline(1e9, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Practical limit (~1B states)')
    ax3.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Grid Points (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('VI Grid Explosion', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Complexity comparison table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_text = """
COMPLEXITY SUMMARY

Value Iteration:
  Grid points: n^d (exponential in d)
  Memory:      O(n^d)
  Feasible:    d ≤ 3-4 dimensions

Score-Life (Factored):
  Coefficients: d × 2^j_max (linear in d)
  Memory:       O(d × 2^j_max)
  Feasible:     d ≤ 1000s dimensions*

  *With function approximation

10000-D Reality:
  ✗ VI:         10^10000 grid points
  ✓ SL:         ~620,000 coefficients
  ✓ Deep SL:    Neural net (millions params)

Practical Strategies:
  1. Dimensionality reduction
  2. Factored representations
  3. Deep learning approximation
  4. Exploit problem structure
"""

    ax4.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Scaling to High-Dimensional State Spaces',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = 'results/high_dimensional_scaling.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def practical_recommendations():
    """Print practical advice for high-dimensional problems."""

    print("\n" + "=" * 80)
    print("PRACTICAL RECOMMENDATIONS FOR 10000-D STATE SPACES")
    print("=" * 80)

    advice = """
1. QUESTION YOUR DIMENSIONALITY
   ═══════════════════════════════
   • Are all 10000 dimensions truly independent?
   • Can you exploit structure (symmetry, hierarchy, sparsity)?
   • Is there redundancy that can be compressed?

   Example: 1000 robot joints → group into limbs → ~10-20 groups

2. DIMENSIONALITY REDUCTION
   ═══════════════════════════
   A. PCA/Factor Analysis
      - Reduce 10000-D → k-D (k = 10-100) preserving 95-99% variance
      - Apply Score-Life in reduced space

   B. Autoencoders
      - Train VAE to compress state representation
      - Learn latent space dynamics

   C. Feature Engineering
      - Domain knowledge to extract key features
      - Often better than automated reduction

3. FACTORED REPRESENTATIONS
   ═══════════════════════════
   If V(x₁,...,x₁₀₀₀₀) has structure:

   A. Additive: V(x) ≈ Σᵢ Vᵢ(xᵢ)
      - Linear in dimensions
      - Score-Life per dimension independently

   B. Hierarchical: V(x) ≈ V_coarse(f(x)) + V_fine(x)
      - Multi-scale approach
      - Coarse policy in low-D, fine-tune in high-D

   C. Graph-structured: V(x) = Σ V_clique(x_subset)
      - Exploit conditional independence
      - Graphical model structure

4. DEEP SCORE-LIFE (Research Direction)
   ════════════════════════════════════
   Replace Faber-Schauder with neural network:

   class DeepScoreLife:
       def __init__(self, state_dim=10000):
           self.net = MLP([state_dim, 512, 256, 128, 1])

       def S(self, l, x):
           return self.net(np.concatenate([x, [l]]))

   Train via policy gradient + Score-Life objective

5. HYBRID APPROACHES
   ══════════════════
   • Use VI/Score-Life for critical low-D subspaces
   • Learned policy for high-D state-action
   • Model-based + model-free combination

6. REALISTIC EXPECTATIONS
   ═══════════════════════
   Dimension   Approach                         Feasible?
   ─────────────────────────────────────────────────────
   1-10        Tabular VI or Score-Life         ✓ Yes
   10-100      Score-Life with good sampling    ✓ Yes
   100-1000    Factored Score-Life              ✓ With structure
   1000-10000  Deep RL + Score-Life objective   ⚠ Research area
   10000+      Requires strong assumptions      ⚠ Very challenging

7. YOUR SPECIFIC CASE: 10000-D
   ════════════════════════════
   Questions to ask:
   • What is the problem domain? (robotics, finance, physics?)
   • Is there natural grouping/hierarchy?
   • How many dimensions truly vary independently?
   • Can you learn a low-D embedding?

   Recommended path:
   1. Start with dimensionality reduction (aim for <100-D)
   2. Apply parallel Score-Life (using large_scale_parallel_scorelife.py)
   3. If reduction loses critical info → factored representation
   4. If still intractable → deep learning approximation

BOTTOM LINE:
═══════════
Raw 10000-D is not directly tractable for ANY classical method.
Success depends on exploiting problem structure.

Score-Life advantages at scale:
  ✓ No discretization (continuous state space)
  ✓ Sample-based (learns from trajectories)
  ✓ Embarrassingly parallel (scales to many cores)
  ✓ Graceful degradation (quality vs samples tradeoff)

But you MUST combine with:
  • Dimensionality reduction, OR
  • Factored representation, OR
  • Deep learning approximation
"""

    print(advice)


def main():
    compare_scaling()
    practical_recommendations()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Describe your 10000-D problem domain")
    print("2. Identify potential structure/factorization")
    print("3. Try dimensionality reduction first")
    print("4. Start with reduced space + parallel Score-Life")
    print("\nThe tools in large_scale_parallel_scorelife.py work for any")
    print("dimension after reduction to feasible size (~10-100 D).")


if __name__ == "__main__":
    main()
