#!/usr/bin/env python
"""
Enhanced Score-Life Function Visualization for Bus Engine Problem
- Multiple states with detailed analysis
- 3D surface plot showing state × life × score relationship
- Heatmap visualization
- Optimal life parameter identification
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

os.makedirs("results", exist_ok=True)


def compute_score_life_grid(states, gamma=0.60, N=16, j_max=6, num_samples=200):
    """Compute Score-Life functions for a grid of states."""
    print(f"Computing Score-Life functions for {len(states)} states...")

    l_values = np.linspace(0, 1, 200)
    score_grid = np.zeros((len(states), len(l_values)))

    for i, state in enumerate(states):
        print(f"  State {i+1}/{len(states)}: {state:.0f} miles")

        env = BusEngineEnvironment(x=state, p=0.1, q=0.3)
        slp = ScoreLifeProgramming(env, gamma, N, j_max, num_samples, state)

        # Compute Score-Life function
        score_func = slp._compute_faber_schauder_coefficients()

        # Evaluate on grid
        for j, l in enumerate(l_values):
            score_grid[i, j] = score_func.compute_fractal(l)

    return score_grid, l_values


def plot_3d_surface(states, l_values, score_grid):
    """Create 3D surface plot of Score-Life function."""
    print("\nCreating 3D surface visualization...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh
    L, S = np.meshgrid(l_values, states)

    # Plot surface
    surf = ax.plot_surface(L, S, score_grid, cmap='viridis',
                           alpha=0.9, edgecolor='none')

    ax.set_xlabel('Life Parameter (l)', fontsize=12)
    ax.set_ylabel('Engine Mileage (miles)', fontsize=12)
    ax.set_zlabel('Score S(l, x)', fontsize=12)
    ax.set_title('Score-Life Function Surface\n(State × Life × Score)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('results/bus_engine_score_life_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_3d.png")
    plt.close()


def plot_heatmap(states, l_values, score_grid):
    """Create heatmap of Score-Life function."""
    print("\nCreating heatmap visualization...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    im = ax.imshow(score_grid, aspect='auto', cmap='coolwarm',
                   origin='lower', interpolation='bilinear',
                   extent=[l_values[0], l_values[-1], states[0], states[-1]])

    ax.set_xlabel('Life Parameter (l)', fontsize=12)
    ax.set_ylabel('Engine Mileage (miles)', fontsize=12)
    ax.set_title('Score-Life Function Heatmap\n(Warmer colors = Higher score)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score S(l, x)', fontsize=12)

    # Add contour lines
    contours = ax.contour(l_values, states, score_grid,
                          levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    plt.tight_layout()
    plt.savefig('results/bus_engine_score_life_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_heatmap.png")
    plt.close()


def plot_optimal_life_curve(states, l_values, score_grid):
    """Plot optimal life parameter for each state."""
    print("\nAnalyzing optimal life parameters...")

    optimal_l = np.zeros(len(states))
    max_scores = np.zeros(len(states))

    for i in range(len(states)):
        max_idx = np.argmax(score_grid[i, :])
        optimal_l[i] = l_values[max_idx]
        max_scores[i] = score_grid[i, max_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Optimal life parameter vs state
    ax1.plot(states, optimal_l, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax1.set_ylabel('Optimal Life Parameter l*', fontsize=12)
    ax1.set_title('Optimal Life Parameter vs State', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Add horizontal line at l=0.5
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='l = 0.5 (midpoint)')
    ax1.legend()

    # Plot 2: Maximum score vs state
    ax2.plot(states, max_scores, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax2.set_ylabel('Maximum Score S(l*, x)', fontsize=12)
    ax2.set_title('Maximum Score vs State', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/bus_engine_score_life_optimal.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_optimal.png")
    plt.close()

    return optimal_l, max_scores


def plot_individual_traces(states, l_values, score_grid, highlight_states=None):
    """Plot individual Score-Life function traces for selected states."""
    print("\nCreating individual trace plots...")

    if highlight_states is None:
        # Select evenly spaced states to highlight
        indices = np.linspace(0, len(states)-1, min(8, len(states)), dtype=int)
        highlight_states = states[indices]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(highlight_states)))

    for i, state in enumerate(highlight_states):
        idx = np.abs(states - state).argmin()
        ax.plot(l_values, score_grid[idx, :],
                linewidth=2.5, color=colors[i],
                label=f'{state:.0f} miles', alpha=0.8)

    ax.set_xlabel('Life Parameter (l)', fontsize=12)
    ax.set_ylabel('Score S(l, x)', fontsize=12)
    ax.set_title('Score-Life Functions for Selected States',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig('results/bus_engine_score_life_traces.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_traces.png")
    plt.close()


def plot_score_derivative(states, l_values, score_grid):
    """Plot derivative of Score-Life function with respect to life parameter."""
    print("\nComputing score derivatives...")

    # Compute derivative using central differences
    dl = l_values[1] - l_values[0]
    score_derivative = np.gradient(score_grid, dl, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap of derivative
    im1 = ax1.imshow(score_derivative, aspect='auto', cmap='RdBu_r',
                     origin='lower', interpolation='bilinear',
                     extent=[l_values[0], l_values[-1], states[0], states[-1]])
    ax1.set_xlabel('Life Parameter (l)', fontsize=12)
    ax1.set_ylabel('Engine Mileage (miles)', fontsize=12)
    ax1.set_title('Score-Life Function Gradient ∂S/∂l', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('∂S/∂l', fontsize=12)

    # Sample traces
    indices = np.linspace(0, len(states)-1, 5, dtype=int)
    colors = plt.cm.plasma(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        ax2.plot(l_values, score_derivative[idx, :],
                linewidth=2, color=colors[i],
                label=f'{states[idx]:.0f} miles', alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Life Parameter (l)', fontsize=12)
    ax2.set_ylabel('∂S/∂l', fontsize=12)
    ax2.set_title('Score Gradient for Selected States', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/bus_engine_score_life_gradient.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_gradient.png")
    plt.close()


def create_comprehensive_dashboard(states, l_values, score_grid, optimal_l, max_scores):
    """Create a comprehensive dashboard with all key visualizations."""
    print("\nCreating comprehensive dashboard...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Heatmap
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im = ax1.imshow(score_grid, aspect='auto', cmap='coolwarm',
                    origin='lower', interpolation='bilinear',
                    extent=[l_values[0], l_values[-1], states[0], states[-1]])
    ax1.set_xlabel('Life Parameter (l)')
    ax1.set_ylabel('Engine Mileage (miles)')
    ax1.set_title('Score-Life Function Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Score S(l, x)')

    # 2. Optimal life parameter
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(states, optimal_l, 'o-', linewidth=2, markersize=6, color='blue')
    ax2.set_xlabel('Mileage')
    ax2.set_ylabel('Optimal l*')
    ax2.set_title('Optimal Life Parameter', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 3. Maximum scores
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(states, max_scores, 's-', linewidth=2, markersize=6, color='green')
    ax3.set_xlabel('Mileage')
    ax3.set_ylabel('Max Score')
    ax3.set_title('Maximum Score', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Individual traces
    ax4 = fig.add_subplot(gs[2, :])
    indices = np.linspace(0, len(states)-1, 6, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        ax4.plot(l_values, score_grid[idx, :],
                linewidth=2, color=colors[i],
                label=f'{states[idx]:.0f} mi', alpha=0.8)

    ax4.set_xlabel('Life Parameter (l)')
    ax4.set_ylabel('Score S(l, x)')
    ax4.set_title('Score-Life Function Traces', fontweight='bold')
    ax4.legend(loc='best', ncol=6, fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Bus Engine Score-Life Programming - Comprehensive Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('results/bus_engine_score_life_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: results/bus_engine_score_life_dashboard.png")
    plt.close()


def main():
    print("=" * 70)
    print("ENHANCED SCORE-LIFE FUNCTION VISUALIZATION")
    print("Bus Engine Replacement Problem")
    print("=" * 70)

    # Define state grid (20 states for detailed analysis)
    states = np.array([
        0, 500, 1000, 1500, 2000, 2041, 2500, 3000, 3500, 4000,
        5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 50000
    ])

    print(f"\nAnalyzing {len(states)} states:")
    print(f"Range: {states[0]:.0f} - {states[-1]:.0f} miles")
    print(f"Including optimal state: 2041 miles")

    # Parameters
    gamma = 0.60
    N = 16
    j_max = 6
    num_samples = 200

    print(f"\nParameters:")
    print(f"  Discount factor γ = {gamma}")
    print(f"  Life discretization N = {N}")
    print(f"  Fractal depth j_max = {j_max}")
    print(f"  Monte Carlo samples = {num_samples}")

    # Compute Score-Life grid
    start_time = time.time()
    score_grid, l_values = compute_score_life_grid(states, gamma, N, j_max, num_samples)
    computation_time = time.time() - start_time
    print(f"\nTotal computation time: {computation_time:.2f} seconds")

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. 3D Surface
    plot_3d_surface(states, l_values, score_grid)

    # 2. Heatmap
    plot_heatmap(states, l_values, score_grid)

    # 3. Optimal life analysis
    optimal_l, max_scores = plot_optimal_life_curve(states, l_values, score_grid)

    # 4. Individual traces
    highlight_states = [0, 1000, 2041, 3000, 5000, 10000, 20000, 50000]
    plot_individual_traces(states, l_values, score_grid, highlight_states)

    # 5. Score derivative
    plot_score_derivative(states, l_values, score_grid)

    # 6. Comprehensive dashboard
    create_comprehensive_dashboard(states, l_values, score_grid, optimal_l, max_scores)

    # Print analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nOptimal Life Parameters:")
    for i in [0, 5, 10, 15, 19]:  # Sample states
        print(f"  State {states[i]:>6.0f} miles: l* = {optimal_l[i]:.4f}, "
              f"max score = {max_scores[i]:.2f}")

    # Find state with highest max score
    best_state_idx = np.argmax(max_scores)
    print(f"\nState with highest maximum score:")
    print(f"  {states[best_state_idx]:.0f} miles with score {max_scores[best_state_idx]:.2f}")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. bus_engine_score_life_3d.png - 3D surface plot")
    print("  2. bus_engine_score_life_heatmap.png - Score heatmap")
    print("  3. bus_engine_score_life_optimal.png - Optimal life curves")
    print("  4. bus_engine_score_life_traces.png - Individual function traces")
    print("  5. bus_engine_score_life_gradient.png - Score gradients")
    print("  6. bus_engine_score_life_dashboard.png - Comprehensive dashboard")


if __name__ == "__main__":
    main()
