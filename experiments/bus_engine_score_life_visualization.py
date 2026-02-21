#!/usr/bin/env python
"""
Visualize Score-Life Functions for 16 States of Bus Engine Problem
With exact optimal state (2041) included
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import time

os.makedirs("results", exist_ok=True)

from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


def plot_score_life_functions(results, optimal_state, save_path):
    """Plot all score-life functions in a single figure."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    l_values = np.linspace(0, 1, 500)

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for idx, (state, score_func) in enumerate(results):
        ax = axes[idx]

        values = [score_func.compute_fractal(l) for l in l_values]

        ax.plot(l_values, values, color=colors[idx], linewidth=1.5)

        is_optimal = state == optimal_state
        title_color = "red" if is_optimal else "black"
        title_weight = "bold" if is_optimal else "normal"

        ax.set_title(
            f"State {int(state)}" + (" (Optimal)" if is_optimal else ""),
            color=title_color,
            fontweight=title_weight,
            fontsize=10,
        )
        ax.set_xlabel("l (life value)", fontsize=8)
        ax.set_ylabel("Score S(l, x)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    plt.suptitle(
        "Score-Life Functions for 16 Bus Engine States\n(Optimal state highlighted in red)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_all_overlay(results, optimal_state, save_path):
    """Plot all score-life functions overlaid on a single axes."""
    fig, ax = plt.subplots(figsize=(14, 9))

    l_values = np.linspace(0, 1, 500)

    colors = plt.cm.plasma(np.linspace(0, 1, len(results)))

    for idx, (state, score_func) in enumerate(results):
        values = [score_func.compute_fractal(l) for l in l_values]

        is_optimal = state == optimal_state
        linewidth = 3.5 if is_optimal else 1.2
        alpha = 1.0 if is_optimal else 0.5

        label = f"State {int(state)}"
        if is_optimal:
            label += " (Optimal: 2041)"

        ax.plot(
            l_values,
            values,
            color=colors[idx],
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )

    ax.set_xlabel("l (life value)", fontsize=12)
    ax.set_ylabel("Score S(l, x)", fontsize=12)
    ax.set_title(
        "Score-Life Functions for 16 Bus Engine States\n(Optimal state = 2041 miles highlighted)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    gamma = 0.60
    N = 16
    j_max = 8
    num_samples = 250
    p = 0.1
    q = 0.3

    optimal_state = 2041

    states = [
        0,
        250,
        500,
        750,
        1000,
        1250,
        1500,
        1750,
        2041,
        2500,
        3000,
        4000,
        5000,
        6500,
        8000,
        10000,
    ]

    print(f"Computing score-life functions for {len(states)} states...")
    print(f"States: {states}")
    print(f"Optimal state: {optimal_state}")

    results = []
    start_time = time.time()

    for state in states:
        env = BusEngineEnvironment(x=state, p=p, q=q)
        slp = ScoreLifeProgramming(env, gamma, N, j_max, num_samples, state)

        print(f"Computing score function for state {state}...")
        score_func = slp._compute_faber_schauder_coefficients()
        results.append((state, score_func))

    end_time = time.time()
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")

    plot_score_life_functions(
        results, optimal_state, "results/bus_engine_score_life_16states.png"
    )

    plot_all_overlay(
        results, optimal_state, "results/bus_engine_score_life_overlay.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
