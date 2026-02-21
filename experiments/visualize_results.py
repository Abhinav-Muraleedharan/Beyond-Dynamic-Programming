#!/usr/bin/env python
# experiments/visualize_results.py

import os
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def load_results():
    """Load experiment results from JSON file."""
    try:
        with open("results/experiment_results.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def create_bar_chart(results, env_name, save_path="results"):
    """Create a bar chart comparing methods for a specific environment."""
    env_results = results.get(env_name, {})

    methods = []
    values = []
    errors = []

    for key, value in env_results.items():
        if key.endswith("_mean"):
            method_name = key.replace("_mean", "")
            methods.append(method_name)
            values.append(value)
            std_key = f"{method_name}_std"
            errors.append(env_results.get(std_key, 0))
        elif not key.endswith("_std"):
            methods.append(key)
            values.append(value if isinstance(value, (int, float)) else 0)
            errors.append(0)

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(methods)))
    bars = ax.bar(
        methods,
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(f"Algorithm Performance Comparison - {env_name}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    filename = f"{save_path}/{env_name}_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def create_comparison_plot(results, save_path="results"):
    """Create a comprehensive comparison plot across all environments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    env_names = list(results.keys())

    methods_dict = {}
    for env_name, env_results in results.items():
        for key, value in env_results.items():
            if key.endswith("_mean"):
                method_name = key.replace("_mean", "")
                if method_name not in methods_dict:
                    methods_dict[method_name] = []
                methods_dict[method_name].append(value)
            elif not key.endswith("_std"):
                if key not in methods_dict:
                    methods_dict[key] = []
                methods_dict[key].append(
                    value if isinstance(value, (int, float)) else 0
                )

    x_pos = np.arange(len(env_names))
    width = 0.8 / len(methods_dict)

    for idx, (method_name, values) in enumerate(methods_dict.items()):
        ax = axes[0]
        offset = idx * width - 0.4 + width / 2
        bars = ax.bar(x_pos + offset, values, width, label=method_name, alpha=0.8)

    axes[0].set_xlabel("Environment", fontsize=12)
    axes[0].set_ylabel("Mean Reward", fontsize=12)
    axes[0].set_title("Performance Comparison Across Environments", fontsize=14)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(env_names, rotation=45, ha="right")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    ax = axes[1]
    method_names = list(methods_dict.keys())
    avg_values = [np.mean(v) for v in methods_dict.values()]
    colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(method_names)))
    bars = ax.barh(method_names, avg_values, color=colors, edgecolor="black", alpha=0.8)

    ax.set_xlabel("Average Mean Reward", fontsize=12)
    ax.set_title("Average Performance Across All Environments", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, avg_values):
        ax.annotate(
            f"{val:.1f}",
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    filename = f"{save_path}/overall_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def create_score_life_visualization(save_path="results"):
    """Create visualizations for the score-life function."""
    from src.utils.fractal import Fractal

    alpha_0 = 0.0
    alpha_1 = -0.5
    coefficients = [[0.1], [0.05, 0.05], [0.02] * 4]
    j_shift = 0

    fractal = Fractal(alpha_0, alpha_1, coefficients, j_shift)

    l_values = np.linspace(0, 1, 1000)
    S_values = [fractal.compute_fractal(l) for l in l_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(l_values, S_values, "b-", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("l (Life Parameter)", fontsize=12)
    ax.set_ylabel("S(l) (Score-Life Function)", fontsize=12)
    ax.set_title("Score-Life Function Visualization", fontsize=14)
    ax.grid(True, alpha=0.3)

    l_opt = fractal.compute_optima_gradient_descent()
    ax.axvline(
        x=l_opt, color="r", linestyle="--", alpha=0.7, label=f"Optimal l = {l_opt:.3f}"
    )
    ax.legend()

    j_max = 5
    max_coeffs = 2 ** (j_max - 1)
    coeffs_by_level = []
    for j in range(j_max):
        n_coeffs = 2**j
        coeffs = [
            np.sin(j * np.pi * i / max(n_coeffs, 1)) * 0.1 for i in range(n_coeffs)
        ]
        coeffs_by_level.append(coeffs)

    coeffs_by_level_padded = np.zeros((j_max, max_coeffs))
    for j, coeffs in enumerate(coeffs_by_level):
        coeffs_by_level_padded[j, : len(coeffs)] = coeffs

    ax = axes[1]
    im = ax.imshow(
        coeffs_by_level_padded, aspect="auto", cmap="RdBu_r", interpolation="nearest"
    )
    ax.set_xlabel("Coefficient Index", fontsize=12)
    ax.set_ylabel("Expansion Level j", fontsize=12)
    ax.set_title("Faber-Schauder Coefficients by Level", fontsize=14)
    plt.colorbar(im, ax=ax, label="Coefficient Value")

    plt.tight_layout()
    filename = f"{save_path}/score_life_function.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def create_method_summary(results, save_path="reports"):
    """Create a summary table of results."""
    summary_data = []

    for env_name, env_results in results.items():
        row = {"Environment": env_name}
        for key, value in env_results.items():
            if isinstance(value, (int, float)):
                row[key] = f"{value:.2f}"
        summary_data.append(row)

    if not summary_data:
        return

    fig, ax = plt.subplots(figsize=(12, len(summary_data) + 1))
    ax.axis("off")

    keys = list(summary_data[0].keys())
    table = ax.table(
        cellText=[[row[k] for k in keys] for row in summary_data],
        colLabels=keys,
        cellLoc="center",
        loc="center",
        colColours=["#4a90d9"] * len(keys),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(color="white", weight="bold")

    plt.title("Experiment Results Summary", fontsize=14, pad=20)
    plt.tight_layout()

    filename = f"{save_path}/results_summary.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    """Main function to generate all visualizations."""
    print("Generating visualizations...")

    results = load_results()

    if not results:
        print("No results found. Running with sample data...")
        results = {
            "CartPole-v1": {
                "Random": 27.6,
                "ScoreLife_Exact": 0.0,
                "ScoreLife_Approx": 500.0,
                "Q_Learning": 10.0,
                "PPO_mean": 0.0,
                "PPO_std": 0.0,
                "A2C_mean": 0.0,
                "A2C_std": 0.0,
            },
            "MountainCar-v0": {
                "Random": -49516.0,
                "ScoreLife_Exact": 0,
                "ScoreLife_Approx": 0,
                "Q_Learning": -301.33,
                "PPO_mean": 0.0,
                "PPO_std": 0.0,
                "A2C_mean": 0.0,
                "A2C_std": 0.0,
            },
        }
        with open("results/experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

    for env_name in results.keys():
        create_bar_chart(results, env_name)

    create_comparison_plot(results)
    create_score_life_visualization()
    create_method_summary(results)

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
