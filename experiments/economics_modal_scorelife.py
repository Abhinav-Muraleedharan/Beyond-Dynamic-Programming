#!/usr/bin/env python
"""
Economics Problems at Scale: Modal + Score-Life

Demonstrates Score-Life on economics MDPs with 1000s of cores:
- Consumption-savings (macro)
- Inventory management (operations)
- Portfolio optimization (finance)

Usage:
  modal run experiments/economics_modal_scorelife.py::consumption --n-states=10000
  modal run experiments/economics_modal_scorelife.py::inventory --n-states=10000
"""

import modal
import numpy as np

# Create Modal app
app = modal.App("economics-score-life")

# Define container image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "gymnasium",
    )
    .copy_local_dir("src", "/root/src")
)


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=600,
)
def compute_consumption_value(state_data):
    """
    Compute optimal consumption-savings policy for a single wealth level.

    Returns optimal lifetime utility at this wealth level.
    """
    import sys
    sys.path.insert(0, '/root')

    from src.environments.consumption_savings import ConsumptionSavingsEnvironment
    from src.utils.score_life_programming import ScoreLifeProgramming

    wealth, gamma, N, num_samples, n_l_points, max_wealth = state_data

    env = ConsumptionSavingsEnvironment(max_wealth=max_wealth)
    env.set_state(wealth)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([wealth])
    )

    # Optimize over l parameter
    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([wealth])) for l in l_values]

    max_idx = np.argmax(scores)
    return {
        'wealth': wealth,
        'lifetime_utility': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


@app.function(
    image=image,
    cpu=1,
    memory=512,
    timeout=600,
)
def compute_inventory_value(state_data):
    """
    Compute optimal inventory policy for a single inventory level.

    Returns optimal expected profit at this inventory level.
    """
    import sys
    sys.path.insert(0, '/root')

    from src.environments.consumption_savings import InventoryManagementEnvironment
    from src.utils.score_life_programming import ScoreLifeProgramming

    inventory, gamma, N, num_samples, n_l_points, max_inventory = state_data

    env = InventoryManagementEnvironment(max_inventory=max_inventory)
    env.set_state(inventory)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([inventory])
    )

    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([inventory])) for l in l_values]

    max_idx = np.argmax(scores)
    return {
        'inventory': inventory,
        'expected_profit': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


@app.local_entrypoint()
def consumption(
    n_states: int = 10000,
    gamma: float = 0.95,
    N: int = 30,
    num_samples: int = 500,
    n_l_points: int = 20,
    max_wealth: float = 1000.0
):
    """
    Solve consumption-savings problem at scale.

    Example:
        modal run economics_modal_scorelife.py::consumption --n-states=10000
    """
    import time

    print("=" * 80)
    print(f"CONSUMPTION-SAVINGS PROBLEM: {n_states:,} WEALTH LEVELS")
    print("=" * 80)

    print(f"\nProblem:")
    print(f"  - Agent has wealth W ∈ [0, {max_wealth}]")
    print(f"  - Chooses consumption C")
    print(f"  - Saves S = W - C with stochastic returns")
    print(f"  - Maximizes lifetime utility u(C)")

    print(f"\nConfiguration:")
    print(f"  Wealth levels: {n_states:,}")
    print(f"  Discount:      γ = {gamma}")
    print(f"  Horizon:       N = {N}")
    print(f"  MC samples:    {num_samples}")

    # Generate wealth levels
    wealth_levels = np.linspace(0, max_wealth, n_states)

    # Prepare input data
    state_data = [
        (w, gamma, N, num_samples, n_l_points, max_wealth)
        for w in wealth_levels
    ]

    print(f"\nSubmitting {n_states:,} tasks to Modal...")

    start_time = time.time()

    # Distribute across Modal cloud
    results = list(compute_consumption_value.map(state_data))

    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"  Throughput: {n_states/elapsed:.2f} wealth levels/second")

    # Analyze results
    utilities = np.array([r['lifetime_utility'] for r in results])

    print(f"\nOptimal Consumption Policy:")
    print(f"  V(wealth=$0):     {utilities[0]:.4f}")
    print(f"  V(wealth=${max_wealth}): {utilities[-1]:.4f}")
    print(f"  Mean utility:      {np.mean(utilities):.4f}")

    # Save
    np.savez_compressed(
        'results/consumption_savings_policy.npz',
        wealth_levels=wealth_levels,
        lifetime_utilities=utilities,
        optimal_ls=np.array([r['optimal_l'] for r in results])
    )
    print(f"\n✓ Saved: results/consumption_savings_policy.npz")

    # Estimate cost
    cpu_seconds = elapsed * min(n_states, 1000)
    cost = cpu_seconds * 0.00001
    print(f"\nEstimated cost: ${cost:.4f}")

    return results


@app.local_entrypoint()
def inventory(
    n_states: int = 10000,
    gamma: float = 0.90,
    N: int = 30,
    num_samples: int = 500,
    n_l_points: int = 20,
    max_inventory: float = 500.0
):
    """
    Solve inventory management problem at scale.

    Example:
        modal run economics_modal_scorelife.py::inventory --n-states=10000
    """
    import time

    print("=" * 80)
    print(f"INVENTORY MANAGEMENT: {n_states:,} INVENTORY LEVELS")
    print("=" * 80)

    print(f"\nProblem:")
    print(f"  - Current inventory I ∈ [0, {max_inventory}]")
    print(f"  - Choose order quantity Q")
    print(f"  - Face stochastic demand D")
    print(f"  - Minimize costs (holding + shortage + ordering)")

    print(f"\nConfiguration:")
    print(f"  Inventory levels: {n_states:,}")
    print(f"  Discount:         γ = {gamma}")
    print(f"  Horizon:          N = {N}")
    print(f"  MC samples:       {num_samples}")

    # Generate inventory levels
    inventory_levels = np.linspace(0, max_inventory, n_states)

    state_data = [
        (inv, gamma, N, num_samples, n_l_points, max_inventory)
        for inv in inventory_levels
    ]

    print(f"\nSubmitting {n_states:,} tasks to Modal...")

    start_time = time.time()

    results = list(compute_inventory_value.map(state_data))

    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"  Throughput: {n_states/elapsed:.2f} inventory levels/second")

    # Analyze
    profits = np.array([r['expected_profit'] for r in results])

    print(f"\nOptimal Inventory Policy:")
    print(f"  V(inventory=0):   {profits[0]:.2f}")
    print(f"  V(inventory={max_inventory}): {profits[-1]:.2f}")
    print(f"  Mean profit:       {np.mean(profits):.2f}")

    # Save
    np.savez_compressed(
        'results/inventory_policy.npz',
        inventory_levels=inventory_levels,
        expected_profits=profits,
        optimal_ls=np.array([r['optimal_l'] for r in results])
    )
    print(f"\n✓ Saved: results/inventory_policy.npz")

    cpu_seconds = elapsed * min(n_states, 1000)
    cost = cpu_seconds * 0.00001
    print(f"\nEstimated cost: ${cost:.4f}")

    return results


if __name__ == "__main__":
    print("Run with Modal CLI:")
    print("  modal run economics_modal_scorelife.py::consumption --n-states=10000")
    print("  modal run economics_modal_scorelife.py::inventory --n-states=10000")
