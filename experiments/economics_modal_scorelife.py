#!/usr/bin/env python
"""
Capital Asset Replacement at Scale: Modal + Score-Life

Economics perspective on equipment replacement using Score-Life with 1000s of cores.

Problem: When should a firm replace its capital asset (bus engine)?
- State: Asset condition (mileage)
- Decision: Keep using vs Replace
- Costs: Maintenance (increasing) vs Replacement (fixed)
- Objective: Minimize expected discounted costs

Applications:
- Industrial organization (capital investment)
- Operations research (equipment replacement)
- Public economics (infrastructure management)

Usage:
  modal run experiments/economics_modal_scorelife.py::main --n-states=10000
  modal run experiments/economics_modal_scorelife.py::main --n-states=100000
"""

import modal
import numpy as np

# Create Modal app
app = modal.App("economics-asset-replacement")

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
def compute_asset_value(state_data):
    """
    Compute optimal replacement policy for a single asset condition level.

    Returns optimal expected cost at this mileage level.
    """
    import sys
    sys.path.insert(0, '/root')

    from src.environments.bus_engine_fixed import BusEngineEnvironment
    from src.utils.score_life_programming import ScoreLifeProgramming

    mileage, gamma, N, num_samples, n_l_points, max_mileage = state_data

    env = BusEngineEnvironment(max_state=max_mileage)
    env.set_state(mileage)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([mileage])
    )

    # Optimize over l parameter
    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([mileage])) for l in l_values]

    max_idx = np.argmax(scores)
    return {
        'mileage': mileage,
        'expected_cost': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


@app.local_entrypoint()
def main(
    n_states: int = 10000,
    gamma: float = 0.9,
    N: int = 50,
    num_samples: int = 3000,
    n_l_points: int = 75,
    max_mileage: float = 10000.0
):
    """
    Solve capital asset replacement problem at scale.

    Example:
        modal run economics_modal_scorelife.py::main --n-states=10000
        modal run economics_modal_scorelife.py::main --n-states=100000
    """
    import time

    print("=" * 80)
    print(f"CAPITAL ASSET REPLACEMENT: {n_states:,} MILEAGE LEVELS")
    print("=" * 80)

    print(f"\nEconomics Problem:")
    print(f"  - Firm owns capital asset (bus engine)")
    print(f"  - Asset condition: mileage ∈ [0, {max_mileage:,.0f}]")
    print(f"  - Decision: Keep using vs Replace")
    print(f"  - Costs: Maintenance (increasing) vs Replacement (fixed)")
    print(f"  - Objective: Minimize expected discounted costs")

    print(f"\nConfiguration:")
    print(f"  Mileage levels: {n_states:,}")
    print(f"  Discount:       γ = {gamma}")
    print(f"  Horizon:        N = {N}")
    print(f"  MC samples:     {num_samples}")
    print(f"  l-grid points:  {n_l_points}")

    # Generate mileage levels
    mileage_levels = np.linspace(0, max_mileage, n_states)

    # Prepare input data
    state_data = [
        (m, gamma, N, num_samples, n_l_points, max_mileage)
        for m in mileage_levels
    ]

    print(f"\nSubmitting {n_states:,} tasks to Modal cloud...")
    print("Modal will auto-scale to 1000s of parallel containers")

    start_time = time.time()

    # Distribute across Modal cloud
    results = list(compute_asset_value.map(state_data))

    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"  Throughput: {n_states/elapsed:.2f} mileage levels/second")

    # Analyze results
    costs = np.array([r['expected_cost'] for r in results])

    print(f"\nOptimal Replacement Policy:")
    print(f"  V(mileage=0):          {costs[0]:.2f}")
    print(f"  V(mileage={max_mileage:,.0f}): {costs[-1]:.2f}")
    print(f"  Mean expected cost:     {np.mean(costs):.2f}")

    # Find replacement threshold (approximately)
    # This is simplistic - real analysis would need policy extraction
    diffs = np.diff(costs)
    if len(diffs) > 0:
        threshold_idx = np.argmax(diffs) if max(diffs) > 0 else len(costs) - 1
        threshold = mileage_levels[min(threshold_idx, len(mileage_levels)-1)]
        print(f"  Approximate threshold:  {threshold:,.0f} miles")

    # Save
    np.savez_compressed(
        'results/asset_replacement_policy.npz',
        mileage_levels=mileage_levels,
        expected_costs=costs,
        optimal_ls=np.array([r['optimal_l'] for r in results]),
        metadata={
            'n_states': n_states,
            'gamma': gamma,
            'N': N,
            'num_samples': num_samples,
            'elapsed': elapsed
        }
    )
    print(f"\n✓ Saved: results/asset_replacement_policy.npz")

    # Estimate cost
    cpu_seconds = elapsed * min(n_states, 1000)
    cost = cpu_seconds * 0.00001
    print(f"\nEstimated Modal cost: ${cost:.4f}")
    print(f"  ({n_states:,} states × ~{elapsed/n_states:.3f}s each)")

    return results


if __name__ == "__main__":
    print("Run with Modal CLI:")
    print("  modal run economics_modal_scorelife.py::main --n-states=10000")
    print("  modal run economics_modal_scorelife.py::main --n-states=100000")
