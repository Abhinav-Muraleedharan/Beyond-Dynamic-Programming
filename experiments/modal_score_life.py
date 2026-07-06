#!/usr/bin/env python
"""
Scale Score-Life to 1000s of cores using Modal serverless compute.

Modal advantages:
- Zero infrastructure setup
- Auto-scales to 1000s of containers
- Pay per second of compute
- Dead simple API
"""

import modal
import numpy as np

# Create Modal app
app = modal.App("score-life-parallel")

# Define container image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "gymnasium",
        "matplotlib",
    )
    .copy_local_dir("src", "/root/src")  # Copy your source code
)


@app.function(
    image=image,
    cpu=1,  # 1 CPU per container
    memory=512,  # 512 MB per container
    timeout=600,  # 10 minute timeout per task
)
def compute_state_value(state_data):
    """
    Modal function - runs in isolated container.

    Each invocation gets its own container with 1 CPU.
    Modal handles scheduling, retries, and scaling.
    """
    import sys
    sys.path.insert(0, '/root')

    from src.environments.bus_engine_fixed import BusEngineEnvironment
    from src.utils.score_life_programming import ScoreLifeProgramming

    state, gamma, N, num_samples, n_l_points, max_mileage = state_data

    env = BusEngineEnvironment(max_state=max_mileage)
    env.set_state(state)

    slp = ScoreLifeProgramming(
        env, gamma=gamma, N=N, j_max=5,
        num_samples=num_samples,
        reference_state=np.array([state])
    )

    # Grid search over l
    l_values = np.linspace(0.0, 1.0, n_l_points)
    scores = [slp.S(l, np.array([state])) for l in l_values]

    max_idx = np.argmax(scores)
    return {
        'state': state,
        'value': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


@app.local_entrypoint()
def main(
    n_states: int = 10000,
    gamma: float = 0.9,
    N: int = 30,
    num_samples: int = 500,
    n_l_points: int = 20,
    max_mileage: float = 100000.0
):
    """
    Main function - runs on your local machine.
    Distributes work to Modal cloud.
    """
    import time

    print("=" * 80)
    print(f"MODAL SERVERLESS: {n_states:,} STATES")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  States:      {n_states:,}")
    print(f"  N (horizon): {N}")
    print(f"  MC samples:  {num_samples}")
    print(f"  l-points:    {n_l_points}")

    # Generate states
    states = np.linspace(0, max_mileage, n_states)

    # Prepare input data
    state_data = [
        (state, gamma, N, num_samples, n_l_points, max_mileage)
        for state in states
    ]

    print(f"\nSubmitting {n_states:,} tasks to Modal...")
    print("Modal will auto-scale to process these in parallel")

    start_time = time.time()

    # Map function across all states
    # Modal automatically:
    # - Spins up containers (up to 1000s)
    # - Distributes work
    # - Handles failures/retries
    # - Collects results
    results = list(compute_state_value.map(state_data))

    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    print(f"  Throughput: {n_states/elapsed:.2f} states/second")
    print(f"  Per-state:  {elapsed/n_states*1000:.2f} ms")

    # Analyze results
    values = np.array([r['value'] for r in results])

    print(f"\nValue Function:")
    print(f"  V(0):     {values[0]:.4f}")
    print(f"  V(max):   {values[-1]:.4f}")
    print(f"  Mean:     {np.mean(values):.4f}")

    # Save results
    np.savez_compressed(
        'results/modal_value_function.npz',
        states=states,
        values=values,
        optimal_ls=np.array([r['optimal_l'] for r in results]),
        metadata={
            'n_states': n_states,
            'gamma': gamma,
            'N': N,
            'elapsed': elapsed
        }
    )
    print(f"\n✓ Saved: results/modal_value_function.npz")

    # Estimate cost
    # Modal charges ~$0.00001 per CPU-second
    cpu_seconds = elapsed * min(n_states, 1000)  # Max 1000 parallel
    cost = cpu_seconds * 0.00001
    print(f"\nEstimated cost: ${cost:.4f}")


# Advanced: Batch processing for very large jobs
@app.function(
    image=image,
    cpu=4,  # Multi-core containers
    memory=4096,
    timeout=3600,
)
def compute_state_batch(batch_data):
    """Process multiple states per container for efficiency."""
    import sys
    sys.path.insert(0, '/root')

    from src.environments.bus_engine_fixed import BusEngineEnvironment
    from src.utils.score_life_programming import ScoreLifeProgramming

    states, gamma, N, num_samples, n_l_points, max_mileage = batch_data

    results = []
    for state in states:
        env = BusEngineEnvironment(max_state=max_mileage)
        env.set_state(state)

        slp = ScoreLifeProgramming(
            env, gamma=gamma, N=N, j_max=5,
            num_samples=num_samples,
            reference_state=np.array([state])
        )

        l_values = np.linspace(0.0, 1.0, n_l_points)
        scores = [slp.S(l, np.array([state])) for l in l_values]

        max_idx = np.argmax(scores)
        results.append({
            'state': state,
            'value': scores[max_idx],
            'optimal_l': l_values[max_idx]
        })

    return results


@app.local_entrypoint()
def main_batched(
    n_states: int = 100000,
    batch_size: int = 100,
    gamma: float = 0.9,
    N: int = 30,
    num_samples: int = 500,
    n_l_points: int = 20,
    max_mileage: float = 100000.0
):
    """
    Batched version for very large jobs (100k+ states).

    Groups states into batches, processes each batch in a 4-core container.
    More efficient for massive jobs.
    """
    import time

    print("=" * 80)
    print(f"MODAL BATCHED: {n_states:,} STATES (batches of {batch_size})")
    print("=" * 80)

    states = np.linspace(0, max_mileage, n_states)

    # Create batches
    n_batches = (n_states + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_states)
        batch_states = states[start_idx:end_idx]
        batches.append((batch_states, gamma, N, num_samples, n_l_points, max_mileage))

    print(f"\nCreated {n_batches} batches")
    print(f"Modal will process up to 1000 batches in parallel")

    start_time = time.time()

    # Process batches in parallel
    batch_results = list(compute_state_batch.map(batches))

    # Flatten results
    results = [r for batch in batch_results for r in batch]

    elapsed = time.time() - start_time

    print(f"\n✓ Completed {n_states:,} states in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"  Throughput: {n_states/elapsed:.2f} states/second")

    return results


if __name__ == "__main__":
    # For local testing (won't actually use Modal)
    print("Run with Modal CLI:")
    print("  modal run modal_score_life.py::main --n-states=10000")
    print("  modal run modal_score_life.py::main_batched --n-states=100000")
