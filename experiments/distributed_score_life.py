#!/usr/bin/env python
"""
Scale Score-Life to thousands of parallel workers using distributed computing.

Options:
1. Single machine: Limited to CPU cores (4-64)
2. Ray (distributed): Scale to 100s-1000s of cores
3. Dask: Cluster computing
4. Cloud batch: AWS Batch, Google Cloud, etc.
"""

import numpy as np
import multiprocessing as mp


def single_machine_limits():
    """Explain single-machine parallelization limits."""

    print("=" * 80)
    print("SINGLE MACHINE PARALLELIZATION")
    print("=" * 80)

    cpu_count = mp.cpu_count()

    print(f"""
Your current system: {cpu_count} CPU cores

REALITY CHECK:
═══════════════
• Python multiprocessing: Limited to your CPU cores ({cpu_count})
• Threading: Limited by GIL (Global Interpreter Lock) - NOT helpful for CPU-bound tasks
• Maximum useful workers: {cpu_count} (matches your cores)

Using more than {cpu_count} workers will HURT performance:
  ✗ Context switching overhead
  ✗ Memory contention
  ✗ Cache thrashing

Optimal configuration:
  n_workers = {cpu_count}  # All cores, no over-subscription

To get beyond {cpu_count} parallel workers, you need DISTRIBUTED computing.
""")


def ray_distributed_approach():
    """Show how to use Ray for distributed Score-Life."""

    print("\n" + "=" * 80)
    print("RAY: DISTRIBUTED SCORE-LIFE (100s-1000s of cores)")
    print("=" * 80)

    code = '''
# Install: pip install ray

import ray
import numpy as np
from src.environments.bus_engine_fixed import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

# Initialize Ray (local or connect to cluster)
ray.init()  # Local
# ray.init(address='ray://your-cluster-address:10001')  # Remote cluster

@ray.remote
def compute_state_value_distributed(state, gamma, N, num_samples, n_l_points, max_mileage):
    """Ray remote function - runs on any worker in cluster."""

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
    return {
        'state': state,
        'value': scores[max_idx],
        'optimal_l': l_values[max_idx]
    }


def run_distributed_score_life(n_states=10000):
    """Run on Ray cluster with 100s-1000s of workers."""

    states = np.linspace(0, 100000, n_states)

    # Submit ALL tasks at once - Ray handles scheduling across cluster
    futures = [
        compute_state_value_distributed.remote(
            state, gamma=0.9, N=30, num_samples=500,
            n_l_points=20, max_mileage=100000
        )
        for state in states
    ]

    # Collect results as they complete
    results = ray.get(futures)

    return results


# Run with massive parallelism
if __name__ == "__main__":
    # This will use ALL available cores in your Ray cluster
    # Could be 10, 100, or 1000+ cores depending on cluster size
    results = run_distributed_score_life(n_states=10000)
    print(f"Computed {len(results)} states using Ray cluster")

    ray.shutdown()


# CLUSTER SETUP:
# =============
# 1. On head node:
#    ray start --head --port=6379
#
# 2. On each worker node:
#    ray start --address='head-node-ip:6379'
#
# 3. Run your script on head node
#
# Ray will automatically distribute across all nodes!
'''

    print(code)

    print("""
RAY ADVANTAGES:
═══════════════
✓ Auto-scales to cluster size (10s-1000s of cores)
✓ Fault tolerance (retries failed tasks)
✓ Resource management (CPU/GPU/memory)
✓ Works on laptop or cloud cluster
✓ Dynamic task scheduling
✓ Built-in monitoring dashboard

SETUP:
═════
1. Install: pip install ray
2. Start cluster or use ray.init() for local
3. Replace multiprocessing with @ray.remote
4. Scale to 1000s of cores!

Performance with 100 cores:
  10,000 states  →  ~20-40 seconds (Fast config)
  100,000 states →  ~3-6 minutes (Fast config)
""")


def dask_approach():
    """Show Dask alternative."""

    print("\n" + "=" * 80)
    print("DASK: ALTERNATIVE DISTRIBUTED FRAMEWORK")
    print("=" * 80)

    code = '''
# Install: pip install dask distributed

from dask.distributed import Client, LocalCluster
import dask.bag as db

# Setup cluster
cluster = LocalCluster()  # Local cluster
# cluster = SLURMCluster()  # HPC cluster
# cluster = KubeCluster()   # Kubernetes
client = Client(cluster)

def compute_state(state):
    # Same worker function as before
    ...
    return result

# Create bag of states and map function
states = np.linspace(0, 100000, 10000)
bag = db.from_sequence(states, npartitions=1000)  # 1000 partitions
results = bag.map(compute_state).compute()

client.close()
'''

    print(code)
    print("\nDask is similar to Ray but integrates better with pandas/numpy workflows")


def cloud_batch_approach():
    """Show cloud batch processing."""

    print("\n" + "=" * 80)
    print("CLOUD BATCH PROCESSING (Max parallelism)")
    print("=" * 80)

    print("""
AWS BATCH / GOOGLE CLOUD:
═════════════════════════

1. Containerize your code (Docker)
2. Define array job with 10,000 tasks
3. Each task computes one state
4. Cloud spins up 100s-1000s of instances
5. Aggregate results when done

Example (AWS Batch):
  aws batch submit-job \\
    --job-name score-life-10000 \\
    --array-properties size=10000 \\
    --job-definition score-life-worker

Each job gets $ARRAY_INDEX:
  state = states[$AWS_BATCH_JOB_ARRAY_INDEX]

Cost: ~$0.001-0.01 per state (spot instances)
Time: 10,000 states in ~5-10 minutes with massive parallelism

WHEN TO USE:
  • One-off large computations
  • Don't have permanent cluster
  • OK with cloud costs
  • Need max parallelism (1000s of cores)
""")


def practical_recommendation():
    """Give practical advice."""

    cpu_count = mp.cpu_count()

    print("\n" + "=" * 80)
    print("PRACTICAL RECOMMENDATION")
    print("=" * 80)

    print(f"""
YOUR CURRENT SYSTEM: {cpu_count} cores

RECOMMENDATION BY SCALE:
═══════════════════════

1. SMALL (< 10,000 states):
   → Use multiprocessing (what we already have)
   → {cpu_count} workers
   → 10,000 states in ~1-5 minutes
   ✓ No additional setup needed

2. MEDIUM (10,000 - 100,000 states):
   → Ray on single machine
   → {cpu_count} workers (local cluster)
   → 100,000 states in ~10-50 minutes
   → Install: pip install ray
   → Code change: 5 minutes

3. LARGE (100,000+ states):
   → Ray cluster (multiple machines)
   → 10-100 workers across machines
   → 1M states in ~1-2 hours
   → Setup: 30 minutes
   → Cost: Free (your own machines)

4. MASSIVE (millions of states):
   → Cloud batch (AWS/Google/Azure)
   → 100-1000s of workers
   → 10M states in ~1-2 hours
   → Setup: 1-2 hours
   → Cost: $10-100 depending on instance types

FOR 10,000 STATES:
══════════════════
You DON'T need 10,000 parallel threads!

Your {cpu_count} cores can handle 10,000 states in ~1-5 minutes.

Parallel threads ≠ states to compute
  • Each thread processes MULTIPLE states
  • Optimal threads = CPU cores ({cpu_count})
  • Queue distributes work automatically

Current code (large_scale_parallel_scorelife.py) is OPTIMAL for your hardware.

ONLY go to distributed computing if:
  ✗ You need < 1 minute for 10,000 states, OR
  ✓ You have 100,000+ states, OR
  ✓ You have access to a cluster/cloud

Bottom line: Your current setup is perfect for 10,000 states!
""")


def show_work_distribution():
    """Visualize how work is distributed."""

    cpu_count = mp.cpu_count()
    n_states = 10000

    print("\n" + "=" * 80)
    print("HOW WORK DISTRIBUTION ACTUALLY WORKS")
    print("=" * 80)

    print(f"""
With {cpu_count} cores and {n_states} states:

Worker 1: states[0], states[{cpu_count}], states[{2*cpu_count}], ... (every {cpu_count}th)
Worker 2: states[1], states[{cpu_count+1}], states[{2*cpu_count+1}], ...
Worker 3: states[2], states[{cpu_count+2}], states[{2*cpu_count+2}], ...
Worker 4: states[3], states[{cpu_count+3}], states[{2*cpu_count+3}], ...

Each worker processes ~{n_states//cpu_count:,} states

This is AUTOMATIC via multiprocessing.Pool.map()

Adding more workers (> {cpu_count}) doesn't help:
  • They compete for same {cpu_count} CPU cores
  • Context switching overhead
  • Slower than optimal {cpu_count} workers

Think of it like checkout lanes:
  • {cpu_count} cashiers (CPU cores)
  • {n_states} customers (states)
  • Each cashier processes {n_states//cpu_count} customers
  • Hiring more cashiers doesn't help if you only have {cpu_count} registers!

To use 10,000 workers, you need 10,000 CPU cores (cluster/cloud).
""")


def main():
    single_machine_limits()
    ray_distributed_approach()
    dask_approach()
    cloud_batch_approach()
    practical_recommendation()
    show_work_distribution()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    cpu_count = mp.cpu_count()
    print(f"""
For 10,000 states on your {cpu_count}-core machine:
  ✓ Current code is OPTIMAL
  ✓ No need for 10,000 threads
  ✓ {cpu_count} workers is perfect
  ✓ Completes in 1-5 minutes

For 100,000+ states:
  → Install Ray: pip install ray
  → Modify code (see above)
  → Scale to cluster if available

For millions of states:
  → Use cloud batch processing
  → AWS/Google/Azure
  → Cost-effective for one-off jobs
""")


if __name__ == "__main__":
    main()
