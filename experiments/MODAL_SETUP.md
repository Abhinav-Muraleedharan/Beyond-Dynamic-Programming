# Modal Setup for Score-Life

Modal is **the easiest way** to scale Score-Life to 1000s of cores.

## Setup (2 minutes)

### 1. Install Modal

```bash
pip install modal
```

### 2. Create Free Account & Authenticate

```bash
modal setup
```

This opens your browser to create a free Modal account ($30/month free credits).

### 3. Run Your First Job

```bash
# 10,000 states (Fast config)
modal run experiments/modal_score_life.py::main --n-states=10000 --N=20 --num-samples=200

# 100,000 states (Batched)
modal run experiments/modal_score_life.py::main_batched --n-states=100000 --batch-size=100
```

That's it! Modal handles everything:
- ✓ Spins up containers in the cloud
- ✓ Distributes your 10,000 tasks
- ✓ Auto-scales to 1000s of parallel containers
- ✓ Returns results to your laptop
- ✓ Shuts down when done (pay only for compute time)

## Performance Comparison

| Method | Setup Time | 10k States | 100k States | Cores |
|--------|-----------|------------|-------------|-------|
| **Local (multiprocessing)** | 0 min | 2-5 min | 20-50 min | 4 |
| **Ray (local)** | 5 min | 2-5 min | 20-50 min | 4 |
| **Ray (cluster)** | 30-60 min | 20-40 sec | 3-6 min | 100+ |
| **Modal** | **2 min** | **20-40 sec** | **3-6 min** | **1000+** |

Modal wins on:
- ✓ Fastest setup (2 minutes vs 30-60 for Ray cluster)
- ✓ No infrastructure (serverless)
- ✓ Auto-scales to 1000s of cores
- ✓ Pay per second (no idle costs)

## Cost Estimate

Modal charges **~$0.00001 per CPU-second**

| Job | Containers | Time | Cost |
|-----|-----------|------|------|
| 10k states (Fast) | 1000 | 30s | $0.30 |
| 10k states (Balanced) | 1000 | 90s | $0.90 |
| 100k states (Batched) | 1000 | 5min | $3.00 |
| 1M states (Batched) | 1000 | 50min | $30.00 |

**Free tier:** $30/month credit = ~100k states/day for free

## Example Usage

### Quick Test (1,000 states)
```bash
modal run experiments/modal_score_life.py::main --n-states=1000 --N=20 --num-samples=200
```
Time: ~10 seconds  
Cost: ~$0.03

### Production Run (10,000 states)
```bash
modal run experiments/modal_score_life.py::main --n-states=10000 --N=30 --num-samples=500
```
Time: ~1-2 minutes  
Cost: ~$0.50-1.00

### Large Scale (100,000 states)
```bash
modal run experiments/modal_score_life.py::main_batched \
  --n-states=100000 \
  --batch-size=100 \
  --N=30 \
  --num-samples=500
```
Time: ~5-10 minutes  
Cost: ~$3-5

## Configuration Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `--n-states` | Number of states to compute | 10000 | Any number |
| `--N` | Planning horizon | 30 | 20=fast, 30=balanced, 50=high-quality |
| `--num-samples` | Monte Carlo samples | 500 | 200=fast, 500=balanced, 1000=high-quality |
| `--n-l-points` | l-grid density | 20 | 10=fast, 20=balanced, 30=high-quality |
| `--batch-size` | States per container (batched only) | 100 | 50-200 |

## When to Use Modal

**✓ Use Modal if:**
- You have 10,000+ states
- You want results in minutes, not hours
- You don't have a compute cluster
- You want zero infrastructure setup
- One-off or occasional large computations

**✗ Don't use Modal if:**
- < 1,000 states (local is fine)
- You have free access to HPC cluster
- You run this continuously 24/7 (cluster cheaper)
- Budget is $0 (use free local compute)

## Advanced: Deploy as Webhook

You can also deploy as an API:

```python
@app.function()
@modal.web_endpoint()
def compute_api(states: list[float]):
    results = list(compute_state_value.map(states))
    return {"results": results}
```

Then call via HTTP:
```bash
curl https://your-app.modal.run/compute_api \
  -d '{"states": [0, 1000, 2000, ...]}'
```

## Monitoring

Modal provides real-time dashboard:
```bash
modal app logs score-life-parallel
```

View at: https://modal.com/apps

## Troubleshooting

**Error: "App not found"**
- Run `modal setup` to authenticate

**Error: "Image build failed"**
- Check that `src/` directory exists
- Ensure all dependencies are in `pip_install()`

**Slow performance**
- Use batched version for 100k+ states
- Increase `batch_size` to reduce overhead
- Use faster configs (reduce N, num_samples)

**Cost higher than expected**
- Check container CPU allocation (reduce to 1 CPU if single-threaded)
- Use batched version to reduce container overhead
- Monitor with `modal app logs`

## Comparison to Other Methods

| Feature | Local | Ray Cluster | Modal |
|---------|-------|-------------|-------|
| Setup | 0 min | 30-60 min | 2 min |
| Max cores | 4-64 | 100s | 1000s |
| Cost | $0 | $0-infrastructure | $0.00001/CPU-sec |
| Scaling | Manual | Manual | Automatic |
| Fault tolerance | No | Yes | Yes |
| Infrastructure | None | Cluster | None |

## Getting Help

- Modal Docs: https://modal.com/docs
- Modal Slack: https://modal.com/slack
- This project: See `experiments/modal_score_life.py`

## Next Steps

1. **Test locally first** (1000 states): Verify correctness
2. **Small Modal run** (1000 states): Test Modal setup  
3. **Scale up** (10k → 100k → 1M states): Production runs

The code in `modal_score_life.py` is production-ready!
