# Bus Engine Experiments - Key Findings Summary

## Executive Summary

Completed comprehensive experiments on the Bus Engine Replacement Problem comparing multiple reinforcement learning approaches. Key finding: **Score-Life Programming and Value Iteration produce different replacement threshold policies**.

## Completed Experiments

### 1. ✅ Basic Value Iteration (`bus_engine_experiment.py`)
- **Status**: Complete
- **Threshold**: ~2,041 miles
- **Average Reward**: -35,210

### 2. ✅ Full Method Comparison (`bus_engine_full_experiment.py`)
- **Status**: Complete
- **Methods Tested**: Value Iteration, Score-Life Exact, Score-Life Approximate, Random
- **Configurations**: 3 different (p,q) combinations
- **Key Result**: Score-Life Exact achieves best rewards across all configurations

**Results Table:**
| Config | VI Reward | SL Exact | SL Approx | Random |
|--------|-----------|----------|-----------|--------|
| Standard (0.1, 0.3) | -35,210 | **-23,420** | -31,006 | -48,004 |
| High Small (0.2, 0.2) | -35,625 | **-22,643** | -30,221 | -46,261 |
| High Large (0.05, 0.4) | -36,403 | **-21,292** | -31,720 | -44,620 |

### 3. ✅ Policy Runtime Analysis (`bus_engine_policy_analysis.py`)
- **Status**: Complete
- **Methods**: Value Iteration, Policy Iteration, Q-Learning
- **Finding**: All three converge to similar ~2,000 mile threshold
- **Runtime**: VI fastest at ~2-3 seconds for 50 states

### 4. ✅ Score-Life Visualization (`bus_engine_score_life_visualization.py`)
- **Status**: Complete
- **States Analyzed**: 16 states from 0 to 10,000 miles
- **Output**: Score-Life function plots showing fractal structure
- **Key State**: 2,041 miles (optimal state) highlighted

### 5. ✅ Detailed Policy Comparison (`bus_engine_policy_comparison_detailed.py`)
- **Status**: Complete
- **Key Finding**: **Policies are NOT identical**
  - VI threshold: 2,525 miles
  - SL threshold: 5,556 miles
  - Agreement: 94% overall
  - Disagreement: 2,525-5,051 mile range

### 6. 🔄 Scalability Experiment (`bus_engine_scalability_experiment.py`)
- **Status**: In Progress (currently at 200/1000 states)
- **Purpose**: Analyze computational complexity
- **Expected**: Timing data for state spaces 100-1000

## Critical Insight: Policy Differences

### Why Do Policies Differ?

**Value Iteration Policy:**
- Threshold: ~2,525 miles
- Based on: Exact Bellman optimality equations
- Approach: Converged value function via dynamic programming

**Score-Life Programming Policy:**
- Threshold: ~5,556 miles (in simplified extraction)
- Based on: Faber-Schauder fractal function representation
- Approach: Life-parametrized action sequences

### Performance vs Policy Trade-off

Despite different policies, **Score-Life Exact achieves better rewards** (-23,420 vs -35,210). This suggests:

1. **Score-Life may be finding a different optimal policy** that Value Iteration misses
2. **Or**, the policy extraction from Score-Life needs refinement
3. **Or**, the hyperparameters (N, j_max, gamma) affect the policy structure

## Interpretation

### Possible Explanations:

1. **Different Approximations**
   - VI uses discrete state space (50-300 states)
   - SL uses continuous fractal representation
   - Discretization may affect threshold location

2. **Discount Factor Sensitivity**
   - VI typically uses γ = 0.99
   - SL visualization used γ = 0.60
   - Different γ values lead to different thresholds

3. **Stochasticity in Transition**
   - Monte Carlo sampling (100 samples) introduces variance
   - Both methods use sampling, but differently
   - May converge to local optima

4. **Policy Extraction Method**
   - VI policy extracted directly from Q-values
   - SL policy extraction is more complex (life → action mapping)
   - The "simplified" SL policy used heuristics, not true SL method

## Recommendations for Further Analysis

### Priority 1: Proper Score-Life Policy Extraction
- Implement exact life-to-action mapping from Score-Life theory
- Use the same γ across both methods
- Compare policies at identical discretization levels

### Priority 2: Sensitivity Analysis
- Test both methods with γ ∈ {0.6, 0.7, 0.8, 0.9, 0.95, 0.99}
- Plot threshold vs discount factor
- Identify where policies align/diverge

### Priority 3: Convergence Verification
- Increase Monte Carlo samples (100 → 1000)
- Increase state space resolution (50 → 500)
- Check if policies converge to same threshold

### Priority 4: Theoretical Analysis
- Review Score-Life Programming theory for policy extraction
- Verify if fractal representation preserves policy structure
- Check if Faber-Schauder coefficients uniquely determine policy

## Files Generated

### Results Data
- `bus_engine_results.json` - Basic VI results
- `bus_engine_experiments.json` - Method comparison
- `bus_engine_policy_analysis.json` - Runtime analysis
- `bus_engine_policy_comparison_detailed.json` - Policy comparison
- `bus_engine_scalability.json` - Scalability data (pending)

### Visualizations
- `bus_engine_results.png` - Value function and policy
- `bus_engine_comparison.png` - Method comparison bars
- `bus_engine_policy_comparison.png` - Policy and runtime plots
- `bus_engine_policy_comparison_detailed.png` - Direct policy comparison
- `bus_engine_score_life_16states.png` - Score-Life functions grid
- `bus_engine_score_life_overlay.png` - Score-Life functions overlay
- `bus_engine_scalability.png` - Scalability plot (pending)

### Reports
- `bus_engine_comprehensive_report.md` - Full experimental report
- `bus_engine_findings_summary.md` - This summary

## Conclusions

1. ✅ **All experiments successfully completed** (except scalability - in progress)
2. ⚠️ **Important finding**: Score-Life and VI produce **different policies**
3. 📊 **Score-Life Exact achieves best performance** in terms of rewards
4. 🔍 **Further investigation needed** to reconcile policy differences
5. 📈 **Scalability analysis ongoing** to complete computational complexity study

## Next Steps

1. Wait for scalability experiment completion
2. Implement proper Score-Life policy extraction (not heuristic)
3. Run sensitivity analysis with matched hyperparameters
4. Theoretical analysis of why policies differ
5. Update bus_engine.py with completed timing data

---

**Status**: 5/6 experiments complete (83% done)  
**Last Updated**: July 5, 2026  
**Scalability ETA**: ~30-60 minutes (depends on convergence)
