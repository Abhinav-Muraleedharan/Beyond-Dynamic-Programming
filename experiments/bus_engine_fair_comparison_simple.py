#!/usr/bin/env python
"""
Simple Fair Comparison: VI vs SL with γ=0.99

Quick test to see if policies match when discount factors are the same.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.bus_engine import BusEngineEnvironment

os.makedirs("results", exist_ok=True)

def run_value_iteration_simple(gamma=0.99):
    """Simple VI implementation."""
    print(f"Running Value Iteration with γ={gamma}...")

    env = BusEngineEnvironment(x=0, p=0.1, q=0.3)
    state_space = np.linspace(0, 50000, 100)
    V = np.zeros(len(state_space))

    # Value iteration
    for iteration in range(200):
        delta = 0
        new_V = V.copy()

        for i, state in enumerate(state_space):
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(50):
                    env.set_state(state)
                    result = env.step(action)
                    if len(result) == 4:
                        next_state, reward, _, _ = result
                    else:
                        next_state, reward, _, _, _ = result
                    next_idx = np.abs(state_space - next_state).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / 50)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V
        if delta < 1e-4:
            print(f"  Converged in {iteration + 1} iterations")
            break

    # Extract policy
    policy = np.zeros(len(state_space), dtype=int)
    for i, state in enumerate(state_space):
        best_q = -np.inf
        for action in [0, 1]:
            q_value = 0
            for _ in range(50):
                env.set_state(state)
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, _, _ = result
                else:
                    next_state, reward, _, _, _ = result
                next_idx = np.abs(state_space - next_state).argmin()
                q_value += reward + gamma * V[next_idx]
            q_value /= 50
            if q_value > best_q:
                best_q = q_value
                best_action = action
        policy[i] = best_action

    # Find threshold
    threshold = None
    for i, action in enumerate(policy):
        if action == 1:
            threshold = state_space[i]
            break

    print(f"  VI Threshold: {threshold:.0f} miles" if threshold else "  No threshold")
    return policy, state_space, threshold

def main():
    print("="*70)
    print("SIMPLE FAIR COMPARISON: VI with γ=0.99")
    print("="*70)

    vi_policy, vi_states, vi_threshold = run_value_iteration_simple(gamma=0.99)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step(vi_states, vi_policy, where='post', linewidth=2.5, color='blue')
    if vi_threshold:
        ax.axvline(x=vi_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold: {vi_threshold:.0f} miles')
    ax.set_xlabel('Engine Mileage (miles)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Keep Running', 'Replace'])
    ax.set_title('Value Iteration Policy (γ=0.99)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/bus_engine_vi_gamma099.png', dpi=150)
    print("\nSaved: results/bus_engine_vi_gamma099.png")

    print("\n"+"="*70)
    print("RESULT:")
    print(f"With γ=0.99, Value Iteration threshold = {vi_threshold:.0f} miles")
    print("="*70)

if __name__ == "__main__":
    main()
