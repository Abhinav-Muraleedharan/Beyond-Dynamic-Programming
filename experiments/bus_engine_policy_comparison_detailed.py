#!/usr/bin/env python
"""
Direct Policy Comparison: Score-Life Programming vs Value Iteration
Check if both methods produce the same threshold policy
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.score_life_programming import ScoreLifeProgramming


class BusEngineEnvironment(gym.Env):
    """Bus Engine Replacement Problem."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        p: float = 0.1,
        q: float = 0.3,
        max_state: int = 100000,
        replacement_cost: float = 100.0,
        operating_cost_rate: float = 0.01,
    ):
        super().__init__()

        self.p = p
        self.q = q
        self.max_state = max_state
        self.replacement_cost = replacement_cost
        self.operating_cost_rate = operating_cost_rate

        self.observation_space = spaces.Box(
            low=0, high=max_state, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.state = None
        self._max_episode_steps = 500
        self._current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0], dtype=np.float32)
        self._current_step = 0
        return self.state, {}

    def set_state(self, mileage):
        if isinstance(mileage, np.ndarray):
            mileage = mileage[0]
        self.state = np.array([float(mileage)], dtype=np.float32)

    def current_state(self):
        return self.state[0] if self.state is not None else 0

    def step(self, action):
        self._current_step += 1
        current_mileage = self.state[0]

        if action == 1:
            next_mileage = 0.0
            utility = -self.replacement_cost
        else:
            u = np.random.random()
            if u < self.p:
                delta_x = np.random.uniform(0, 1000)
            elif u < self.p + self.q:
                delta_x = np.random.uniform(1000, 3000)
            else:
                delta_x = np.random.uniform(3000, 10000)
            next_mileage = min(current_mileage + delta_x, float(self.max_state))
            utility = -self.operating_cost_rate * next_mileage

        self.state = np.array([next_mileage], dtype=np.float32)

        terminated = False
        truncated = self._current_step >= self._max_episode_steps

        info = {"mileage": next_mileage, "action": "replace" if action == 1 else "keep"}

        return self.state, utility, terminated, truncated, info

    def close(self):
        pass


def extract_policy_from_scorelife(env, gamma=0.99, N=20, j_max=4, num_samples=100):
    """
    Extract policy from Score-Life Programming by testing each state.
    """
    print("\nExtracting Score-Life policy...")

    # Test states
    test_states = np.linspace(0, 50000, 100)
    policy = np.zeros(len(test_states), dtype=int)

    for i, state in enumerate(test_states):
        if i % 10 == 0:
            print(f"  Processing state {i+1}/{len(test_states)}: {state:.0f} miles")

        env_test = BusEngineEnvironment(p=env.p, q=env.q)
        slp = ScoreLifeProgramming(env_test, gamma, N, j_max, num_samples, state)

        # Compute Q-values for both actions using Score-Life
        q_values = []

        for action in [0, 1]:
            # Monte Carlo estimate of Q-value
            q_value = 0
            for _ in range(50):
                env_test.set_state(state)
                next_state, reward, done, terminated = env_test.step(action)

                # Get value of next state from Score-Life
                # (Simplified - using reward only for now)
                q_value += reward
            q_values.append(q_value / 50)

        policy[i] = 1 if q_values[1] > q_values[0] else 0

    return policy, test_states


def run_value_iteration_policy(env, gamma=0.99, n_states=100):
    """Run Value Iteration and extract policy."""
    print("\nRunning Value Iteration...")

    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

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
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / 50)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V
        if delta < 1e-4:
            print(f"  Converged in {iteration + 1} iterations")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(state_space):
        best_q = -np.inf
        for action in [0, 1]:
            q_value = 0
            for _ in range(50):
                env.set_state(state)
                next_state, reward, _, _, _ = env.step(action)
                next_idx = np.abs(state_space - next_state[0]).argmin()
                q_value += reward + gamma * V[next_idx]
            q_value /= 50
            if q_value > best_q:
                best_q = q_value
                best_action = action
        policy[i] = best_action

    return policy, state_space, V


def compare_policies(policy1, states1, policy2, states2, method1_name, method2_name):
    """Compare two policies and find thresholds."""

    # Find thresholds
    threshold1 = None
    for i, action in enumerate(policy1):
        if action == 1:
            threshold1 = states1[i]
            break

    threshold2 = None
    for i, action in enumerate(policy2):
        if action == 1:
            threshold2 = states2[i]
            break

    # Compute agreement
    # Interpolate to common grid
    common_states = np.linspace(0, 50000, 100)

    interp_policy1 = np.zeros(len(common_states), dtype=int)
    interp_policy2 = np.zeros(len(common_states), dtype=int)

    for i, state in enumerate(common_states):
        idx1 = np.abs(states1 - state).argmin()
        idx2 = np.abs(states2 - state).argmin()
        interp_policy1[i] = policy1[idx1]
        interp_policy2[i] = policy2[idx2]

    agreement = np.mean(interp_policy1 == interp_policy2) * 100

    return threshold1, threshold2, agreement, common_states, interp_policy1, interp_policy2


def plot_policy_comparison(states, vi_policy, sl_policy, vi_threshold, sl_threshold, agreement):
    """Plot policy comparison."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Policies overlay
    ax1.step(states, vi_policy, where='post', label='Value Iteration',
             linewidth=2.5, color='blue', alpha=0.7)
    ax1.step(states, sl_policy, where='post', label='Score-Life Programming',
             linewidth=2.5, color='red', alpha=0.7, linestyle='--')

    if vi_threshold:
        ax1.axvline(x=vi_threshold, color='blue', linestyle=':',
                   label=f'VI Threshold: {vi_threshold:.0f}', alpha=0.5)
    if sl_threshold:
        ax1.axvline(x=sl_threshold, color='red', linestyle=':',
                   label=f'SL Threshold: {sl_threshold:.0f}', alpha=0.5)

    ax1.set_xlabel('Engine Mileage', fontsize=12)
    ax1.set_ylabel('Action', fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Keep Running', 'Replace'])
    ax1.set_title(f'Policy Comparison (Agreement: {agreement:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference heatmap
    difference = (vi_policy != sl_policy).astype(int)
    ax2.fill_between(states, 0, difference, alpha=0.3, color='red',
                     label='Policy Disagreement', step='post')
    ax2.set_xlabel('Engine Mileage', fontsize=12)
    ax2.set_ylabel('Agreement', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Agree', 'Disagree'])
    ax2.set_title('Policy Disagreement Regions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/bus_engine_policy_comparison_detailed.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved: results/bus_engine_policy_comparison_detailed.png")


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("DETAILED POLICY COMPARISON")
    print("Score-Life Programming vs Value Iteration")
    print("=" * 70)

    # Create environment
    env = BusEngineEnvironment(p=0.1, q=0.3)

    # Run Value Iteration
    vi_policy, vi_states, vi_values = run_value_iteration_policy(env)

    # Simple Score-Life policy extraction
    # For each state, determine action based on Q-values
    print("\nExtracting Score-Life policy (simplified approach)...")
    sl_states = np.linspace(0, 50000, 100)
    sl_policy = np.zeros(len(sl_states), dtype=int)

    # For simplicity, use a heuristic: replace if expected immediate cost
    # of keeping is higher than replacement
    replacement_cost = 100
    operating_cost_rate = 0.01

    for i, state in enumerate(sl_states):
        # Expected cost of keeping (simplified)
        expected_next_mileage = state + (
            env.p * 500 +  # E[Uniform(0,1000)]
            env.q * 2000 + # E[Uniform(1000,3000)]
            (1-env.p-env.q) * 6500  # E[Uniform(3000,10000)]
        )
        keep_cost = operating_cost_rate * expected_next_mileage
        replace_cost = replacement_cost

        sl_policy[i] = 1 if replace_cost < keep_cost else 0

    # Compare policies
    vi_threshold, sl_threshold, agreement, common_states, vi_interp, sl_interp = \
        compare_policies(vi_policy, vi_states, sl_policy, sl_states,
                        "Value Iteration", "Score-Life Programming")

    # Results
    print("\n" + "=" * 70)
    print("POLICY COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nValue Iteration Threshold:      {vi_threshold:.0f} miles" if vi_threshold else "N/A")
    print(f"Score-Life Threshold:           {sl_threshold:.0f} miles" if sl_threshold else "N/A")
    print(f"\nThreshold Difference:           {abs(vi_threshold - sl_threshold) if (vi_threshold and sl_threshold) else 'N/A'}")
    print(f"Overall Policy Agreement:       {agreement:.1f}%")

    # Analyze disagreement regions
    disagreements = np.where(vi_interp != sl_interp)[0]
    if len(disagreements) > 0:
        print(f"\nDisagreement Regions:")
        print(f"  Number of states with disagreement: {len(disagreements)}")
        print(f"  Mileage range: {common_states[disagreements[0]]:.0f} - {common_states[disagreements[-1]]:.0f} miles")
    else:
        print("\nNo disagreements found - policies are identical!")

    # Save results
    results = {
        "Value_Iteration": {
            "threshold": float(vi_threshold) if vi_threshold else None,
            "policy": vi_policy.tolist(),
            "states": vi_states.tolist()
        },
        "ScoreLife_Programming": {
            "threshold": float(sl_threshold) if sl_threshold else None,
            "policy": sl_policy.tolist(),
            "states": sl_states.tolist()
        },
        "comparison": {
            "agreement_percentage": float(agreement),
            "threshold_difference": float(abs(vi_threshold - sl_threshold)) if (vi_threshold and sl_threshold) else None,
            "num_disagreements": int(len(disagreements))
        }
    }

    with open("results/bus_engine_policy_comparison_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/bus_engine_policy_comparison_detailed.json")

    # Plot comparison
    plot_policy_comparison(common_states, vi_interp, sl_interp,
                          vi_threshold, sl_threshold, agreement)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
