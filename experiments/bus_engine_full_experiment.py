#!/usr/bin/env python
"""
Bus Engine Problem - Complete Experiments with Score-Life Programming
"""

import time
import numpy as np
import gym
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore")

# Import Score-Life Programming methods
from src.score_life_programming.exact_methods import run_exact_method
from src.score_life_programming.approximate_methods import run_approximate_method


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

    def render(self, mode="human"):
        print(f"Mileage: {self.state[0]:.2f}")

    def close(self):
        pass


def run_value_iteration(env, gamma=0.99, n_states=50):
    """Run Value Iteration."""
    print("\n--- Value Iteration ---")

    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

    for iteration in range(200):
        delta = 0
        new_V = V.copy()

        for i, state in enumerate(state_space):
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(20):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / 20)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V
        if delta < 1e-4:
            print(f"Converged in {iteration + 1} iterations")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(state_space):
        best_q = -np.inf
        best_action = 0
        for action in [0, 1]:
            q_value = 0
            for _ in range(20):
                env.set_state(state)
                next_state, reward, _, _, _ = env.step(action)
                next_idx = np.abs(state_space - next_state[0]).argmin()
                q_value += reward + gamma * V[next_idx]
            q_value /= 20
            if q_value > best_q:
                best_q = q_value
                best_action = action
        policy[i] = best_action

    replace_idx = np.where(policy == 1)[0]
    threshold = state_space[replace_idx[0]] if len(replace_idx) > 0 else None

    return V, policy, state_space, threshold


def test_policy(env, policy, state_space, n_episodes=5):
    """Test a policy and return average reward."""
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            idx = np.abs(state_space - state[0]).argmin()
            action = policy[idx]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards)


def run_all_experiments():
    """Run all experiments."""
    print("=" * 70)
    print("BUS ENGINE REPLACEMENT PROBLEM - COMPREHENSIVE EXPERIMENTS")
    print("=" * 70)

    results = {}

    # Test different configurations
    configs = [
        {"p": 0.1, "q": 0.3, "name": "Standard (p=0.1, q=0.3)"},
        {"p": 0.2, "q": 0.2, "name": "High small damage (p=0.2, q=0.2)"},
        {"p": 0.05, "q": 0.4, "name": "High large damage (p=0.05, q=0.4)"},
    ]

    all_results = []

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config['name']}")
        print("=" * 70)

        env = BusEngineEnvironment(p=config["p"], q=config["q"])

        # Value Iteration
        V, policy, state_space, threshold = run_value_iteration(env)
        vi_reward = test_policy(env, policy, state_space)

        print(f"VI - Replacement threshold: {threshold:.0f} miles")
        print(f"VI - Average reward: {vi_reward:.2f}")

        # Score-Life Exact
        print("\n--- Score-Life Exact Method ---")
        env_sl = BusEngineEnvironment(p=config["p"], q=config["q"])
        try:
            sl_exact_reward = run_exact_method(
                env_sl, gamma=0.99, N=20, j_max=3, num_samples=3
            )
            print(f"SL Exact - Reward: {sl_exact_reward:.2f}")
        except Exception as e:
            print(f"SL Exact - Error: {e}")
            sl_exact_reward = 0

        # Score-Life Approximate
        print("\n--- Score-Life Approximate Method ---")
        env_sl_approx = BusEngineEnvironment(p=config["p"], q=config["q"])
        try:
            sl_approx_reward = run_approximate_method(
                env_sl_approx, gamma=0.99, N=15, n=5, num_samples=3
            )
            print(f"SL Approx - Reward: {sl_approx_reward:.2f}")
        except Exception as e:
            print(f"SL Approx - Error: {e}")
            sl_approx_reward = 0

        # Random baseline
        print("\n--- Random Baseline ---")
        random_rewards = []
        for _ in range(3):
            state, _ = env.reset()
            total = 0
            done = False
            while not done:
                action = env.action_space.sample()
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            random_rewards.append(total)
        random_reward = np.mean(random_rewards)
        print(f"Random - Average reward: {random_reward:.2f}")

        config_results = {
            "config": config["name"],
            "p": config["p"],
            "q": config["q"],
            "Value_Iteration": {
                "reward": float(vi_reward),
                "threshold": float(threshold) if threshold else None,
            },
            "ScoreLife_Exact": float(sl_exact_reward),
            "ScoreLife_Approx": float(sl_approx_reward),
            "Random": float(random_reward),
        }

        all_results.append(config_results)

        # Print comparison
        print("\n--- Comparison ---")
        print(f"Random:        {random_reward:>10.2f}")
        print(f"SL Exact:      {sl_exact_reward:>10.2f}")
        print(f"SL Approx:     {sl_approx_reward:>10.2f}")
        print(f"Value Iter:    {vi_reward:>10.2f}  (threshold: {threshold:.0f})")

    # Save results
    with open("results/bus_engine_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Create comparison plot
    create_comparison_plot(all_results)

    return all_results


def create_comparison_plot(all_results):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = [r["config"] for r in all_results]
    x = np.arange(len(configs))
    width = 0.2

    # Rewards comparison
    ax = axes[0]
    methods = ["Random", "ScoreLife_Exact", "ScoreLife_Approx", "Value_Iteration"]
    colors = ["gray", "blue", "green", "red"]

    for i, method in enumerate(methods):
        if method == "Value_Iteration":
            values = [r[method]["reward"] for r in all_results]
        else:
            values = [r[method] for r in all_results]
        ax.bar(x + i * width, values, width, label=method, color=colors[i], alpha=0.8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Average Reward")
    ax.set_title("Method Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.split("(")[0].strip() for c in configs], rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Thresholds
    ax = axes[1]
    thresholds = [r["Value_Iteration"]["threshold"] for r in all_results]
    ax.bar(
        [c.split("(")[0].strip() for c in configs], thresholds, color="red", alpha=0.8
    )
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Replacement Threshold (miles)")
    ax.set_title("Optimal Replacement Threshold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/bus_engine_comparison.png", dpi=150)
    print("\nSaved: results/bus_engine_comparison.png")


def main():
    import os

    os.makedirs("results", exist_ok=True)

    results = run_all_experiments()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nResults saved to: results/bus_engine_experiments.json")
    print(f"Plots saved to: results/bus_engine_comparison.png")


if __name__ == "__main__":
    main()
