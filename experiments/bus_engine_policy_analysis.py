#!/usr/bin/env python
"""
Bus Engine Problem - Policy Comparison and Runtime Analysis
Focus: Value Iteration vs optimized implementations
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


def run_value_iteration(env, gamma=0.99, n_states=50, n_samples=20, max_iters=200):
    """Run Value Iteration with detailed timing."""
    start_time = time.time()

    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

    # Value iteration
    for iteration in range(max_iters):
        new_V = V.copy()
        for i, state in enumerate(state_space):
            q_values = []
            for action in [0, 1]:
                q_value = 0
                for _ in range(n_samples):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_values.append(q_value / n_samples)
            new_V[i] = max(q_values)
        V = new_V

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    value_function = V.copy()

    for i, state in enumerate(state_space):
        best_q = -np.inf
        for action in [0, 1]:
            q_value = 0
            for _ in range(n_samples):
                env.set_state(state)
                next_state, reward, _, _, _ = env.step(action)
                next_idx = np.abs(state_space - next_state[0]).argmin()
                q_value += reward + gamma * V[next_idx]
            q_value /= n_samples
            if q_value > best_q:
                best_q = q_value
                best_action = action
        policy[i] = best_action

    vi_time = time.time() - start_time

    return policy, state_space, value_function, vi_time


def run_policy_iteration(env, gamma=0.99, n_states=50, n_samples=20, max_iters=50):
    """Run Policy Iteration."""
    start_time = time.time()

    state_space = np.linspace(0, 50000, n_states)
    policy = np.zeros(n_states, dtype=int)  # Initial policy: always keep running

    for iteration in range(max_iters):
        # Policy evaluation
        V = np.zeros(n_states)
        for _ in range(100):  # Evaluate for fixed iterations
            for i, state in enumerate(state_space):
                action = policy[i]
                q_value = 0
                for _ in range(n_samples):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                V[i] = q_value / n_samples

        # Policy improvement
        policy_stable = True
        for i, state in enumerate(state_space):
            old_action = policy[i]
            best_action = 0
            best_q = -np.inf

            for action in [0, 1]:
                q_value = 0
                for _ in range(n_samples):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]
                q_value /= n_samples

                if q_value > best_q:
                    best_q = q_value
                    best_action = action

            policy[i] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            print(f"Policy iteration converged in {iteration + 1} iterations")
            break

    pi_time = run_time = time.time() - start_time

    return policy, state_space, V, pi_time


def run_q_learning(
    env, gamma=0.99, n_states=50, n_episodes=500, alpha=0.1, epsilon=0.1
):
    """Run Q-Learning."""
    start_time = time.time()

    state_space = np.linspace(0, 50000, n_states)
    Q = np.zeros((n_states, 2))

    for episode in range(n_episodes):
        state, _ = env.reset()
        idx = np.abs(state_space - state[0]).argmin()
        done = False
        eps = epsilon * (1 - episode / n_episodes)

        while not done:
            if np.random.random() < eps:
                action = np.random.randint(2)
            else:
                action = np.argmax(Q[idx])

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_idx = np.abs(state_space - next_state[0]).argmin()
            done = terminated or truncated

            Q[idx, action] = Q[idx, action] + alpha * (
                reward + gamma * (1 - done) * np.max(Q[next_idx]) - Q[idx, action]
            )
            idx = next_idx

    policy = np.argmax(Q, axis=1)
    ql_time = time.time() - start_time

    return policy, state_space, Q, ql_time


def test_policy(env, policy, state_space, n_episodes=10):
    """Test a policy."""
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            idx = min(np.abs(state_space - state[0]).argmin(), len(policy) - 1)
            action = policy[idx]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def main():
    import os

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("POLICY COMPARISON AND RUNTIME ANALYSIS")
    print("Bus Engine Replacement Problem")
    print("=" * 70)

    env = BusEngineEnvironment(p=0.1, q=0.3)

    results = {}

    # 1. Value Iteration
    print("\n1. Running Value Iteration...")
    vi_policy, state_space, vi_value, vi_time = run_value_iteration(
        env, n_states=50, n_samples=20
    )
    vi_reward, vi_std = test_policy(env, vi_policy, state_space)
    print(f"   Time: {vi_time:.3f}s, Reward: {vi_reward:.2f} +/- {vi_std:.2f}")
    results["Value_Iteration"] = {"time": vi_time, "reward": vi_reward, "std": vi_std}

    # 2. Policy Iteration
    print("\n2. Running Policy Iteration...")
    env_pi = BusEngineEnvironment(p=0.1, q=0.3)
    pi_policy, state_space, pi_value, pi_time = run_policy_iteration(
        env_pi, n_states=50, n_samples=20
    )
    pi_reward, pi_std = test_policy(env_pi, pi_policy, state_space)
    print(f"   Time: {pi_time:.3f}s, Reward: {pi_reward:.2f} +/- {pi_std:.2f}")
    results["Policy_Iteration"] = {"time": pi_time, "reward": pi_reward, "std": pi_std}

    # 3. Q-Learning
    print("\n3. Running Q-Learning...")
    env_ql = BusEngineEnvironment(p=0.1, q=0.3)
    ql_policy, state_space, Q, ql_time = run_q_learning(
        env_ql, n_states=50, n_episodes=500
    )
    ql_reward, ql_std = test_policy(env_ql, ql_policy, state_space)
    print(f"   Time: {ql_time:.3f}s, Reward: {ql_reward:.2f} +/- {ql_std:.2f}")
    results["Q_Learning"] = {"time": ql_time, "reward": ql_reward, "std": ql_std}

    # Get thresholds
    vi_threshold = None
    for i, a in enumerate(vi_policy):
        if a == 1:
            vi_threshold = state_space[i]
            break

    pi_threshold = None
    for i, a in enumerate(pi_policy):
        if a == 1:
            pi_threshold = state_space[i]
            break

    ql_threshold = None
    for i, a in enumerate(ql_policy):
        if a == 1:
            ql_threshold = state_space[i]
            break

    results["thresholds"] = {
        "VI": float(vi_threshold) if vi_threshold else None,
        "PI": float(pi_threshold) if pi_threshold else None,
        "QL": float(ql_threshold) if ql_threshold else None,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Time (s)':<12} {'Reward':<15} {'Threshold':<15}")
    print("-" * 65)
    print(
        f"{'Value Iteration':<20} {vi_time:<12.3f} {vi_reward:<15.2f} {vi_threshold:<15.0f}"
    )
    print(
        f"{'Policy Iteration':<20} {pi_time:<12.3f} {pi_reward:<15.2f} {pi_threshold:<15.0f}"
    )
    print(
        f"{'Q-Learning':<20} {ql_time:<12.3f} {ql_reward:<15.2f} {ql_threshold:<15.0f}"
    )

    # Create plots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Policy Comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.step(
        state_space,
        vi_policy,
        where="post",
        label="Value Iteration",
        linewidth=2,
        color="blue",
    )
    ax1.step(
        state_space,
        pi_policy,
        where="post",
        label="Policy Iteration",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax1.step(
        state_space,
        ql_policy,
        where="post",
        label="Q-Learning",
        linewidth=2,
        color="green",
        linestyle=":",
    )
    ax1.set_xlabel("Engine Mileage", fontsize=12)
    ax1.set_ylabel("Action", fontsize=12)
    ax1.set_title("Policy Comparison (0=Keep, 1=Replace)", fontsize=14)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Keep Running", "Replace"])
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Value Functions
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(state_space, vi_value, "b-", linewidth=2, label="Value Iteration")
    ax2.plot(state_space, pi_value, "r--", linewidth=2, label="Policy Iteration")
    ax2.set_xlabel("Engine Mileage", fontsize=12)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Value Functions", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Runtime Comparison
    ax3 = plt.subplot(2, 2, 3)
    methods = ["Value\nIteration", "Policy\nIteration", "Q-Learning"]
    times = [vi_time, pi_time, ql_time]
    colors = ["blue", "red", "green"]
    bars = ax3.bar(methods, times, color=colors, alpha=0.7, edgecolor="black")
    ax3.set_ylabel("Time (seconds)", fontsize=12)
    ax3.set_title("Computational Runtime", fontsize=14)
    for bar, t in zip(bars, times):
        ax3.annotate(
            f"{t:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Performance Comparison
    ax4 = plt.subplot(2, 2, 4)
    rewards = [vi_reward, pi_reward, ql_reward]
    stds = [vi_std, pi_std, ql_std]
    bars = ax4.bar(
        methods,
        rewards,
        yerr=stds,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        capsize=5,
    )
    ax4.set_ylabel("Average Reward", fontsize=12)
    ax4.set_title("Policy Performance (with std dev)", fontsize=14)
    for bar, r in zip(bars, rewards):
        ax4.annotate(
            f"{r:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "results/bus_engine_policy_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("\nSaved: results/bus_engine_policy_comparison.png")

    # Save results
    with open("results/bus_engine_policy_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved: results/bus_engine_policy_analysis.json")

    # Print thresholds
    print("\n" + "=" * 70)
    print("REPLACEMENT THRESHOLDS")
    print("=" * 70)
    print(
        f"Value Iteration:  {vi_threshold:.0f} miles"
        if vi_threshold
        else "Value Iteration:  N/A"
    )
    print(
        f"Policy Iteration: {pi_threshold:.0f} miles"
        if pi_threshold
        else "Policy Iteration: N/A"
    )
    print(
        f"Q-Learning:       {ql_threshold:.0f} miles"
        if ql_threshold
        else "Q-Learning:       N/A"
    )


if __name__ == "__main__":
    main()
