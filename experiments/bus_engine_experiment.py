#!/usr/bin/env python
"""
Bus Engine Replacement Problem - Verification and Experiments
"""

import time
import numpy as np
import gym
from gym import spaces
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


class BusEngineEnvironment(gym.Env):
    """
    Bus Engine Replacement Problem.

    This is a classic sequential decision problem:
    - State: Engine mileage (x)
    - Action: 0 = Keep running, 1 = Replace engine
    """

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
        """Set the current mileage state."""
        if isinstance(mileage, np.ndarray):
            mileage = mileage[0]
        self.state = np.array([float(mileage)], dtype=np.float32)

    def step(self, action):
        self._current_step += 1

        current_mileage = self.state[0]

        if action == 1:  # Replace engine
            next_mileage = 0.0
            utility = -self.replacement_cost
        else:  # Keep running
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


def verify_environment():
    """Verify the environment works."""
    print("=" * 60)
    print("Verifying Bus Engine Environment")
    print("=" * 60)

    env = BusEngineEnvironment(p=0.1, q=0.3)

    state, _ = env.reset()
    print(f"Initial state: {state}")

    state, reward, done, truncated, info = env.step(0)
    print(f"After keep: mileage={state[0]:.2f}, reward={reward:.2f}")

    state, reward, done, truncated, info = env.step(1)
    print(f"After replace: mileage={state[0]:.2f}, reward={reward:.2f}")

    print("\n✓ Environment verification passed!")
    return env


def run_value_iteration(env, gamma=0.99, n_states=50, max_iterations=200):
    """Value iteration with proper state handling."""
    print("\n" + "=" * 60)
    print("Running Value Iteration")
    print("=" * 60)

    state_space = np.linspace(0, 50000, n_states)
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0
        new_V = V.copy()

        for i, state in enumerate(state_space):
            q_values = []

            for action in [0, 1]:
                q_value = 0
                n_samples = 20

                for _ in range(n_samples):
                    env.set_state(state)
                    next_state, reward, _, _, _ = env.step(action)
                    next_idx = np.abs(state_space - next_state[0]).argmin()
                    q_value += reward + gamma * V[next_idx]

                q_values.append(q_value / n_samples)

            new_V[i] = max(q_values)
            delta = max(delta, abs(V[i] - new_V[i]))

        V = new_V

        if delta < 1e-4:
            print(f"Converged after {iteration + 1} iterations")
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for i, state in enumerate(state_space):
        best_action = 0
        best_q = -np.inf

        for action in [0, 1]:
            q_value = 0
            n_samples = 20

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

    # Find replacement threshold
    replace_idx = np.where(policy == 1)[0]
    replace_threshold = state_space[replace_idx[0]] if len(replace_idx) > 0 else None

    return V, policy, state_space, replace_threshold


def test_policy(env, policy, state_space, n_episodes=5):
    """Test the policy."""
    print("\nTesting policy:")
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        replacements = 0

        done = False
        while not done:
            idx = np.abs(state_space - state[0]).argmin()
            action = policy[idx]

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if action == 1:
                replacements += 1

            done = terminated or truncated

        rewards.append(total_reward)
        print(
            f"  Episode {ep + 1}: Reward={total_reward:.2f}, Steps={steps}, Replacements={replacements}"
        )

    print(f"  Average: {np.mean(rewards):.2f}")
    return np.mean(rewards)


def main():
    import os

    os.makedirs("results", exist_ok=True)

    # Verify
    env = verify_environment()

    # Run VI
    V, policy, state_space, threshold = run_value_iteration(env)

    print(
        f"\nReplacement threshold: {threshold:.0f}"
        if threshold
        else "No threshold found"
    )

    # Test
    avg_reward = test_policy(env, policy, state_space)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(state_space, V, "b-", linewidth=2)
    ax1.set_xlabel("Mileage")
    ax1.set_ylabel("Value")
    ax1.set_title("Value Function")
    ax1.grid(True, alpha=0.3)

    ax2.step(state_space, policy, where="post", linewidth=2)
    ax2.set_xlabel("Mileage")
    ax2.set_ylabel("Action")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Keep", "Replace"])
    ax2.set_title(
        f"Policy (Replace at {threshold:.0f} miles)" if threshold else "Policy"
    )
    ax2.grid(True, alpha=0.3)
    if threshold:
        ax2.axvline(x=threshold, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig("results/bus_engine_results.png", dpi=150)
    print("\nSaved: results/bus_engine_results.png")

    # Save
    results = {
        "replacement_threshold": float(threshold) if threshold else None,
        "average_reward": float(avg_reward),
    }
    with open("results/bus_engine_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary: Replace at {threshold:.0f} miles, Avg reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
