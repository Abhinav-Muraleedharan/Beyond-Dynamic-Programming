#!/usr/bin/env python
# experiments/run_all_experiments.py

import os
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

import gym
from src.score_life_programming.exact_methods import run_exact_method
from src.score_life_programming.approximate_methods import run_approximate_method
from src.environments.mountain_car import MountainCarEnv


def run_single_episode(env, policy_fn, max_steps=500):
    """Run a single episode with the given policy function."""
    result = env.reset()
    if isinstance(result, tuple):
        observation = result[0]
    else:
        observation = result
    total_reward = 0

    for _ in range(max_steps):
        action = policy_fn(observation)
        result = env.step(action)
        if len(result) == 5:
            observation, reward, done, truncated, _ = result
            done = done or truncated
        else:
            observation, reward, done, _ = result
        total_reward += reward

        if done:
            break

    return total_reward


def run_random_agent(env, n_episodes=10):
    """Run random agent as baseline."""
    rewards = []
    for _ in range(n_episodes):
        result = env.reset()
        if isinstance(result, tuple):
            observation = result[0]
        else:
            observation = result
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            result = env.step(action)
            if len(result) == 5:
                observation, reward, done, truncated, _ = result
                done = done or truncated
            else:
                observation, reward, done, _ = result
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)


def run_q_learning(env, n_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Run Q-learning on the environment."""
    n_states = 20
    n_actions = env.action_space.n

    try:
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
    except:
        obs_low = np.array([-1.2, -0.07])
        obs_high = np.array([0.6, 0.07])

    def discretize_state(state):
        pos_bins = np.linspace(obs_low[0], obs_high[0], n_states)
        vel_bins = np.linspace(obs_low[1], obs_high[1], n_states)

        pos = min(np.digitize(state[0], pos_bins), n_states - 1)
        vel = min(np.digitize(state[1], vel_bins), n_states - 1)

        return pos * n_states + vel

    Q = np.random.uniform(-1, 1, (n_states**2, n_actions))

    for episode in range(n_episodes):
        result = env.reset()
        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result
        s = discretize_state(state)
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[s])

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, done, truncated, _ = result
                done = done or truncated
            else:
                next_state, reward, done, _ = result

            next_s = discretize_state(next_state)

            if done:
                Q[s, action] = Q[s, action] + alpha * (reward - Q[s, action])
            else:
                Q[s, action] = Q[s, action] + alpha * (
                    reward + gamma * np.max(Q[next_s]) - Q[s, action]
                )

            s = next_s

    def policy_fn(state):
        s = discretize_state(state)
        return np.argmax(Q[s])

    return policy_fn


def run_sb3_benchmark(env, algo="PPO", total_timesteps=5000, eval_episodes=3):
    """Run Stable Baselines3 benchmark."""
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.evaluation import evaluate_policy
        from shimmy import GymV21CompatibilityV0

        env = GymV21CompatibilityV0(env)
    except Exception as e:
        print(f"SB3 setup error: {e}")
        return 0.0, 0.0

    env = Monitor(env)

    if algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, n_steps=128, n_epochs=4)
    elif algo == "A2C":
        model = A2C("MlpPolicy", env, verbose=0)
    elif algo == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    return mean_reward, std_reward


def run_all_experiments():
    """Run all experiments and collect results."""
    results = {}

    environments = [
        ("CartPole-v1", lambda: gym.make("CartPole-v1")),
        ("MountainCar-v0", lambda: MountainCarEnv()),
    ]

    for env_name, env_fn in environments:
        print(f"\n{'=' * 50}")
        print(f"Running experiments for {env_name}")
        print("=" * 50)

        env = env_fn()
        env_results = {}

        print(f"  Running Random Agent Baseline...")
        try:
            random_reward = run_random_agent(env, n_episodes=5)
            env_results["Random"] = float(random_reward)
            print(f"    Result: {random_reward:.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            env_results["Random"] = 0

        print(f"  Running Score-Life Exact Method...")
        try:
            exact_reward = run_exact_method(
                env, gamma=0.99, N=50, j_max=10, num_samples=3
            )
            env_results["ScoreLife_Exact"] = exact_reward
            print(f"    Result: {exact_reward}")
        except Exception as e:
            print(f"    Error: {e}")
            env_results["ScoreLife_Exact"] = 0

        print(f"  Running Score-Life Approximate Method...")
        try:
            approx_reward = run_approximate_method(
                env, gamma=0.99, N=30, n=5, num_samples=3
            )
            env_results["ScoreLife_Approx"] = approx_reward
            print(f"    Result: {approx_reward}")
        except Exception as e:
            print(f"    Error: {e}")
            env_results["ScoreLife_Approx"] = 0

        print(f"  Running Q-Learning...")
        try:
            q_learning_policy = run_q_learning(env, n_episodes=50)
            q_rewards = [run_single_episode(env, q_learning_policy) for _ in range(3)]
            env_results["Q_Learning"] = float(np.mean(q_rewards))
            print(f"    Result: {np.mean(q_rewards):.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            env_results["Q_Learning"] = 0

        for algo in ["PPO", "A2C"]:
            print(f"  Running {algo}...")
            try:
                mean_r, std_r = run_sb3_benchmark(
                    env, algo=algo, total_timesteps=3000, eval_episodes=3
                )
                env_results[f"{algo}_mean"] = float(mean_r)
                env_results[f"{algo}_std"] = float(std_r)
                print(f"    Result: {mean_r:.2f} +/- {std_r:.2f}")
            except Exception as e:
                print(f"    Error: {e}")
                env_results[f"{algo}_mean"] = 0
                env_results[f"{algo}_std"] = 0

        results[env_name] = env_results
        env.close()

        with open("results/experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n" + "=" * 50)
    print("All experiments completed!")
    print("=" * 50)
    print(json.dumps(results, indent=2))
