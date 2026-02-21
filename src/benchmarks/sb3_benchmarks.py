# src/benchmarks/sb3_benchmarks.py

import warnings

warnings.filterwarnings("ignore")

try:
    from stable_baselines3 import A2C, PPO, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def train_and_evaluate(env, algorithm, total_timesteps=100000, eval_episodes=10):
    if not SB3_AVAILABLE:
        return 0.0, 0.0

    try:
        import gymnasium as gym

        has_gymnasium = True
    except ImportError:
        has_gymnasium = False

    if has_gymnasium:
        from shimmy import GymV21CompatibilityV0

        env = GymV21CompatibilityV0(env)

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    if algorithm == "A2C":
        model = A2C("MlpPolicy", env, verbose=0)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=0)
    elif algorithm == "DQN":
        model = DQN("MlpPolicy", env, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    return mean_reward, std_reward


def run_benchmarks(
    env, algorithms=["A2C", "PPO", "DQN"], total_timesteps=100000, eval_episodes=10
):
    results = {}
    if not SB3_AVAILABLE:
        for algo in algorithms:
            results[algo] = {"mean_reward": 0, "std_reward": 0}
        return results

    for algo in algorithms:
        mean_reward, std_reward = train_and_evaluate(
            env, algo, total_timesteps, eval_episodes
        )
        results[algo] = {"mean_reward": mean_reward, "std_reward": std_reward}
    return results
