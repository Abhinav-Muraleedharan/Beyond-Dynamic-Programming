# src/benchmarks/sb3_benchmarks.py

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import warnings

def train_and_evaluate(env, algorithm, total_timesteps=100000, eval_episodes=10, normalize=False, return_model=False):
    """
    Train and evaluate a reinforcement learning agent using Stable Baselines3.

    Args:
        env: The environment to train on
        algorithm: Algorithm name ('A2C', 'PPO', or 'DQN')
        total_timesteps: Number of timesteps to train for
        eval_episodes: Number of episodes to evaluate the policy
        normalize: Whether to use VecNormalize wrapper (not recommended for all environments)
        return_model: Whether to return the trained model

    Returns:
        If return_model=False: (mean_reward, std_reward)
        If return_model=True: (mean_reward, std_reward, model, env)
    """
    try:
        # Wrap the environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Only normalize if requested (not always appropriate)
        if normalize:
            env = VecNormalize(env)

        # Create and train the model
        if algorithm == 'A2C':
            model = A2C('MlpPolicy', env, verbose=0)
        elif algorithm == 'PPO':
            model = PPO('MlpPolicy', env, verbose=0)
        elif algorithm == 'DQN':
            model = DQN('MlpPolicy', env, verbose=0)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported algorithms: A2C, PPO, DQN")

        model.learn(total_timesteps=total_timesteps)

        # Evaluate the trained model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

        if return_model:
            return mean_reward, std_reward, model, env
        else:
            return mean_reward, std_reward

    except Exception as e:
        print(f"Error training {algorithm}: {str(e)}")
        raise

def run_benchmarks(env, algorithms=['A2C', 'PPO', 'DQN'], total_timesteps=100000, eval_episodes=10, normalize=False):
    """
    Run benchmarks for multiple algorithms.

    Args:
        env: The environment to train on
        algorithms: List of algorithm names to benchmark
        total_timesteps: Number of timesteps to train for each algorithm
        eval_episodes: Number of episodes to evaluate each policy
        normalize: Whether to use VecNormalize wrapper

    Returns:
        Dictionary with results for each algorithm
    """
    results = {}
    for algo in algorithms:
        try:
            print(f"Training {algo}...")
            mean_reward, std_reward = train_and_evaluate(
                env, algo, total_timesteps, eval_episodes, normalize=normalize, return_model=False
            )
            results[algo] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'success': True
            }
            print(f"{algo} - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Failed to train {algo}: {str(e)}")
            results[algo] = {
                'mean_reward': None,
                'std_reward': None,
                'success': False,
                'error': str(e)
            }
    return results