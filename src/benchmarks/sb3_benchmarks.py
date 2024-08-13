# src/benchmarks/sb3_benchmarks.py

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def train_and_evaluate(env, algorithm, total_timesteps=100000, eval_episodes=10):
    # Wrap the environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    # Create and train the model
    if algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=0)
    elif algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=0)
    elif algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.learn(total_timesteps=total_timesteps)

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    return mean_reward, std_reward

def run_benchmarks(env, algorithms=['A2C', 'PPO', 'DQN'], total_timesteps=100000, eval_episodes=10):
    results = {}
    for algo in algorithms:
        mean_reward, std_reward = train_and_evaluate(env, algo, total_timesteps, eval_episodes)
        results[algo] = {'mean_reward': mean_reward, 'std_reward': std_reward}
    return results