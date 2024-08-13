import gymnasium as gym
env = gym.make('Pendulum-v1', render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(100):
    action =env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
