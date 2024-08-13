import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    print(env.action_space.sample())
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
