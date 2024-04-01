import gymnasium as gym
import time
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    env.render()
    if terminated or truncated:
        print("terminating...Iterations:",i)
        break
        observation, info = env.reset()
env.close()
