# src/environments/acrobot.py

import gym
import numpy as np

class AcrobotEnv(gym.Env):
    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def discretize_state(self, state):
        # Discretize the continuous state space
        cos_theta1_bins = np.linspace(-1, 1, 10)
        sin_theta1_bins = np.linspace(-1, 1, 10)
        cos_theta2_bins = np.linspace(-1, 1, 10)
        sin_theta2_bins = np.linspace(-1, 1, 10)
        theta1dot_bins = np.linspace(-12.57, 12.57, 10)
        theta2dot_bins = np.linspace(-28.27, 28.27, 10)
        
        discretized = (
            np.digitize(state[0], cos_theta1_bins),
            np.digitize(state[1], sin_theta1_bins),
            np.digitize(state[2], cos_theta2_bins),
            np.digitize(state[3], sin_theta2_bins),
            np.digitize(state[4], theta1dot_bins),
            np.digitize(state[5], theta2dot_bins)
        )
        
        return discretized

    def get_all_states(self):
        return [(i, j, k, l, m, n) for i in range(10) for j in range(10) 
                for k in range(10) for l in range(10) for m in range(10) for n in range(10)]

    def get_next_states(self, state, action):
        # Simulate the environment for one step
        self.env.state = self.continuous_state(state)
        next_state, reward, done, _ = self.env.step(action)
        return self.discretize_state(next_state), reward, done

    def continuous_state(self, discrete_state):
        cos_theta1_bins = np.linspace(-1, 1, 10)
        sin_theta1_bins = np.linspace(-1, 1, 10)
        cos_theta2_bins = np.linspace(-1, 1, 10)
        sin_theta2_bins = np.linspace(-1, 1, 10)
        theta1dot_bins = np.linspace(-12.57, 12.57, 10)
        theta2dot_bins = np.linspace(-28.27, 28.27, 10)
        
        return [cos_theta1_bins[discrete_state[0]], sin_theta1_bins[discrete_state[1]],
                cos_theta2_bins[discrete_state[2]], sin_theta2_bins[discrete_state[3]],
                theta1dot_bins[discrete_state[4]], theta2dot_bins[discrete_state[5]]]