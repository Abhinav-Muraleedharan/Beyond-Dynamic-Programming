# src/environments/lunar_lander.py

import gym
import numpy as np

class LunarLanderEnv(gym.Env):
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
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
        x_bins = np.linspace(-1, 1, 10)
        y_bins = np.linspace(-1, 1, 10)
        vx_bins = np.linspace(-2, 2, 10)
        vy_bins = np.linspace(-2, 2, 10)
        angle_bins = np.linspace(-3.14, 3.14, 10)
        angular_velocity_bins = np.linspace(-5, 5, 10)
        left_leg_bins = np.linspace(0, 1, 2)
        right_leg_bins = np.linspace(0, 1, 2)
        
        discretized = (
            np.digitize(state[0], x_bins),
            np.digitize(state[1], y_bins),
            np.digitize(state[2], vx_bins),
            np.digitize(state[3], vy_bins),
            np.digitize(state[4], angle_bins),
            np.digitize(state[5], angular_velocity_bins),
            np.digitize(state[6], left_leg_bins),
            np.digitize(state[7], right_leg_bins)
        )
        
        return discretized

    def get_all_states(self):
        return [(i, j, k, l, m, n, o, p) for i in range(10) for j in range(10) 
                for k in range(10) for l in range(10) for m in range(10) 
                for n in range(10) for o in range(2) for p in range(2)]

    def get_next_states(self, state, action):
        # Simulate the environment for one step
        self.env.state = self.continuous_state(state)
        next_state, reward, done, _ = self.env.step(action)
        return self.discretize_state(next_state), reward, done

    def continuous_state(self, discrete_state):
        x_bins = np.linspace(-1, 1, 10)
        y_bins = np.linspace(-1, 1, 10)
        vx_bins = np.linspace(-2, 2, 10)
        vy_bins = np.linspace(-2, 2, 10)
        angle_bins = np.linspace(-3.14, 3.14, 10)
        angular_velocity_bins = np.linspace(-5, 5, 10)
        left_leg_bins = np.linspace(0, 1, 2)
        right_leg_bins = np.linspace(0, 1, 2)
        
        return [x_bins[discrete_state[0]], y_bins[discrete_state[1]],
                vx_bins[discrete_state[2]], vy_bins[discrete_state[3]],
                angle_bins[discrete_state[4]], angular_velocity_bins[discrete_state[5]],
                left_leg_bins[discrete_state[6]], right_leg_bins[discrete_state[7]]]