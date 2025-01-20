# src/environments/mountain_car.py

import gym 
import numpy as np

class MountainCarEnv(gym.Env):
    def __init__(self):
        self.env = gym.make('MountainCar-v0', render_mode="rgb_array")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()
    
    def set_state(self, state):
        self.env.state = state 

    def step(self, action):
        #print(self.env.step(action))
        next_state, reward, done,truncated,_ = self.env.step(action)
        return next_state, reward, done, truncated

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def discretize_state(self, state):
        # Discretize the continuous state space
        pos_bins = np.linspace(-1.2, 0.6, 20)
        vel_bins = np.linspace(-0.07, 0.07, 20)
        
        pos_digit = np.digitize(state[0], pos_bins)
        vel_digit = np.digitize(state[1], vel_bins)
        
        return (pos_digit, vel_digit)

    def get_all_states(self):
        return [(i, j) for i in range(20) for j in range(20)]

    def get_next_states(self, state, action):
        # Simulate the environment for one step
        self.env.state = self.continuous_state(state)
        next_state, reward, done, _ = self.env.step(action)
        return self.discretize_state(next_state), reward, done

    def continuous_state(self, discrete_state):
        pos_bins = np.linspace(-1.2, 0.6, 20)
        vel_bins = np.linspace(-0.07, 0.07, 20)
        
        return [pos_bins[discrete_state[0]], vel_bins[discrete_state[1]]]