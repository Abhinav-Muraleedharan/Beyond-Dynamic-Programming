import gymnasium as gym 
import numpy as np
from src.utils.monte_carlo import Monte_Carlo


gamma = 0.01
episode_length = 100 
env = gym.make("CartPole-v1", render_mode="human")
env_name = "CartPole-v1"
action_size = 2
state = np.array([0.5, 0,0,0])

experiment = monte_carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
experiment.plot(state) #plot results