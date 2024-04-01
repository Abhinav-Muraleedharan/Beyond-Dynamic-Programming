import gymnasium as gym
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import scipy
from scipy.stats import gaussian_kde
import matplotlib as mpl

def custom_reward(state,action):
    des_state = np.array([0.5,0])
    n = len(state)
    q = np.ones(n)
    Q = np.diag(q)
    reward = (state-des_state)@Q@(state-des_state).T
    return reward

class Monte_Carlo:
    def __init__(self,action_size,gamma,episode_length,state,env,env_name):
        self.state = state
        self.gamma = gamma
        self.episode_length = episode_length
        self.score_life_function = []
        self.env = env
        self.R_array = np.empty(1)
        self.l_array = np.empty(1)
        self.env_name = env_name
        self.action_size = action_size

    def reset(self):
        self.R_array = np.empty(1)
        self.l_array = np.empty(1)
    def run_monte_carlo(self,desired_state,max_iterations):
        iterations = 0
        self.env.reset()
        while iterations < max_iterations:
            l  = 0
            R = 0
            self.env.state = self.env.unwrapped.state = desired_state
            print(self.env.state)
            for i in range(self.episode_length):
                action = self.env.action_space.sample()
                print("action",action)
                l = l + ((int(self.action_size))**(-i-1))*action
                observation, reward, terminated, truncated, info = self.env.step(action)
                reward = -reward
                R = (self.gamma**(i))*reward + R
                if terminated == True:
                    self.env.reset()
                    break
            self.R_array = np.append(self.R_array,R)
            self.l_array = np.append(self.l_array,l)
            iterations = iterations + 1           
        
    def plot(self,iteration_no):
        if not os.path.exists(f'results_monte_carlo_simulation_2{self.env_name}'):
            os.makedirs(f'results_monte_carlo_simulation_2{self.env_name}')
        sns.set_theme(style="ticks")
        x = self.l_array
        y = self.R_array
        plt.clf()
        kernel = gaussian_kde(np.vstack([x, y]))
        c = kernel(np.vstack([x, y]))
        plt.xlim(0,1)
        plt.scatter(x, y, s=1, c=c, cmap=mpl.cm.viridis, edgecolor='none')
#        sns.scatterplot(x=self.l_array, y=self.R_array, color='purple')
        # Add labels and title to the plot
        plt.xlabel('life Values')
        plt.ylabel('Score Values')
        plt.title(f'Score life function')
        # Create a folder to save the image
        # Save the plot as a high-quality jpg image
        plt.savefig(f'results_monte_carlo_simulation_2{self.env_name}/monte_carlo_{iteration_no}.jpg', dpi=300)
        plt.close()
        plt.clf()

   

### define variables:
        
if __name__ == "main":
    env = gym.make("CartPole-v1", render_mode="human")
    env2 = gym.make("CartPole-v1", render_mode="human")
    gamma = 0.5
    action_size = env.action_space.n
    episode_length = 500
    max_iterations = 500
    #####
    i = 0
    N = 20
    #state = np.array([-0.16889614, -0.9115705]) #initialize state
    #
    #experiment = monte_carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
    #experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
    #experiment.plot(state) #plot results
    #experiment.reset()
    initial_state = np.array([0.5, 0,0,0])
    state = initial_state
    env2.reset()
    env2.state = env2.unwrapped.state = initial_state
    while i < N:
        for a in range(action_size):
#           state, reward, terminated, truncated, info = env2.step(a)
            experiment = monte_carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
            experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
            experiment.plot(state) #plot results
            state, reward, terminated, truncated, info = env2.step(a)
            print(state)
#        experiment.reset()
        i = i + 1

######random score life functions
#
#lower_bound = -10
#upper_bound =  10
#i = 0
#N = 0
#while i < N:
#    random_state = np.random.uniform(lower_bound, upper_bound, size=4)
#    experiment = monte_carlo(action_size, gamma,episode_length,random_state,env,env_name) #initialize class
#    experiment.run_monte_carlo(random_state,max_iterations) #run monte carlo simulations
#    experiment.plot(random_state) #plot results
#    experiment.reset()
#    i = i + 1
