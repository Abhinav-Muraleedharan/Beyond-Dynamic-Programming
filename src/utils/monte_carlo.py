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

    """
    Implements a Monte Carlo simulation approach for evaluating the performance 
    of different action sequences in a given environment. This class focuses on 
    exploring the state-action space to approximate the score-life function, which 
    represents the utility or value of different states or actions in the environment.
    
    The Monte Carlo method used here samples different action sequences to estimate
    their expected rewards, allowing for the evaluation of the effectiveness of 
    various strategies over the episode length specified.

    Attributes:
        state (numpy.ndarray): The current state of the environment.
        gamma (float): Discount factor, which balances the importance of immediate 
                       versus future rewards.
        episode_length (int): The number of steps/actions to simulate in each episode.
        score_life_function (list): Placeholder for storing the score-life function 
                                    results. [Currently not used in the provided code]
        env (gym.Env): An instance of a Gym environment where the simulation is run.
        R_array (numpy.ndarray): An array to store the cumulative rewards for each 
                                 sampled action sequence.
        l_array (numpy.ndarray): An array to store the real-number representations 
                                 of the action sequences sampled.
        env_name (str): Name of the Gym environment being used for simulation.
        action_size (int): The number of possible actions in the environment's action space.

    Methods:
        reset(): Resets the arrays used to store simulation results.
        run_monte_carlo(desired_state, max_iterations): Runs the Monte Carlo simulation 
                                                        for a specified number of iterations, 
                                                        starting from the desired_state.
        plot(iteration_no): Generates and saves a scatter plot of the simulation results, 
                            plotting the real-number representations of action sequences 
                            against their corresponding cumulative rewards.
    
    The Monte Carlo simulation generates a variety of action sequences, evaluates their 
    outcomes in terms of cumulative rewards, and visualizes these results to aid in 
    understanding the potential effectiveness of different strategies within the specified 
    environment. This approach is useful in environments where calculating the exact 
    value of states or actions is computationally infeasible, providing a method to 
    approximate these values through sampling.
    
    Note:
        The `custom_reward` function is defined outside of this class and is intended to 
        calculate a custom reward based on the current state and the action taken. This 
        function can be customized or replaced depending on the specific requirements of 
        the environment or the goals of the simulation.
     """

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
        plt.xlabel('life Values')
        plt.ylabel('Score Values')
        plt.title(f'Score life function')
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
    i = 0
    N = 20
    initial_state = np.array([0.5, 0,0,0])
    state = initial_state
    env2.reset()
    env2.state = env2.unwrapped.state = initial_state
    while i < N:
        for a in range(action_size):
            experiment = Monte_Carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
            experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
            experiment.plot(state) #plot results
            state, reward, terminated, truncated, info = env2.step(a)
            print(state)
        i = i + 1
