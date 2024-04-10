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
from src.utils.fractal import Fractal

class ScoreLifeProgramming:

    def __init__(self, env, gamma):

        """
        Initialize the Score-life programming algorithm.

        :param model: An instance of a dynamics model (e.g., CartPoleModel).
        :param gamma: Discount factor for future rewards.
        :param coeff: Coefficients of Faber Schauder Expansion

        """
        self.env = env
        self.gamma = gamma



    def _action_sequence_to_real(self, action_sequence) -> float:

        """
        Maps an action sequence to a real number in the interval [0, 1).

        :param action_sequence: A sequence of actions.
        :return: A real number representing the action sequence.
        """

        pass

    def _real_to_action_sequence(self, real_number,num_bits):

        """
        Maps a real number back to an action sequence.

        :param real_number: A real number in the interval [0, 1).
        :return: String Representing Corresponding action sequence.

        """
        if real_number == 0:
            return '.' + '0' * (num_bits - 1)
        elif real_number == 1:
            return  '.' + '1' * num_bits
        else:
            binary = ''
            # Check if the fraction is less than 1
        if real_number < 1:
            binary += '.'
        for i in range(num_bits):
            real_number *= 2
            if real_number >= 1:
                binary += '1'
                real_number -= 1
            else:
                binary += '0'
        return binary


    def compute_life_value(self, state) -> int:
        """
        Computes the 'life value' for a given state.

        :param state: The state for which to compute the life value.
        :return: The computed life value.
        """
        pass

    def optimize_score_life_function(self, initial_state):
        """
        Optimizes the Score-life function to find the optimal action sequence.

        :param initial_state: The state from which optimization starts.
        :return: The optimal action sequence and its corresponding life value.
        """
        pass

    def custom_reward(state,action):

        """
        Custom Reward Function implementing LQR cost

        :param n: state dimension
        :param q: diagonal entries of Q Matrix
        :return reward: computed reward/cost 
        
        """

        n = len(state)
        q = np.ones(n)
        q = np.array([2,1,8,1])
        Q = np.diag(q)
        reward = state@Q@state.T

        return reward
    

    def S(self,l,X,N):
        self.env.reset()
        R = 0
        action_sequence =self._real_to_action_sequence(l,num_bits = N)
        self.env.state = self.env.unwrapped.state = X
        for i in range(len(action_sequence)-1):
            action = int(action_sequence[i+1])
            state, reward, terminated, truncated, info  = self.env.step(action)
            #reward = custom_reward(state,action) optional to implement custom reward functions
            R = (self.gamma**(i))*reward + R
        self.env.close()
        return R

       

    def _compute_a_ij(self,i,j,X,N):
        l_1 = (2*i + 1)/(2**(j+1))
        l_2 = i/(2**j)
        l_3 = (i+1)/(2**j)
        a_ij = self.S(l_1,X, self.gamma,N,self.env) - 0.5*(S(l_2,X, self.gamma,N,self.env)+ S(l_3,X,self.gamma,N,self.env))
        return a_ij
    
    def compute_faber_schauder_coefficients(self, X, N,j_max):
        a_0 = self.S(0,X,self.gamma,N,self.env)
        print(a_0)
        a_1 = self.S(1,X,self.gamma,N,self.env) - self.S(0,X,self.gamma,N,self.env)
        print(a_1)
        ####compute a_i,j
        i = 0
        j = 0
        coefficients = []
        while j < j_max:
            i = 0
            c_j = []
            while i <= 2**j - 1:
                a_i_j = self._compute_a_ij(i,j,X,N)
                c_j.append(a_i_j)
                i = i + 1
            coefficients.append(c_j)
            j = j + 1
        fractal_function = Fractal(a_0,a_1,coefficients)
        return fractal_function
    
    def _derivative_mod_x(a,b,x):
        ##function to compute derivative of |ax - b|
        if x == b/a:
            derivative = -a
        else:
            derivative = a*(abs(a*x - b)/(a*x - b))
        return derivative
       
    def d_S_i_j(self,l,i,j):
        derivative = (2**j)*(self._derivative_mod_x(1,(i/(2**j)),l) + self._derivative_mod_x(1,((i+1)/(2**j)),l) - self._derivative_mod_x(2,((2*i+1)/(2**j)),l))
        return derivative

    def grad_score_life_function(self, a_0,a_1,coefficients,l):
        grad_f = a_1
        j_max = len(coefficients)
        j = 0
        while j < j_max:
            i = 0
            while i <=2**j - 1:
                grad_f = grad_f + self.d_S_i_j(l,i,j)*coefficients[j][i]
                i = i + 1
            j = j + 1
        return grad_f

    def S_i_j(l,i,j):
        val = (2**j)*(abs(l-(i/(2**j))) + abs(l-((i+1)/(2**j))) - abs(2*l-((2*i+1)/(2**(j)))))
        return val
    
    def compute_score_life_function(self,a_0,a_1,coefficients,l):
        f = a_0 + a_1*l
        j_max = len(coefficients)
        j = 0
        while j < j_max:
            i = 0
            while i <= 2**j - 1:
                f = f + self.S_i_j(l,i,j)*coefficients[j][i]
                i = i + 1
            j = j + 1
        return f
    
    
    

env = gym.make("CartPole-v0")
N = 100
X = np.array([0,0,0,0])
gamma = 0.5
a_0 = S(0,X,gamma,N,env)
#print(a_0)
a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
#print(a_1)

j_max = 10

####plot faber schauder function
a_0,a_1,coefficients = compute_faber_schauder_coefficients(X,gamma,N,j_max,env)
# Generate data for x and y values
l = np.linspace(0, 1, 1000)
y = []
for val in l:
    y_val = compute_score_life_function(a_0,a_1,coefficients,val)
    y.append(y_val)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(l, y, color='blue', linewidth=0.5)

# Set the title and axis labels
ax.set_title('Exact Representation of Score-life function')
ax.set_xlabel('l')
ax.set_ylabel('S(l,x)')

# Set the x-axis limits
ax.set_xlim([0, 1])

# Display the plot
#plt.show()
plt.savefig("Score_life_function")
#print(S_i_j(0,0,0))

l_optimal, J_optimal,i_array,l_array,grad_array = compute_optimal_l(a_0,a_1,coefficients)

fig, ax = plt.subplots()

# Plot the data
ax.plot(i_array,grad_array, color='blue', linewidth=2)
ax.plot(i_array,l_array, color='red', linewidth=2)

# Set the title and axis labels
ax.set_title('Fractal Optimization Convergence Plot')
ax.set_xlabel('l')
ax.set_ylabel('grad_squared')

# Set the x-axis limits
ax.set_xlim([0, 1])

# Display the plot
plt.show()
#plt.savefig("")
env_2 = gym.make("CartPole-v1", render_mode="human")
gamma = 0.5
N = 100
j_max = 10
observation, info = env_2.reset(seed=42)
k = 0
N_action_horizon = 10
x_array =[]
x_dot_array = []
theta_array = []
theta_dot_array = []

for i in range(1000):
    #compute faber schauder coefficients:
    if k == 0:
        a_0,a_1,coefficients = compute_faber_schauder_coefficients(observation,gamma,N,j_max,env)
        l_optimal, J_optimal,i_array,l_array,grad_array = compute_optimal_l(a_0,a_1,coefficients)
        action_sequence = fraction_to_binary(l_optimal,N_action_horizon)
        print(action_sequence)
    if k < N_action_horizon - 1:
        action = int(action_sequence[k+1])
        k = k + 1
        print(k)
    if k == N_action_horizon-1:
        k = 0
#    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env_2.step(action)
    print(observation)
    x_array.append(observation[0])
    x_dot_array.append(observation[1])
    theta_array.append(observation[2])
    theta_dot_array.append(observation[3])
    env_2.render()
    if terminated or truncated:
        print("terminating...Iterations:",i)
        break
        observation, info = env.reset()

fig, ax = plt.subplots()
ax.plot(x_array, label=f'Trajectory - x')
ax.plot(x_dot_array, label=f'Trajectory  - x_dot')
ax.plot(theta_array, label=f'Trajectory - theta')
ax.plot(theta_dot_array, label=f'Trajectory  - theta_dot')
ax.set_title('Simulation Trajectories')
ax.set_xlabel('Time')
ax.set_ylabel('Values')
# Show legend
ax.legend()
# Display the plot
plt.savefig(f'Simulation_results_exact.jpg', dpi=300)
plt.show()
env.close()
env_2.close()
