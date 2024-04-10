import gymnasium as gym
env_name = "CartPole-v1"
#env = gym.make(env_name, render_mode="human")
env = gym.make(env_name)
env2  = gym.make(env_name)
#env.action_space.seed(42)

from scipy.optimize import curve_fit
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import scipy
from scipy.stats import gaussian_kde
import matplotlib as mpl
# Create the FrozenLake environment
#env = gym.make('FrozenLake-v0')
def quadratic_S(l, a, b, c):
    return a *( l**2) + b * l + c

def optimize_quadratic(coefficients):
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    ###evaluate at 0
    S_0 = quadratic_S(0,a,b,c)
    ###evaluate at 1
    S_1 = quadratic_S(1,a,b,c)
    ###find minima,
    l_optima = -b/(2*a)
    if 0 <= l_optima <= 1:
        S_optima = quadratic_S(l_optima,a,b,c)
    else:
        S_optima = 1e16
    cost_to_go = min(S_0,S_1,S_optima)
    return cost_to_go
    
    
def compute_a_ij(i,j,X,gamma,N,env):
  l_1 = (2*i + 1)/(2**(j+1))
  l_2 = i/(2**j)
  l_3 = (i+1)/(2**j)
  a_ij = S(l_1,X, gamma,N,env) - 0.5*(S(l_2,X, gamma,N,env)+ S(l_3,X,gamma,N,env))
  return a_ij

def custom_reward(state,action):
    n = len(state)
    q = np.ones(n)
    q = np.array([10,20,1,1])
    Q = np.diag(q)
    reward = state@Q@state.T
    return reward


def fraction_to_binary(l,num_bits,M):
    import math
    if l == 0:
      return '.' + '0' * (num_bits - 1)
    elif l == 1:
       return  '.' + str(M-1) * num_bits
    else:
      binary = '.'
      for i in range(num_bits):
          f,i = math.modf(M*l)
          l = f
          # print(l)
          # print(i)
          binary   += str(int(i))
      return binary

def S(l,X,gamma,N,env):
#    env = gym.make("CartPole-v0")
    env.reset()
    M = env.action_space.n
    M=3
    R = 0
    action_sequence = fraction_to_binary(l,N,M)
#    print(action_sequence)
#    print(action_sequence)
    env.state = env.unwrapped.state = X
    for i in range(len(action_sequence)-1):
        action = int(action_sequence[i+1])
        state, reward, terminated, truncated, info  = env.step(action)
#        reward = custom_reward(state,action)
        reward = -reward
        R = (gamma**(i))*reward + R
        if terminated == True:
#            R = R +(gamma**(i))*100
             print("hi")
#            break
    env.close()
    env.reset()
    return R
  
def derivative_mod_x(a,b,x):
    ##function to compute derivative of |ax - b|
    if x == b/a:
        derivative = -a
    else:
        derivative = a*(abs(a*x - b)/(a*x - b))
    return derivative
    
def d_S_i_j(l,i,j):
    derivative = (2**j)*(derivative_mod_x(1,(i/(2**j)),l) + derivative_mod_x(1,((i+1)/(2**j)),l) - derivative_mod_x(2,((2*i+1)/(2**j)),l))
    return derivative

def grad_score_life_function(a_0,a_1,coefficients,l):
    grad_f = a_1
    j_max = len(coefficients)
    j = 0
    while j < j_max:
        i = 0
        while i <=2**j - 1:
            grad_f = grad_f + d_S_i_j(l,i,j)*coefficients[j][i]
            i = i + 1
        j = j + 1
    return grad_f


def S_i_j(l,i,j):
    val = (2**j)*(abs(l-(i/(2**j))) + abs(l-((i+1)/(2**j))) - abs(2*l-((2*i+1)/(2**(j)))))
    return val
    
def compute_score_life_function(a_0,a_1,coefficients,l):
    f = a_0 + a_1*l
    j_max = len(coefficients)
    j = 0
    while j < j_max:
        i = 0
        while i <= 2**j - 1:
            f = f + S_i_j(l,i,j)*coefficients[j][i]
            i = i + 1
        j = j + 1
    
    return f
  
def compute_optimal_l(a_0,a_1,coefficients):
    ####optimize Score-life function:
    max_iter = 6000
    i = 0
    lr = 0.01 # learning rate
    l = 0.001 #initialize l
    grad_prev = 0
    l_array = []
    grad_array = []
    i_array = []
    while i < max_iter:
        if i == 0:
            l = random.random()
        grad = grad_score_life_function(a_0,a_1,coefficients,l)
        l = l - grad*lr
        lr = lr*(2**(-i))
#        print(l)
#        print("square of gradient:")
        grad_sq = grad**2
#        print(grad**2)
        grad_array.append(grad_sq)
        l_array.append(l)
        i_array.append(i)
        if grad*grad_prev < 0:
#            print("grad square:")
#            print(grad**2)
            if grad**2 < 0.01:
                break
        grad_prev = grad
        if l < 0:
            l = 0
            break
        if l > 1:
            l = 0.9999999
            break
        i = i +1
    print("Optimal l:!!")
    print("iterations:",i)
    print(l)
    print("Optimal Cost:")
    J_optimal = compute_score_life_function(a_0,a_1,coefficients,l)
    print(J_optimal)
    return l, J_optimal,i_array,l_array,grad_array
    
def evaluate_quadratic_score_life_function(state,n,N_horizon,gamma,env):
    l = np.random.uniform(0, 1, n)
    S_approx = []
    for el in l:
        S_val = S(el,state,gamma,N_horizon,env)
        S_approx.append(S_val)
    S_approx = np.array(S_approx)
    popt, pcov = curve_fit(quadratic_S,l,S_approx)
    a_opt, b_opt, c_opt = popt
    return a_opt,b_opt,c_opt

def plot_quadratic(a_opt,b_opt,c_opt):
    l = np.linspace(0, 1, 100)
    S_approx = quadratic_S(l,a_opt,b_opt,c_opt)
    plt.plot(l,S_approx,color = 'red',label ='Approximate Score-life function')
    plt.xlabel('l')
    plt.ylabel('Approximate Score-life-function')
    return plt
    
def compute_faber_schauder_coefficients(X,gamma,N,j_max,env):
    a_0 = S(0,X,gamma,N,env)
    print(a_0)
    a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
    print(a_1)
    ####compute a_i,j
    i = 0
    j = 0
    coefficients = []
    while j < j_max:
        i = 0
        c_j = []
        while i <= 2**j - 1:
            a_i_j = compute_a_ij(i,j,X,gamma,N,env)
            c_j.append(a_i_j)
            i = i + 1
        coefficients.append(c_j)
        j = j + 1
    return a_0,a_1, coefficients

def compute_cost_to_go(state,n,N,gamma,env):
    a_opt,b_opt,c_opt = evaluate_quadratic_score_life_function(state,n,N,gamma,env)
    coefficients_quad = [a_opt,b_opt,c_opt]
    J = optimize_quadratic(coefficients_quad)
    return J
    

def compute_Q(state,env,n,N,gamma):
    Q = []
    for a in range(env.action_space.n):
        env.state = env.unwrapped.state = state
        next_state, reward, terminated, truncated, info = env.step(a)
#        reward = custom_reward(observation,action)
        J = compute_cost_to_go(next_state,n,N,gamma,env)
#        reward = custom_reward(next_state,a)
        reward = -reward
        J =  reward + gamma*J
        Q.append(J)
    env.reset()
    return Q
    
    
#
#env = gym.make("CartPole-v0")
#N = 200
#X = np.array([0,0,0,0])  #####state
#gamma = 0.8
#a_0 = S(0,X,gamma,N,env)
#a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
#j_max = 10
#n = 100
#a_0,a_1,coefficients = compute_faber_schauder_coefficients(X,gamma,N,j_max,env)
#a_opt,b_opt,c_opt = evaluate_quadratic_score_life_function(X,n,N,gamma,env)
#plt = plot_quadratic(a_opt,b_opt,c_opt)
#
## Generate data for x and y values
#l = np.linspace(0, 1, 1000)
#y = []
#for val in l:
#    y_val = compute_score_life_function(a_0,a_1,coefficients,val)
#    y.append(y_val)
#
## Create a figure and axis object
##fig, ax = plt.subplots()
## Plot the data
#plt.plot(l, y, color='blue', linewidth=0.5)


# Set the title and axis labels
#ax.set_title('Exact Representation of Score-life function')
#ax.set_xlabel('l')
#ax.set_ylabel('S(l,x)')

# Set the x-axis limits
#ax.set_xlim([0, 1])

# Display the plot
#plt.show()




#plt.show()
#coefficients_quad = [a_opt,b_opt,c_opt]
#J = optimize_quadratic(coefficients_quad)
#print("Cost to go: ")
#print(J)
##print(S_i_j(0,0,0))
##plt.savefig("")
#env_2 = gym.make("CartPole-v1", render_mode="human")
#gamma = 0.8
#N = 200
#j_max = 10
#observation, info = env_2.reset()
#k = 0
##N_action_horizon = 10
#n = 200
#### simulation::::
#for i in range(1000):
#    #compute faber schauder coefficients:
##    action = env.action_space.sample()
#    env.reset()
#    Q = compute_Q(observation,env,n,N,gamma)
#    print(Q)
#    action = Q.index(min(Q))
#    print("action:")
#    print(action)
#    observation, reward, terminated, truncated, info = env_2.step(action)
#    print(observation)
#    env_2.render()
#    if terminated or truncated:
#        print("terminating...Iterations:",i)
#        break
#        observation, info = env.reset()
#env.close()
#env_2.close()
