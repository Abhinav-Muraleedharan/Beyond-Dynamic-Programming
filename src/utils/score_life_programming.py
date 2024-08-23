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

    def __init__(self, env, gamma, N,j_max, num_samples, reference_state):

        """
        Initialize the Score-life programming algorithm.

        :param model: An instance of a dynamics model (e.g., CartPoleModel).
        :param gamma: Discount factor for future rewards.
        :ground_state: Initial State x_0 
        :param N: Maximum number of timesteps taken in simulation to evaluate score-life function
        :param j_max: max order of expansion of Faber Schauder Series
        :param score_function_ground_state: Score-life function of ground state

        """
        self.env = env
        self.gamma = gamma
        self.reference_state = reference_state
        self.N = N
        self.j_max = j_max
        self.num_samples = num_samples
        self.score_function_reference_state = self._compute_faber_schauder_coefficients()


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
    
    def _divide_nested_list(self,nested_list, constant):
        if isinstance(nested_list, list):
            return [self._divide_nested_list(item, constant) for item in nested_list]
        else:
            return nested_list / constant

    def compute_score_function_one_step(self,state,action) -> Fractal:
        avg_reward = 0
        self.env.set_state(state)
        for i in range(self.num_samples):
            nxt_state,reward = self.env.step(action)
            avg_reward = avg_reward + reward
            self.env.set_state(state)
        avg_reward = avg_reward/self.num_samples
        print(avg_reward)
        beta_0 = (self.score_function_reference_state.alpha_0 + avg_reward)*self.gamma
        beta_1 = self.score_function_reference_state.alpha_1*self.gamma
        print(self.score_function_reference_state.coefficients)
        coeff_prime = self._divide_nested_list(self.score_function_reference_state.coefficients,1/self.gamma)
        j_shift = 1
        score_function_one_step = Fractal(beta_0,beta_1,coeff_prime ,j_shift)
        return score_function_one_step 



    def compute_score_function(self, state) -> Fractal:
        """
        Computes the Score-life function of input state and compute optimal action sequence

        :param state: The state for which to compute the life value.
        :return: The computed Fractal Function.

        """
        score_life_function = self._compute_faber_schauder_coefficients(self, state, self.N,self.j_max)
        l_optimal = score_life_function.compute_optima_gradient_descent()

        return score_life_function,l_optimal


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
    

    def S(self,l,X):


        """

        Evaluates Score function at a specific l-value for a given state X

        :param l: life-value
        :param X: State Vector
        :return R: computed infinite horizon cost
        
        """

        self.env.reset()
        R = 0
        action_sequence =self._real_to_action_sequence(l,num_bits = self.N)
        self.env.set_state(X)
        avg_R = 0

        for j in range(self.num_samples):
            for i in range(len(action_sequence)-1):
                action = int(action_sequence[i+1])
                state, reward  = self.env.step(action)
                #reward = custom_reward(state,action) optional to implement custom reward functions
                R = (self.gamma**(i))*reward + R
        avg_R = avg_R + R
        avg_R = avg_R/self.num_samples
        
        return R

       

    def _compute_a_ij(self,i,j,X):

        """
        Computes a specific Faber Schauder Coefficient of a given state indexed by i and j [Eq(23)]
        :param i: index of coefficient
        :param j: index of coefficient
        :param X: input state

        :return a_ij: computed coefficient
        
        """
    
        l_1 = (2*i + 1)/(2**(j+1))
        l_2 = i/(2**j)
        l_3 = (i+1)/(2**j)
        a_ij = self.S(l_1,X) - 0.5*(self.S(l_2,X)+ self.S(l_3,X))
        return a_ij
    
    def _compute_faber_schauder_coefficients(self):

        """
        Computes all Faber Schauder Coefficients of a given state X
        :param X: State
        :param j_max: Max value of the index
        
        :return fractal_function: fractal_function representing Score-life function of the given state X
        
        """
        X = self.reference_state
        j_max = self.j_max 
        a_0 = self.S(0,X)
        a_1 = self.S(1,X,) - self.S(0,X)
        ####compute a_i,j
        i = 0
        j = 0
        coefficients = []
        while j < j_max:
            i = 0
            c_j = []
            while i <= 2**j - 1:
                a_i_j = self._compute_a_ij(i,j,X)
                c_j.append(a_i_j)
                i = i + 1
            coefficients.append(c_j)
            j = j + 1
        fractal_function = Fractal(a_0,a_1,coefficients,0)
        return fractal_function
    
    def _derivative_mod_x(a,b,x):

        """
        Function to compute derivate of |ax - b|. Used for computing derivatives of Score-life function
        :param x: Value x
        :param a: Value a
        :param b: Value b

        :return derivative: derivative of |ax - b| at x. 
        
        """
        if x == b/a:
            derivative = -a
        else:
            derivative = a*(abs(a*x - b)/(a*x - b))
        return derivative
       
    def d_S_i_j(self,l,i,j):
        
        """
        Function to compute derivate of e_ij term (basis functions of Faber Schuder Expansion). 
        :param l: Life Value l
        :param i: Index i
        :param j: Index j
        
        :return derivative: derivative of e_ij(l) at l. 
        
        """
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
    

