import gym
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
import multiprocessing as mp
from functools import partial
import time

def parallel_simulate_trajectory(params):
    """
    Standalone function for parallel simulation
    params: tuple of (X, action_sequence, p, q, gamma)
    """
    X, action_sequence, p, q, gamma = params
    state = X
    R = 0
    
    for i in range(len(action_sequence)-1):
        action = int(action_sequence[i+1])
        
        # Replicate the step logic from BusEngineEnvironment
        if action == 1:
            next_state = 0
            utility = -2 * ((1-action) * state) - action * 100
            state = next_state
        else:
            u = np.random.uniform()
            if u < p:
                delta_x = np.random.uniform(0, 5000)
            elif p <= u < p + q:
                delta_x = np.random.uniform(5000, 10000)
            else:
                delta_x = np.random.uniform(10000, 100000000)
            
            next_state = state + delta_x
            state = next_state
            utility = -2 * ((1-action) * state) - action * 100
        
        R = (gamma**i) * utility + R
        
        # if action == 1:  # Terminal state
        #     break
    
    return R

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
        self.action_dim = env.action_space.n
        self.j_max = j_max
        self.num_samples = num_samples
        self.n_cores = mp.cpu_count()
        #print(f"Available CPU cores: {self.n_cores}")

    def compute_score_function_reference_state(self):
        return self._compute_faber_schauder_coefficients()
        
    def compute_score_function_reference_state_parallel(self):
        return self._compute_faber_schauder_coefficients_parallel()

    def _real_to_action_sequence_base(self, real_number, num_bits, base):
        """
        Maps a real number to its representation in the specified base.

        :param real_number: A real number in the interval [0, 1).
        :param num_bits: Number of digits to compute in the representation.
        :param base: The base to convert to (e.g., 2 for binary, 3 for ternary).
        :return: String representing the number in the specified base.
        """
        if real_number == 0:
            return '.' + '0' * num_bits
        elif real_number == 1:
            return '.' + str(base-1) * num_bits
        
        result = '.'
        current_number = real_number
        
        for _ in range(num_bits):
            current_number *= base
            digit = int(current_number)  # Get the integer part
            result += str(digit)
            current_number -= digit  # Subtract the integer part
            
        return result






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
            nxt_state,reward,_,_ = self.env.step(action)
            #print(nxt_state)
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

        #self.env.reset()
        R = 0
        #action_sequence =self._real_to_action_sequence(l,num_bits = self.N)
        M = self.action_dim
        action_sequence = self._real_to_action_sequence_base(l,self.N,M)
        #action_sequence =self._real_to_action_sequence(l,num_bits = self.N)
        #print(action_sequence)
        self.env.set_state(X)
        avg_R = 0

        for j in range(self.num_samples):
            R = 0
            self.env.set_state(X)
            #print("Initial State",self.env.current_state())
            for i in range(len(action_sequence)-1):
                action = int(action_sequence[i+1])
                state, reward,done,truncated  = self.env.step(action)
                #reward = (state[0])**2 + reward
                #print(reward)
                #reward = custom_reward(state,action) optional to implement custom reward functions
                R = (self.gamma**(i))*reward + R
                #print("reward",reward)
                #print("state",state)
                if done:
                    #print("done")
                    break
                if truncated:
                    #print("truncated")
                    break
            avg_R = avg_R + R
        avg_R = avg_R/self.num_samples
        
        return avg_R

       

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
        a_1 = self.S(1,X) - self.S(0,X)
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
    
    def _simulate_trajectory_parallel(self, X, action_sequence, seed):
        """Helper function to simulate a single trajectory in parallel"""
        np.random.seed(seed)
        self.env.set_state(X)
        R = 0
        for i in range(len(action_sequence)-1):
            action = int(action_sequence[i+1])
            state, reward, done, truncated = self.env.step(action)
            R = (self.gamma**i) * reward + R
            if done or truncated:
                break
        return R

    def S_parallel(self, l, X, existing_pool=None):
        """Modified parallel Score function evaluation that can use an existing pool"""
        start_time = time.time()
        action_sequence = self._real_to_action_sequence_base(l, self.N, self.action_dim)
        
        # Prepare parameters for parallel simulation
        params = [(X, action_sequence, self.env.p, self.env.q, self.gamma) 
                 for _ in range(self.num_samples)]
        
        # Use existing pool if provided, otherwise create new one
        if existing_pool is not None:
            results = existing_pool.map(parallel_simulate_trajectory, params)
        else:
            with mp.Pool(processes=self.n_cores) as pool:
                results = pool.map(parallel_simulate_trajectory, params)
        
        avg_R = sum(results) / self.num_samples
        end_time = time.time()
        print(f"Parallel S computation took: {end_time - start_time:.2f} seconds")
        return avg_R

    def _compute_a_ij_parallel(self, X, i, j, pool):
        """Modified helper function that uses existing pool"""
        l_1 = (2*i + 1)/(2**(j+1))
        l_2 = i/(2**j)
        l_3 = (i+1)/(2**j)
        a_ij = self.S_parallel(l_1, X, pool) - 0.5*(self.S_parallel(l_2, X, pool) + self.S_parallel(l_3, X, pool))
        return a_ij

    def _compute_faber_schauder_coefficients_parallel(self):
        """Modified parallel version that uses a single pool"""
        start_time = time.time()
        X = self.reference_state
        j_max = self.j_max

        # Create a single pool for all computations
        with mp.Pool(processes=self.n_cores) as pool:
            print("Computing a_0 and a_1...")
            a_0 = self.S_parallel(0, X, pool)
            a_1 = self.S_parallel(1, X, pool) - self.S_parallel(0, X, pool)

            # Prepare all parameters for coefficient computation
            params = []
            for j in range(j_max):
                for i in range(2**j):
                    l_1 = (2*i + 1)/(2**(j+1))
                    l_2 = i/(2**j)
                    l_3 = (i+1)/(2**j)
                    params.append((X, l_1, l_2, l_3, self.env.p, self.env.q, self.gamma, self.N, self.action_dim, self.num_samples))

            print(f"Computing {len(params)} Faber-Schauder coefficients in parallel...")
            results = pool.map(parallel_compute_coefficient, params)

        # Reorganize results
        coefficients = []
        idx = 0
        for j in range(j_max):
            c_j = []
            for i in range(2**j):
                c_j.append(results[idx])
                idx += 1
            coefficients.append(c_j)

        end_time = time.time()
        print(f"Parallel Faber-Schauder computation complete! Took {end_time - start_time:.2f} seconds")
        return Fractal(a_0, a_1, coefficients, 0)

    def compare_performance(self, l, X):
        """
        Compare performance between parallel and sequential implementations
        """
        print("\nComparing performance between sequential and parallel implementations:")
        
        # Sequential timing
        start_time = time.time()
        sequential_result = self.S(l, X)
        sequential_time = time.time() - start_time
        print(f"Sequential computation took: {sequential_time:.2f} seconds")
        
        # Parallel timing
        start_time = time.time()
        parallel_result = self.S_parallel(l, X)
        parallel_time = time.time() - start_time
        print(f"Parallel computation took: {parallel_time:.2f} seconds")
        
        speedup = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results are similar
        print(f"Results difference: {abs(sequential_result - parallel_result):.6f}")
        
        return {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'sequential_result': sequential_result,
            'parallel_result': parallel_result
        }

def parallel_compute_coefficient(params):
    """Standalone function for parallel coefficient computation"""
    X, l_1, l_2, l_3, p, q, gamma, N, action_dim, num_samples = params
    
    # Compute S values for each l
    s1 = parallel_compute_S(X, l_1, p, q, gamma, N, action_dim, num_samples)
    s2 = parallel_compute_S(X, l_2, p, q, gamma, N, action_dim, num_samples)
    s3 = parallel_compute_S(X, l_3, p, q, gamma, N, action_dim, num_samples)
    
    # Compute coefficient
    a_ij = s1 - 0.5 * (s2 + s3)
    return a_ij

def parallel_compute_S(X, l, p, q, gamma, N, action_dim, num_samples):
    """Standalone function for parallel S computation"""
    # Convert l to action sequence
    action_sequence = _real_to_action_sequence_base_standalone(l, N, action_dim)
    
    # Run simulations
    results = []
    for _ in range(num_samples):
        params = (X, action_sequence, p, q, gamma)
        results.append(parallel_simulate_trajectory(params))
    
    return sum(results) / num_samples

def _real_to_action_sequence_base_standalone(real_number, num_bits, base):
    """Standalone version of _real_to_action_sequence_base"""
    if real_number == 0:
        return '.' + '0' * num_bits
    elif real_number == 1:
        return '.' + str(base-1) * num_bits
    
    result = '.'
    current_number = real_number
    
    for _ in range(num_bits):
        current_number *= base
        digit = int(current_number)
        result += str(digit)
        current_number -= digit
        
    return result


