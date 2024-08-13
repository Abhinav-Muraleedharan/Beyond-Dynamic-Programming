# src/score_life_programming/approximate_methods.py

import numpy as np
from scipy.optimize import curve_fit

def quadratic_S(l, a, b, c):
    return a * (l**2) + b * l + c

def optimize_quadratic(coefficients):
    a, b, c = coefficients
    S_0 = quadratic_S(0, a, b, c)
    S_1 = quadratic_S(1, a, b, c)
    l_optima = -b / (2*a)
    if 0 <= l_optima <= 1:
        S_optima = quadratic_S(l_optima, a, b, c)
    else:
        S_optima = float('inf')
    return min(S_0, S_1, S_optima)

def evaluate_quadratic_score_life_function(state, n, N_horizon, gamma, env):
    def S(l, X, gamma, N, env):
        # Implement the S function as described in the paper
        pass

    l = np.random.uniform(0, 1, n)
    S_approx = [S(el, state, gamma, N_horizon, env) for el in l]
    popt, _ = curve_fit(quadratic_S, l, S_approx)
    return popt

def compute_cost_to_go(state, n, N, gamma, env):
    a_opt, b_opt, c_opt = evaluate_quadratic_score_life_function(state, n, N, gamma, env)
    coefficients_quad = [a_opt, b_opt, c_opt]
    J = optimize_quadratic(coefficients_quad)
    return J

def compute_Q(state, env, n, N, gamma):
    Q = []
    for a in range(env.action_space.n):
        next_state, reward, _, _, _ = env.step(a)
        J = compute_cost_to_go(next_state, n, N, gamma, env)
        Q.append(-reward + gamma * J)
    env.reset()
    return Q

# Add more functions as needed