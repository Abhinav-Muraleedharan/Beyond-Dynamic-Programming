# src/score_life_programming/approximate_methods.py

import numpy as np
from scipy.optimize import curve_fit
from src.score_life_programming.exact_methods import S as compute_S


def quadratic_S(l, a, b, c):
    return a * (l**2) + b * l + c


def optimize_quadratic(coefficients):
    a, b, c = coefficients
    S_0 = quadratic_S(0, a, b, c)
    S_1 = quadratic_S(1, a, b, c)
    if a != 0:
        l_optima = -b / (2 * a)
        if 0 <= l_optima <= 1:
            S_optima = quadratic_S(l_optima, a, b, c)
        else:
            S_optima = float("inf")
    else:
        S_optima = float("inf")
    return min(S_0, S_1, S_optima)


def evaluate_quadratic_score_life_function(
    state, n, N_horizon, gamma, env, num_samples=5
):
    """Evaluate quadratic approximation of score-life function."""
    l_samples = np.random.uniform(0, 1, n)
    S_approx = [
        compute_S(el, state, gamma, N_horizon, env, num_samples) for el in l_samples
    ]

    try:
        popt, _ = curve_fit(quadratic_S, l_samples, S_approx, maxfev=5000)
    except Exception:
        popt = [0, 0, np.mean(S_approx)]

    return popt


def compute_cost_to_go(state, n, N, gamma, env):
    """Compute cost-to-go using quadratic approximation."""
    a_opt, b_opt, c_opt = evaluate_quadratic_score_life_function(
        state, n, N, gamma, env
    )
    coefficients_quad = [a_opt, b_opt, c_opt]
    J = optimize_quadratic(coefficients_quad)
    return J


def compute_Q(state, env, n, N, gamma):
    """Compute Q-values using approximate score-life programming."""
    Q = []
    for a in range(env.action_space.n):
        try:
            next_state, reward, done, truncated, _ = env.step(a)
            if done or truncated:
                J = 0
            else:
                J = compute_cost_to_go(next_state, n, N, gamma, env)
            Q.append(-reward + gamma * J)
        except Exception:
            Q.append(0)

    env.reset()
    return Q


def run_approximate_method(env, gamma=0.99, N=100, n=10, num_samples=5):
    """Run the approximate score-life programming method on an environment."""
    observation, _ = env.reset()
    total_reward = 0
    max_steps = 500

    for _ in range(max_steps):
        try:
            Q = compute_Q(observation, env, n, N, gamma)
            action = np.argmin(Q) if Q else env.action_space.sample()
        except Exception:
            action = env.action_space.sample()

        try:
            observation, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if done or truncated:
                break
        except Exception:
            break

    return total_reward
