# src/score_life_programming/exact_methods.py

import numpy as np
import random
from src.utils.fractal import Fractal


def _real_to_action_sequence_base(real_number, num_bits, base):
    """Maps a real number to its representation in the specified base."""
    if real_number == 0:
        return "." + "0" * num_bits
    elif real_number == 1:
        return "." + str(base - 1) * num_bits

    result = "."
    current_number = real_number

    for _ in range(num_bits):
        current_number *= base
        digit = int(current_number)
        result += str(digit)
        current_number -= digit

    return result


def S(l, X, gamma, N, env, num_samples=10):
    """Evaluates Score function at a specific l-value for a given state X."""
    R = 0
    M = env.action_space.n
    action_sequence = _real_to_action_sequence_base(l, N, M)

    avg_R = 0
    for _ in range(num_samples):
        R = 0
        env.reset()
        if hasattr(env, "set_state"):
            env.set_state(X)
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "state"):
            env.unwrapped.state = X

        for i in range(len(action_sequence) - 1):
            action = int(action_sequence[i + 1])
            if hasattr(env, "step"):
                try:
                    result = env.step(action)
                    if len(result) == 5:
                        state, reward, done, truncated, _ = result
                    else:
                        state, reward, done, _ = result
                except:
                    state = X
                    reward = 0
                    done = True
                    truncated = False
            else:
                state = X
                reward = 0
                done = True
                truncated = False

            R = (gamma**i) * reward + R

            if done or truncated:
                break

        avg_R += R

    avg_R = avg_R / num_samples
    return avg_R


def compute_faber_schauder_coefficients(x, gamma, N, j_max, env):
    """Computes all Faber Schauder Coefficients of a given state X."""
    a_0 = S(0, x, gamma, N, env)
    a_1 = S(1, x, gamma, N, env) - S(0, x, gamma, N, env)

    coefficients = []
    for j in range(j_max):
        c_j = []
        for i in range(2**j):
            l_1 = (2 * i + 1) / (2 ** (j + 1))
            l_2 = i / (2**j)
            l_3 = (i + 1) / (2**j)
            a_ij = S(l_1, x, gamma, N, env) - 0.5 * (
                S(l_2, x, gamma, N, env) + S(l_3, x, gamma, N, env)
            )
            c_j.append(a_ij)
        coefficients.append(c_j)

    return a_0, a_1, coefficients


def _e_i_j(l, i, j, coefficients, j_shift=0):
    """Compute the e_ij term in Faber Schauder Expansion."""
    j = j - j_shift
    val = (2**j) * (
        abs(l - (i / (2**j)))
        + abs(l - ((i + 1) / (2**j)))
        - abs(2 * l - ((2 * i + 1) / (2**j)))
    )
    return val


def _grad_fractal(l, a_0, a_1, coefficients, j_shift=0):
    """Function to estimate derivative of Fractal function at any input value l."""
    grad_f = a_1
    j_max = len(coefficients)
    for j in range(j_max):
        for i in range(2**j):
            derivative = (2**j) * (
                _derivative_mod_x(1, (i / (2**j)), l)
                + _derivative_mod_x(1, ((i + 1) / (2**j)), l)
                - _derivative_mod_x(2, ((2 * i + 1) / (2**j)), l)
            )
            grad_f += derivative * coefficients[j][i]
    return grad_f


def _derivative_mod_x(a, b, x):
    """Function to compute derivative of |ax - b|."""
    if x == b / a:
        return -a
    else:
        return a * (abs(a * x - b) / (a * x - b))


def compute_score_life_function(l, a_0, a_1, coefficients):
    """Compute the score life function value at l."""
    f = a_0 + a_1 * l
    j_max = len(coefficients)
    for j in range(j_max):
        for i in range(2**j):
            f += _e_i_j(l, i, j, coefficients) * coefficients[j][i]
    return f


def grad_score_life_function(l, a_0, a_1, coefficients):
    """Compute the gradient of the score life function at l."""
    return _grad_fractal(l, a_0, a_1, coefficients)


def compute_optimal_l(a_0, a_1, coefficients):
    """Compute optimal l using gradient descent."""
    max_iter = 100
    lr = 0.01
    l = random.random()
    grad_prev = 0

    for i in range(max_iter):
        grad = grad_score_life_function(l, a_0, a_1, coefficients)
        l = l - grad * lr
        lr = lr * (2 ** (-i))

        if l < 0:
            l = 0
        elif l > 1:
            l = 0.9999999

        if grad * grad_prev < 0 and grad**2 < 0.01:
            break

        grad_prev = grad

    J_optimal = compute_score_life_function(l, a_0, a_1, coefficients)
    return l, J_optimal


def run_exact_method(env, gamma=0.99, N=100, j_max=10, num_samples=5):
    """Run the exact score-life programming method on an environment."""
    observation, _ = env.reset()
    total_reward = 0
    max_steps = 500

    for _ in range(max_steps):
        try:
            a_0, a_1, coefficients = compute_faber_schauder_coefficients(
                observation, gamma, N, j_max, env
            )
            l_optimal, _ = compute_optimal_l(a_0, a_1, coefficients)
            action = int(l_optimal * env.action_space.n)
            action = max(0, min(action, env.action_space.n - 1))
        except Exception as e:
            action = env.action_space.sample()

        observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            break

    return total_reward
