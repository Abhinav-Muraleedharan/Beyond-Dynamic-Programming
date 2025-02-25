import numpy as np
import time
from src.environments.mountain_car import MountainCarEnv
from src.utils.score_life_programming import ScoreLifeProgramming

def run_mountain_car_example():
    """
    Run Score Life Programming on Mountain Car environment.
    """
    mountain_car_env = MountainCarEnv()
    gamma = 0.99
    N = 195
    j_max = 7
    num_samples = 1
    reference_state = [0.4, 0.02]
    
    print("Running Score Life Programming on Mountain Car environment...")
    start_time = time.time()
    
    slp = ScoreLifeProgramming(mountain_car_env, gamma, N, j_max, num_samples, reference_state)
    
    print(slp.score_function_reference_state.coefficients)
    slp.score_function_reference_state.visualize_fractal()
    
    end_time = time.time()
    
    print("Total Time taken to compute Score-life function of reference state:", end_time-start_time)
    print("The minima of Score life function is given by:", slp.score_function_reference_state.holder_minimize(1e-6, 1000))
    
    # Uncomment to compute score-function for arbitrary state:
    # time_state_start = time.time()
    # state = 100
    # slp.compute_score_function_one_step(state, 1).visualize_fractal()
    # time_state_end = time.time()
    # print("Time taken to compute Score-life function a particular state:", time_state_end-time_state_start)

if __name__ == "__main__":
    run_mountain_car_example()