import numpy as np
import time
from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming
import multiprocessing as mp

def main():
    
    gamma = 0.60
    N = 100
    j_max = 9
    num_samples  = 500
    reference_state = 0
    bus_engine_env =  BusEngineEnvironment(x=reference_state,p=0.3,q=0.4)

    
    s = ScoreLifeProgramming(bus_engine_env,gamma, N,j_max,num_samples,reference_state)
    start_time = time.time()
    #print(s.compute_score_function_reference_state().coefficients)
    score_function_ref_state = s.compute_score_function_reference_state()
    end_time = time.time()
    print("Total Time taken to compute Score-life function of reference state sequentially:", end_time-start_time)

    start_time_parallel = time.time()
    score_function_ref_state_parallel = s.compute_score_function_reference_state_parallel()
    end_time_parallel = time.time()
    print("Total Time taken to compute Score-life function of reference state in parallel:", end_time_parallel-start_time_parallel)

    print(score_function_ref_state.compute_optima_gradient_descent())
    print(score_function_ref_state.holder_minimize(1e-6,1000))
    score_function_ref_state.visualize_fractal()
    score_function_ref_state_parallel.visualize_fractal()

    # compute score-function for arbitrary state:
    # time_state_start = time.time()
    # state = 0
    # s.compute_score_function_one_step(state,1).visualize_fractal()
    # time_state_end = time.time()

    #print("Time taken to compute Score-life function a particular state:", time_state_end-time_state_start)

if __name__ == '__main__':
    # Required for multiprocessing on Windows and some Unix systems
    mp.freeze_support()
    main()

