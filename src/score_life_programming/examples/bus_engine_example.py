import numpy as np
import time
import multiprocessing as mp
from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming

def run_bus_engine_example():
    """
    Run Score Life Programming on Bus Engine environment with sequential and parallel computation.
    """
    gamma = 0.60
    N = 100
    j_max = 9
    num_samples = 500
    reference_state = 0
    
    print("Running Score Life Programming on Bus Engine environment...")
    
    bus_engine_env = BusEngineEnvironment(x=reference_state, p=0.3, q=0.4)
    slp = ScoreLifeProgramming(bus_engine_env, gamma, N, j_max, num_samples, reference_state)
    
    # Sequential computation
    start_time = time.time()
    score_function_ref_state = slp.compute_score_function_reference_state()
    end_time = time.time()
    print("Total Time taken to compute Score-life function of reference state sequentially:", end_time-start_time)

    # Parallel computation
    start_time_parallel = time.time()
    score_function_ref_state_parallel = slp.compute_score_function_reference_state_parallel()
    end_time_parallel = time.time()
    print("Total Time taken to compute Score-life function of reference state in parallel:", end_time_parallel-start_time_parallel)

    # Calculate and display optima
    print("Optima (gradient descent):", score_function_ref_state.compute_optima_gradient_descent())
    print("Optima (Hölder method):", score_function_ref_state.holder_minimize(1e-6, 1000))
    
    # Visualize fractal functions
    print("Visualizing sequential and parallel results...")
    score_function_ref_state.visualize_fractal()
    score_function_ref_state_parallel.visualize_fractal()
    
    # Uncomment to compute score-function for arbitrary state:
    # time_state_start = time.time()
    # state = 0
    # slp.compute_score_function_one_step(state, 1).visualize_fractal()
    # time_state_end = time.time()
    # print("Time taken to compute Score-life function a particular state:", time_state_end-time_state_start)

if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    mp.freeze_support()
    run_bus_engine_example()