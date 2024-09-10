import numpy as np
import time
from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


bus_engine_env =  BusEngineEnvironment(x=2,p=0.1,q=0.3)
gamma = 0.65
N = 100
j_max = 8
num_samples  = 1000
reference_state = 0

start_time = time.time()
s = ScoreLifeProgramming(bus_engine_env,gamma, N, j_max,num_samples,reference_state )


print(s.score_function_reference_state.coefficients)
s.score_function_reference_state.visualize_fractal()
end_time = time.time()
print("Total Time taken to compute Score-life function of reference state:", end_time-start_time)


print(s.score_function_reference_state.compute_optima_gradient_descent())

# compute score-function for arbitrary state:
time_state_start = time.time()
state = 1000000000
s.compute_score_function_one_step(state,1).visualize_fractal()
time_state_end = time.time()

print("Time taken to compute Score-life function a particular state:", time_state_end-time_state_start)