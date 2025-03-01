import numpy as np
import time
from src.environments.mountain_car import MountainCarEnv
from src.utils.score_life_programming import ScoreLifeProgramming


mountain_car_env =  MountainCarEnv()
gamma = 0.99
N = 195
j_max = 7
num_samples  = 1
reference_state = [0.4,0.02]

start_time = time.time()
s = ScoreLifeProgramming(mountain_car_env,gamma, N,j_max,num_samples,reference_state)

print(s.score_function_reference_state.coefficients)
s.score_function_reference_state.visualize_fractal()
end_time = time.time()
print("Total Time taken to compute Score-life function of reference state:", end_time-start_time)
print("The minima of Score life function is given by:",s.score_function_reference_state.holder_minimize(1e-6,1000))

# compute score-function for arbitrary state:
# time_state_start = time.time()
# state = 100
# s.compute_score_function_one_step(state,1).visualize_fractal()
# time_state_end = time.time()

# print("Time taken to compute Score-life function a particular state:", time_state_end-time_state_start)

