import numpy as np
from src.environments.bus_engine import BusEngineEnvironment
from src.utils.score_life_programming import ScoreLifeProgramming


bus_engine_env =  BusEngineEnvironment(x=2,p=0.1,q=0.3)
gamma = 0.65
N = 100
j_max = 8
num_samples  = 1000
reference_state = 0

s = ScoreLifeProgramming(bus_engine_env,gamma, N, j_max,num_samples,reference_state )


print(s.score_function_reference_state.coefficients)
s.score_function_reference_state.visualize_fractal()