"""
Test Methods for computing and optimizing Fractal Function
"""



from src.utils.fractal import Fractal  
import pytest
from unittest.mock import patch

def test_initialization():
    fractal = Fractal(1, 2, [3, 4, 5])
    print("sfs")
    assert fractal.alpha_0 == 1
    assert fractal.alpha_1 == 2
    assert fractal.coefficients == [3, 4, 5]

def test_derivative_mod_x():
    fractal = Fractal(0, 1, [[2]])
    result = fractal._derivative_mod_x(2, 3, 1.5)
    assert result == -2  # Adjust expected result based on correct behavior

def test_d_e_i_j():
    fractal = Fractal(0, 1, [[2]])
    result = fractal._d_e_i_j(0.5, 0, 0)
    assert result is not None  # Example assertion, replace with actual expected behavior

def test_grad_fractal():
    fractal = Fractal(0, 1, [[2]])
    result = fractal._grad_fractal(0.5)
    assert result is not None  # Example assertion, replace with actual expected behavior

def test_e_i_j():
    fractal = Fractal(0, 1, [[2]])
    result = fractal._e_i_j(0.5, 0, 0)
    assert result is not None  # Example assertion, replace with actual expected behavior

def test_compute_fractal():
    fractal = Fractal(1, 2, [[3], [4,5]])
    result = fractal.compute_fractal(0.5)
    assert result is not None  # Example assertion, replace with actual expected behavior

@patch('random.random', return_value=0.5)
def test_compute_optima_gradient_descent(mock_random):
    fractal = Fractal(1, 2, [[3], [4,5]] )
    optimum = fractal.compute_optima_gradient_descent()
    assert 0 <= optimum <= 1  # Example assertion, replace with actual expected behavior


