"""
Test Methods for computing and optimizing Fractal Function
"""

import pytest
import numpy as np
from src.utils.fractal import Fractal
from unittest.mock import patch

def test_initialization():
    fractal = Fractal(1, 2, [3, 4, 5])
    assert fractal.alpha_0 == 1
    assert fractal.alpha_1 == 2
    assert fractal.coefficients == [3, 4, 5]

# Add new comprehensive tests
def test_simple_linear_fractal():
    """Test with a simple linear function (should have minimum at 0)"""
    fractal = Fractal(alpha_0=0, alpha_1=1, coefficients=[], j_shift=0)
    
    # Test compute_fractal
    assert abs(fractal.compute_fractal(0) - 0) < 1e-6
    assert abs(fractal.compute_fractal(1) - 1) < 1e-6
    
    # Test minimizers
    min_val, min_x = fractal.holder_minimize()
    assert abs(min_x) < 1e-4  # Should be very close to 0
    
    min_x_grad = fractal.compute_optima_gradient_descent()
    assert abs(min_x_grad) < 1e-4  # Should be very close to 0

def test_quadratic_like_fractal():
    """Test with a fractal that approximates a quadratic function"""
    coeffs = [[0.5, -0.5]]  # One level of coefficients
    fractal = Fractal(alpha_0=0, alpha_1=0, coefficients=coeffs, j_shift=0)
    
    # The minimum should be around 0.5
    min_val, min_x = fractal.holder_minimize()
    assert 0.4 < min_x < 0.6
    
    min_x_grad = fractal.compute_optima_gradient_descent()
    assert 0.4 < min_x_grad < 0.6

def test_boundary_conditions():
    """Test that the optimizers respect the [0,1] boundary"""
    fractal = Fractal(alpha_0=0, alpha_1=-10, coefficients=[], j_shift=0)
    
    # Both minimizers should return x=1 as it's the maximum allowed value
    min_val, min_x = fractal.holder_minimize()
    assert abs(min_x - 1.0) < 1e-4
    
    min_x_grad = fractal.compute_optima_gradient_descent()
    assert abs(min_x_grad - 1.0) < 1e-4

def test_consistency_between_optimizers():
    """Test that both optimization methods give similar results"""
    coeffs = [[0.3, -0.2], [0.1, 0.1, 0.1, 0.1]]
    fractal = Fractal(alpha_0=1, alpha_1=0.5, coefficients=coeffs, j_shift=0)
    
    # Get results from both methods
    min_val, min_x_holder = fractal.holder_minimize()
    min_x_grad = fractal.compute_optima_gradient_descent()
    
    # Compute function values at both minima
    val_holder = fractal.compute_fractal(min_x_holder)
    val_grad = fractal.compute_fractal(min_x_grad)
    
    # The function values at the minima should be close
    assert abs(val_holder - val_grad) < 0.1

def test_input_validation():
    """Test that the Fractal class properly handles invalid inputs"""
    with pytest.raises(Exception):
        # Test with invalid coefficients structure
        Fractal(alpha_0=0, alpha_1=0, coefficients=[[1], [1, 1, 1]], j_shift=0)
        
    # Test with invalid input to compute_fractal
    fractal = Fractal(alpha_0=0, alpha_1=0, coefficients=[], j_shift=0)
    with pytest.raises(Exception):
        fractal.compute_fractal("invalid")

# Keep existing helper function tests
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
    fractal = Fractal(1, 2, [[3], [4,5]])
    optimum = fractal.compute_optima_gradient_descent()
    assert 0 <= optimum <= 1  # Example assertion, replace with actual expected behavior

if __name__ == "__main__":
    pytest.main([__file__])


