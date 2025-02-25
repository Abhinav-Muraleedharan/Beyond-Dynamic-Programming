"""
Score Life Programming - A Beyond Dynamic Programming Method

This package implements the Score Life Programming method as described in
the "Beyond Dynamic Programming" paper. It provides both exact and approximate
methods for computing optimal policies using fractal functions.

The package includes:
- Exact methods for computing Faber-Schauder coefficients
- Approximate methods for faster computation
- Example implementations for various environments
"""

from src.score_life_programming.exact_methods import compute_faber_schauder_coefficients, compute_optimal_l
from src.score_life_programming.approximate_methods import approximate_score_life_function

__all__ = [
    'compute_faber_schauder_coefficients',
    'compute_optimal_l',
    'approximate_score_life_function',
]