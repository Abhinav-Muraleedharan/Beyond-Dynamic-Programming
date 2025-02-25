#!/usr/bin/env python

"""
Test script to verify the refactored Score Life Programming examples are working correctly.
"""

import os
import sys
import multiprocessing as mp
from src.score_life_programming.examples import run_mountain_car_example, run_bus_engine_example

def main():
    """Run a quick test of the refactored examples."""
    print("Testing refactored Score Life Programming examples...")
    
    # Setup for multiprocessing
    mp.freeze_support()
    
    # Use command line argument to determine which example to run
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
    else:
        test_name = "mountain_car"  # Default to mountain car as it's faster
    
    print(f"Running test for: {test_name}")
    
    if test_name == "mountain_car":
        run_mountain_car_example()
    elif test_name == "bus_engine":
        run_bus_engine_example()
    else:
        print(f"Unknown test: {test_name}")
        print("Available tests: mountain_car, bus_engine")
        return 1
    
    print("Test completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())