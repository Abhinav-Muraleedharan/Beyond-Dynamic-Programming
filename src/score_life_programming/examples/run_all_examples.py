"""
Run all Score Life Programming examples.

This script provides a simple command-line interface to run all or specific
Score Life Programming examples from the examples package.
"""

import sys
import time
import multiprocessing as mp
from src.score_life_programming.examples import run_mountain_car_example, run_bus_engine_example

def main():
    """
    Run examples based on command-line arguments.
    
    Usage:
        python -m src.score_life_programming.examples.run_all_examples [example_name]
        
    If no example_name is provided, all examples will be run sequentially.
    
    Example names:
        - mountain_car
        - bus_engine
        - all (default)
    """
    # Setup for multiprocessing support
    mp.freeze_support()
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
    else:
        example_name = 'all'
    
    if example_name == 'mountain_car':
        print("Running Mountain Car example...")
        run_mountain_car_example()
    elif example_name == 'bus_engine':
        print("Running Bus Engine example...")
        run_bus_engine_example()
    elif example_name == 'all':
        print("Running all examples sequentially...")
        print("\n" + "="*50)
        print("MOUNTAIN CAR EXAMPLE")
        print("="*50)
        run_mountain_car_example()
        
        print("\n" + "="*50)
        print("BUS ENGINE EXAMPLE")
        print("="*50)
        run_bus_engine_example()
        
        print("\n" + "="*50)
        print("All examples completed successfully")
        print("="*50)
    else:
        print(f"Unknown example: {example_name}")
        print("Available examples: mountain_car, bus_engine, all")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())