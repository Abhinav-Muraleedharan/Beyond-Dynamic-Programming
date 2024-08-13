#!/bin/bash

# run_experiments.sh

# Activate virtual environment (uncomment and modify if using a virtual environment)
# source /path/to/your/venv/bin/activate

# Set the Python path to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run all experiments
python experiments/run_experiments.py

# Visualize results
python experiments/visualize_results.py

echo "All experiments completed and results visualized."