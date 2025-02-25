# Score Life Programming Examples

This directory contains example implementations of the Score Life Programming method applied to different reinforcement learning environments.

## Available Examples

### Mountain Car

The Mountain Car example applies Score Life Programming to the classic Mountain Car environment where a car must build momentum to reach the top of a hill.

```python
# Run the Mountain Car example
python -m src.score_life_programming.examples.run_all_examples mountain_car
```

### Bus Engine

The Bus Engine example demonstrates using Score Life Programming in a maintenance scheduling environment, showing both sequential and parallel implementations.

```python
# Run the Bus Engine example
python -m src.score_life_programming.examples.run_all_examples bus_engine
```

## Running All Examples

To run all examples sequentially:

```python
# Run all examples
python -m src.score_life_programming.examples.run_all_examples
# or
python -m src.score_life_programming.examples.run_all_examples all
```

## Adding New Examples

To add a new example:

1. Create a new Python file in this directory (e.g. `new_environment_example.py`)
2. Implement a main function that shows Score Life Programming in your environment
3. Update the `__init__.py` file to export your example function
4. Add your example to the `run_all_examples.py` script

## Performance Notes

- The Bus Engine example demonstrates both sequential and parallel implementations
- Parallel implementation is particularly useful for larger state spaces and longer time horizons
- Visualizations of the resulting score functions are produced automatically