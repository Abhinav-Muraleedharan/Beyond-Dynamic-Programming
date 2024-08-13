# Beyond Dynamic Programming

This project implements and compares various reinforcement learning algorithms, including the novel Score-life programming method introduced in the "Beyond Dynamic Programming" paper, alongside state-of-the-art methods from Stable Baselines 3.

## Setup

### Using Conda (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/your-username/beyond-dynamic-programming.git
   cd beyond-dynamic-programming
   ```

2. Create the conda environment from the `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```

3. Activate the conda environment:
   ```
   conda activate beyond-dp
   ```

### Alternative: Using pip

If you prefer not to use conda, you can set up a virtual environment and install the dependencies using pip:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/beyond-dynamic-programming.git
   cd beyond-dynamic-programming
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running Experiments

You can run experiments for all environments or a specific environment using the provided shell script.

To run all experiments:

```
chmod +x run_experiments.sh
./run_experiments.sh
```

To run experiments for a specific environment:

```
./run_experiments.sh <environment_name>
```

Replace `<environment_name>` with one of the following:

- CartPole-v1
- MountainCar-v0
- Acrobot-v1
- LunarLander-v2
- GridWorld
- Taxi-v3
- FrozenLake-v1
- Blackjack-v1
- CliffWalking-v0

For example:

```
./run_experiments.sh CartPole-v1
```

## Results

The results of the experiments will be saved in the `results/` directory. Each experiment will generate plots comparing the performance of different methods for each environment.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.