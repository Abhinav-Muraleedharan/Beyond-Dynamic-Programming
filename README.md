# Beyond Dynamic Programming

This project implements and compares various reinforcement learning algorithms, including the novel Score-life programming method introduced in the "Beyond Dynamic Programming" paper, alongside benchmarks from Stable Baselines 3.

## Project Structure

```
beyond-dynamic-programming/
├── README.md
├── requirements.txt
├── src/
│   ├── score_life_programming/
│   ├── benchmarks/
│   ├── environments/
│   └── utils/
├── tests/
├── experiments/
├── results/
└── run_experiments.sh
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/beyond-dynamic-programming.git
   cd beyond-dynamic-programming
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running Experiments

To run all experiments, use the provided shell script:

```
chmod +x run_experiments.sh
./run_experiments.sh
```

This script will run experiments for CartPole, Mountain Car, Acrobot, and Lunar Lander environments, comparing the novel Score-life programming methods with Stable Baselines 3 algorithms (A2C, PPO, DQN).

## Results

The results of the experiments will be saved in the `results/` directory. Each experiment will generate plots and data files comparing the performance of different methods.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.