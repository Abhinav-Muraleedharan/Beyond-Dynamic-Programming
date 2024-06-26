Beyond Dynamic Programming
==============================

Code Associated with the paper [Beyond Dynamic Programming](https://arxiv.org/abs/2306.15029)

![Local Image](./src/visualization/fractal_image_1.jpeg)


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── envs          <- RL Environments
    │   │   └── acrobot
    |   |   └── cart_pole
    |   |   └── lunar_lander
    |   |   └── mountain_car
    |   |   └── simple_pendulum
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   ├── utils         <- Scripts continaing Score-life programming methods
    |   |   |
    |   |   ├── approximate_score_life_programming.py   <- Approximate Method
    |   |   ├── fractal.py                              <- Class definition of Fractal Functions
    |   |   ├── monte_carlo.py                          <- Monte Carlo Evaluation Scripts
    |   |   ├── score_life_programming.py               <- Exact Method

    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



