"""
Bus Engine Replacement Problem - CANONICAL IMPLEMENTATION

This is the single, bug-free, canonical implementation.
All experiments should import from here.

Fixed bugs:
1. Boundary condition: p < u < p+q -> u < p+q
2. Consistent state representation (numpy array)
3. Gymnasium-compliant API (5-tuple return)
4. Configurable parameters (no hardcoded values)
5. Correct utility calculation order
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BusEngineEnvironment(gym.Env):
    """
    Bus Engine Replacement Problem.

    State: Engine mileage (continuous)
    Actions: 0 = Keep running, 1 = Replace engine

    Dynamics:
    - If replace: mileage resets to 0, cost = replacement_cost
    - If keep: mileage increases by random amount, cost = operating_cost_rate * mileage

    Mileage increment distribution (when keeping):
    - With probability p: Δx ~ Uniform(0, small_max)
    - With probability q: Δx ~ Uniform(small_max, medium_max)
    - With probability (1-p-q): Δx ~ Uniform(medium_max, large_max)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        p: float = 0.1,
        q: float = 0.3,
        max_state: int = 100000,
        replacement_cost: float = 100.0,
        operating_cost_rate: float = 0.01,
        small_max: float = 1000.0,
        medium_max: float = 3000.0,
        large_max: float = 10000.0,
    ):
        """
        Initialize Bus Engine Environment.

        Args:
            p: Probability of small damage
            q: Probability of medium damage
            max_state: Maximum mileage before capping
            replacement_cost: Cost to replace engine
            operating_cost_rate: Cost per unit mileage
            small_max: Upper bound for small damage range
            medium_max: Upper bound for medium damage range
            large_max: Upper bound for large damage range
        """
        super().__init__()

        self.p = p
        self.q = q
        self.max_state = max_state
        self.replacement_cost = replacement_cost
        self.operating_cost_rate = operating_cost_rate
        self.small_max = small_max
        self.medium_max = medium_max
        self.large_max = large_max

        # Gymnasium API requirements
        self.observation_space = spaces.Box(
            low=0, high=max_state, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.state = None
        self._max_episode_steps = 500
        self._current_step = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = np.array([0.0], dtype=np.float32)
        self._current_step = 0
        return self.state, {}

    def set_state(self, mileage):
        """Set current mileage state (for value iteration)."""
        if isinstance(mileage, np.ndarray):
            mileage = float(mileage[0])
        self.state = np.array([float(mileage)], dtype=np.float32)

    def current_state(self):
        """Get current mileage as scalar."""
        return self.state[0] if self.state is not None else 0.0

    def step(self, action):
        """
        Take action in environment.

        Args:
            action: 0 = keep running, 1 = replace engine

        Returns:
            observation: np.array([mileage])
            reward: negative cost (utility)
            terminated: always False (no terminal states)
            truncated: True if max_episode_steps reached
            info: dict with mileage and action
        """
        self._current_step += 1
        current_mileage = float(self.state[0])

        if action == 1:  # Replace engine
            next_mileage = 0.0
            cost = self.replacement_cost
        else:  # Keep running
            # Sample damage/mileage increment
            u = np.random.random()

            # FIXED: Correct boundary condition (no gap at u == p)
            if u < self.p:
                # Small damage
                delta_x = np.random.uniform(0, self.small_max)
            elif u < self.p + self.q:  # FIXED: was "self.p < u < self.p + self.q"
                # Medium damage
                delta_x = np.random.uniform(self.small_max, self.medium_max)
            else:
                # Large damage
                delta_x = np.random.uniform(self.medium_max, self.large_max)

            next_mileage = min(current_mileage + delta_x, float(self.max_state))
            cost = self.operating_cost_rate * next_mileage

        # Update state
        self.state = np.array([next_mileage], dtype=np.float32)

        # Compute reward (negative cost)
        reward = -cost

        # Episode termination
        terminated = False  # No terminal states in this problem
        truncated = self._current_step >= self._max_episode_steps

        # Info dict
        info = {
            "mileage": next_mileage,
            "action": "replace" if action == 1 else "keep",
            "cost": cost
        }

        return self.state, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render current state."""
        print(f"Mileage: {self.state[0]:.2f}")

    def close(self):
        """Clean up resources."""
        pass


# Backward compatibility: create old-style environment that matches original
class BusEngineEnvironmentLegacy:
    """
    Legacy version matching original src/environments/bus_engine.py
    (with bugs fixed).

    Use BusEngineEnvironment instead - this is only for comparing old results.
    """

    class ActionSpace:
        def __init__(self):
            self.n = 2
            self.actions = [0, 1]

    def __init__(self, x, p, q):
        """Initialize with specific starting mileage."""
        self.state = x
        self.p = p
        self.q = q
        self.action_space = self.ActionSpace()

        # Using experiment file parameters for consistency
        self.replacement_cost = 100.0
        self.operating_cost_rate = 0.01
        self.small_max = 1000.0
        self.medium_max = 3000.0
        self.large_max = 10000.0

    def reset(self):
        """Reset to initial state."""
        self.state = 0
        return self.state

    def current_state(self):
        """Get current state."""
        return self.state

    def set_state(self, x):
        """Set state."""
        self.state = x

    def step(self, action):
        """
        Take action, return (next_state, reward, done, terminated).

        FIXED: Correct boundary condition
        """
        if action == 1:  # Replace
            next_state = 0
            cost = self.replacement_cost
            self.state = next_state
        else:  # Keep running
            u = np.random.uniform()

            # FIXED: Correct boundary condition
            if u < self.p:
                delta_x = np.random.uniform(0, self.small_max)
            elif u < self.p + self.q:  # FIXED
                delta_x = np.random.uniform(self.small_max, self.medium_max)
            else:
                delta_x = np.random.uniform(self.medium_max, self.large_max)

            next_state = self.state + delta_x
            cost = self.operating_cost_rate * next_state
            self.state = next_state

        reward = -cost
        done = False
        terminated = False

        return next_state, reward, done, terminated
