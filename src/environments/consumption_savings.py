"""
Consumption-Savings Problem (Classic Economics MDP)

Agent has wealth and must decide how much to consume vs save.
- State: Current wealth
- Actions: Consumption level (0 to current wealth)
- Dynamics: Stochastic returns on savings
- Reward: Utility from consumption

This is a canonical problem in macroeconomics and lifecycle planning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConsumptionSavingsEnvironment(gym.Env):
    """
    Optimal consumption-savings problem.

    At each period:
    - Agent has wealth W
    - Chooses consumption C ∈ [0, W]
    - Savings S = W - C earn stochastic return
    - Next wealth: W' = S * (1 + r), where r ~ distribution
    - Utility: u(C) = C^(1-γ) / (1-γ)  (CRRA utility)

    Parameters:
    - max_wealth: Maximum wealth (state space cap)
    - mean_return: Expected return on savings (e.g., 0.05 = 5%)
    - return_std: Std dev of returns (risk)
    - gamma_utility: Risk aversion parameter (higher = more risk averse)
    - min_consumption: Minimum consumption (subsistence)
    """

    def __init__(
        self,
        max_wealth=1000.0,
        mean_return=0.05,
        return_std=0.10,
        gamma_utility=2.0,
        min_consumption=10.0,
        max_steps=100
    ):
        super().__init__()

        self.max_wealth = max_wealth
        self.mean_return = mean_return
        self.return_std = return_std
        self.gamma_utility = gamma_utility
        self.min_consumption = min_consumption
        self.max_steps = max_steps

        # State: current wealth
        self.observation_space = spaces.Box(
            low=0.0,
            high=max_wealth,
            shape=(1,),
            dtype=np.float32
        )

        # Action: consumption amount [0, 1] (scaled to actual wealth)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.state = None
        self._current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start with random initial wealth
        initial_wealth = self.np_random.uniform(100, 300)
        self.state = np.array([initial_wealth], dtype=np.float32)
        self._current_step = 0

        return self.state, {}

    def set_state(self, wealth):
        """Set wealth state directly (for value iteration)."""
        if isinstance(wealth, np.ndarray):
            wealth = float(wealth[0])
        self.state = np.array([float(wealth)], dtype=np.float32)
        self._current_step = 0

    def step(self, action):
        """
        Execute one time step.

        Args:
            action: Consumption rate [0, 1] as fraction of wealth

        Returns:
            next_state, reward, terminated, truncated, info
        """
        wealth = float(self.state[0])
        consumption_rate = float(action[0])

        # Ensure valid consumption
        consumption_rate = np.clip(consumption_rate, 0.0, 1.0)

        # Actual consumption (with minimum)
        consumption = max(
            wealth * consumption_rate,
            min(self.min_consumption, wealth)
        )
        consumption = min(consumption, wealth)

        # Utility from consumption (CRRA)
        if self.gamma_utility == 1.0:
            utility = np.log(consumption)
        else:
            utility = (consumption ** (1 - self.gamma_utility)) / (1 - self.gamma_utility)

        # Savings
        savings = wealth - consumption

        # Stochastic return on savings
        return_rate = self.np_random.normal(self.mean_return, self.return_std)

        # Next period wealth
        next_wealth = savings * (1 + return_rate)
        next_wealth = np.clip(next_wealth, 0, self.max_wealth)

        self.state = np.array([next_wealth], dtype=np.float32)
        self._current_step += 1

        # Episode ends if wealth depleted or max steps reached
        terminated = next_wealth < self.min_consumption
        truncated = self._current_step >= self.max_steps

        info = {
            'wealth': wealth,
            'consumption': consumption,
            'savings': savings,
            'return': return_rate,
            'next_wealth': next_wealth,
            'utility': utility
        }

        return self.state, utility, terminated, truncated, info

    def render(self):
        wealth = float(self.state[0])
        print(f"Wealth: ${wealth:.2f}")


class InventoryManagementEnvironment(gym.Env):
    """
    Inventory management problem (operations research / economics).

    At each period:
    - State: Current inventory level
    - Action: Order quantity
    - Dynamics: Stochastic demand
    - Costs: Holding cost + shortage cost + ordering cost
    - Reward: Negative costs

    Classic newsvendor / economic order quantity problem.
    """

    def __init__(
        self,
        max_inventory=500,
        holding_cost=1.0,
        shortage_cost=10.0,
        order_cost=50.0,
        mean_demand=50.0,
        demand_std=15.0,
        max_steps=100
    ):
        super().__init__()

        self.max_inventory = max_inventory
        self.holding_cost = holding_cost  # Cost per unit per period
        self.shortage_cost = shortage_cost  # Cost per unit short
        self.order_cost = order_cost  # Fixed cost per order
        self.mean_demand = mean_demand
        self.demand_std = demand_std
        self.max_steps = max_steps

        # State: current inventory
        self.observation_space = spaces.Box(
            low=0.0,
            high=max_inventory,
            shape=(1,),
            dtype=np.float32
        )

        # Action: order quantity [0, max_inventory]
        self.action_space = spaces.Box(
            low=0.0,
            high=max_inventory,
            shape=(1,),
            dtype=np.float32
        )

        self.state = None
        self._current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start with random initial inventory
        initial_inventory = self.np_random.uniform(50, 150)
        self.state = np.array([initial_inventory], dtype=np.float32)
        self._current_step = 0

        return self.state, {}

    def set_state(self, inventory):
        """Set inventory state directly."""
        if isinstance(inventory, np.ndarray):
            inventory = float(inventory[0])
        self.state = np.array([float(inventory)], dtype=np.float32)
        self._current_step = 0

    def step(self, action):
        """Execute one time step."""
        inventory = float(self.state[0])
        order_qty = float(action[0])
        order_qty = np.clip(order_qty, 0, self.max_inventory - inventory)

        # Ordering cost
        if order_qty > 0:
            ordering_cost = self.order_cost
        else:
            ordering_cost = 0

        # Inventory after order
        inventory_after_order = inventory + order_qty

        # Stochastic demand
        demand = max(0, self.np_random.normal(self.mean_demand, self.demand_std))

        # Inventory after demand
        inventory_after_demand = inventory_after_order - demand

        # Costs
        if inventory_after_demand >= 0:
            # No shortage
            holding = inventory_after_demand * self.holding_cost
            shortage = 0
            next_inventory = inventory_after_demand
        else:
            # Shortage
            holding = 0
            shortage = abs(inventory_after_demand) * self.shortage_cost
            next_inventory = 0

        next_inventory = min(next_inventory, self.max_inventory)

        total_cost = ordering_cost + holding + shortage
        reward = -total_cost  # Negative cost as reward

        self.state = np.array([next_inventory], dtype=np.float32)
        self._current_step += 1

        terminated = False
        truncated = self._current_step >= self.max_steps

        info = {
            'inventory': inventory,
            'order_qty': order_qty,
            'demand': demand,
            'ordering_cost': ordering_cost,
            'holding_cost': holding,
            'shortage_cost': shortage,
            'total_cost': total_cost,
            'next_inventory': next_inventory
        }

        return self.state, reward, terminated, truncated, info

    def render(self):
        inventory = float(self.state[0])
        print(f"Inventory: {inventory:.0f} units")
