import numpy as np
import time
import matplotlib.pyplot as plt
from src.environments.mountain_car import MountainCarEnv  # Ensure this import matches your environment setup

# Configuration for the algorithm and environment
config = {
    'discretization_factors': [100, 100],  # Position and velocity bins
    'gamma': 0.95,                       # Discount factor
    'theta': 1e-6,                       # Convergence threshold
    'max_iterations': 1000,              # Max value iterations
    'env_params': {                      # Mountain Car parameters
        'min_position': -1.2,
        'max_position': 0.6,
        'max_speed': 0.07,
        'goal_position': 0.5,
        'force': 0.001,
        'gravity': 0.0025
    }
}

def compute_next_state(pos, vel, action, params):
    """Model-based transition function for Mountain Car dynamics."""
    force = (action - 1) * params['force']
    vel_next = vel + force + np.cos(3 * pos) * (-params['gravity'])
    vel_next = np.clip(vel_next, -params['max_speed'], params['max_speed'])
    pos_next = pos + vel_next
    pos_next = np.clip(pos_next, params['min_position'], params['max_position'])
    # Reset velocity if hit left boundary
    if pos_next == params['min_position'] and vel_next < 0:
        vel_next = 0.0
    return pos_next, vel_next

def value_iteration(config):
    """Generic Value Iteration algorithm for discrete state-action spaces."""
    params = config['env_params']
    df = config['discretization_factors']
    
    # Calculate discrete state space dimensions
    num_pos = int((params['max_position'] - params['min_position']) * df[0]) + 1
    num_vel = int(2 * params['max_speed'] * df[1]) + 1
    print(f"Discretized state space: {num_pos}x{num_vel}")

    # Initialize value table
    V = np.zeros((num_pos, num_vel))
    delta_history = []
    policy = np.zeros((num_pos, num_vel), dtype=int)
    for it in range(config['max_iterations']):
        delta = 0
        new_V = V.copy()
        for i in range(num_pos):
            for j in range(num_vel):
                # Convert to continuous state
                pos = params['min_position'] + i * (params['max_position'] - params['min_position']) / (num_pos - 1)
                vel = -params['max_speed'] + j * (2 * params['max_speed']) / (num_vel - 1)
                
                if pos >= params['goal_position']:
                    new_V[i,j] = 0
                    continue
                
                max_value = -np.inf
                for action in [0, 1, 2]:
                    npos, nvel = compute_next_state(pos, vel, action, params)
                    # Check if next state is terminal
                    if npos >= params['goal_position']:
                        value = -1 + config['gamma'] * 0
                    else:
                        # Discretize next state
                        i_next = int(round((npos - params['min_position']) * df[0] / 
                                        (params['max_position'] - params['min_position'])))
                        i_next = np.clip(i_next, 0, num_pos-1)
                        j_next = int(round((nvel + params['max_speed']) * df[1] / 
                                        (2 * params['max_speed'])))
                        j_next = np.clip(j_next, 0, num_vel-1)
                        value = -1 + config['gamma'] * V[i_next, j_next]
                    
                    if value > max_value:
                        max_value = value
                
                delta = max(delta, abs(new_V[i,j] - max_value))
                new_V[i,j] = max_value
        
        V = new_V
        delta_history.append(delta)
        if delta < config['theta']:
            print(f"Converged after {it+1} iterations")
            break

    # Extract policy
    
    for i in range(num_pos):
        for j in range(num_vel):
            pos = params['min_position'] + i * (params['max_position'] - params['min_position']) / (num_pos - 1)
            vel = -params['max_speed'] + j * (2 * params['max_speed']) / (num_vel - 1)
            
            if pos >= params['goal_position']:
                continue
            
            best_value = -np.inf
            for action in [0, 1, 2]:
                npos, nvel = compute_next_state(pos, vel, action, params)
                if npos >= params['goal_position']:
                    value = -1 + config['gamma'] * 0
                else:
                    i_next = int(round((npos - params['min_position']) * df[0] / 
                                    (params['max_position'] - params['min_position'])))
                    i_next = np.clip(i_next, 0, num_pos-1)
                    j_next = int(round((nvel + params['max_speed']) * df[1] / 
                                    (2 * params['max_speed'])))
                    j_next = np.clip(j_next, 0, num_vel-1)
                    value = -1 + config['gamma'] * V[i_next, j_next]
                
                if value > best_value:
                    best_value = value
                    policy[i,j] = action

    return V, policy, delta_history

def test_policy(env, policy, config):
    """Test learned policy and gather statistics."""
    params = config['env_params']
    df = config['discretization_factors']
    total_rewards = []
    steps_list = []
    
    for _ in range(100):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            frame = env.render() 
            if frame is not None:
                plt.imshow(frame)
                plt.axis('off')
                plt.pause(0.00001)  # Small pause to allow rendering
            # Convert state to discrete
            print(steps)
            print("state:",state)
            if steps == 0:
                pos, vel = state[0]
            else:
                pos, vel = state
            print("pos:",pos)
            print("vel:",vel)
            i = int(round((pos - params['min_position']) * df[0] / 
                        (1)))
            print("i value:",i)
            i = np.clip(i, 0, df[0])
            j = int(round((vel + params['max_speed']) * df[1] / 
                        (1)))
            print("j value:",j)

            j = np.clip(j, 0, df[1]-1)
            print("i value:",i)
            print("j value:",j) 
            action = policy[i, j]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        steps_list.append(steps)
    
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average steps: {np.mean(steps_list):.2f}")
    return total_rewards, steps_list

if __name__ == "__main__":
    env = MountainCarEnv()
    
    start_time = time.time()
    V, policy, deltas = value_iteration(config)
    print("Value Function:",V)
    print("Policy:",policy)
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds")
    
    # Plot convergence
    plt.plot(deltas)
    plt.title("Value Iteration Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Max Delta")
    plt.show()
    
    # Test policy performance
    # rewards, steps = test_policy(env, policy, config)
    
    # # Visualize policy
    # plt.imshow(policy.T, cmap='viridis', origin='lower')
    # plt.xlabel("Position Bins")
    # plt.ylabel("Velocity Bins")
    # plt.title("Learned Policy")
    # plt.colorbar(label="Action")
    # plt.show()