# Import necessary libraries: NumPy, Gym, and Matplotlib.
import time
import numpy as np
from src.environments.mountain_car import MountainCarEnv
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment


# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    print(num_states)
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
     # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        print(i)
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        #print(state[0])
        # Discretize state
        #print(env.observation_space.low)
        state_adj = (np.array(state[0] - env.observation_space.low))*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
            # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 10 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
     #Finally, the function returns the list of average rewards per episode.
   
    return ave_reward_list,Q

def test_policy(env, Q, num_episodes=5, render=True):
    """
    Test the learned Q-function on the environment.
    
    Args:
        env: Mountain Car environment
        Q: Learned Q-function table
        num_episodes: Number of episodes to test
        render: Whether to render the environment
    
    Returns:
        rewards: List of total rewards for each episode
        steps: List of steps taken for each episode
    """
    rewards = []
    steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            if render:
                frame = env.render()
                if frame is not None:
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.pause(0.00001) 
          # Add small delay to make rendering visible
            print(state)
            # Discretize state the same way as in training
            if step == 0:
                state_adj = (state[0] - env.observation_space.low)*np.array([10, 100])
                state_adj = np.round(state_adj, 0).astype(int)
            else:
                state_adj = (state - env.observation_space.low)*np.array([10, 100])
                state_adj = np.round(state_adj, 0).astype(int)
            print(state_adj)
            # Select action greedily
            action = np.argmax(Q[state_adj[0], state_adj[1]])
            
            # Take action
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step}")
                if state[0] >= 0.5:
                    print("Success! Reached the goal!")
                break
        
        rewards.append(total_reward)
        steps.append(step)
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards during Testing')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode during Testing')
    
    plt.tight_layout()
    plt.show()
    
    return rewards, steps



if __name__ =='__main__':
    # Run Q-learning algorithm
    print("Running Q-Learning algorithm")
    env = MountainCarEnv()
    env.reset()
    start_time = time.time()
    rewards,Q = QLearning(env, 0.2, 0.9, 0.8, 0, 50)
    end_time = time.time()
    #test_policy(env, Q, num_episodes=5, render=False)
    print(rewards)
    # Plot Rewards
    plt.plot(1*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.show()
    print(Q)
    print("Time taken to compute Q-learning algorithm:", end_time-start_time)

