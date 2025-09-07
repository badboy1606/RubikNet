import gymnasium as gym  # OpenAI Gym for RL environments
import numpy as np  # For numerical operations and arrays
import math  # For mathematical operations

env= gym.make("CartPole-v1")  # Create CartPole environment
q_table=np.zeros((10,10,10,10,2))  # Initialize Q-table for discretized state space

lower_bounds = [-4.8, -3.0, -0.418, -math.radians(50)]  # Minimum values for each state variable
upper_bounds = [4.8, 3.0, 0.418, math.radians(50)]  # Maximum values for each state variable

def discritsize(obs):  # Convert continuous observations to discrete state indices
    cart_pos, cart_vel, pole_pos, pol_vel=obs  # Unpack observation components
    cart_pos_scaled=(cart_pos+4.8)/9.6  # Scale cart position to [0,1]
    cart_vel_scaled=(cart_vel+3)/6  # Scale cart velocity to [0,1]
    pole_pos_scales=(pole_pos+0.418)/0.836  # Scale pole angle to [0,1]
    pol_vel_scaled=(pol_vel+math.radians(50))/(2*math.radians(50))  # Scale pole angular velocity to [0,1]

    cart_pos_want = int(np.clip(cart_pos_scaled * 10, 0, 9))  # Discretize cart position to bins 0-9
    cart_vel_want = int(np.clip(cart_vel_scaled * 10, 0, 9))  # Discretize cart velocity to bins 0-9
    pole_pos_want = int(np.clip(pole_pos_scales * 10, 0, 9))  # Discretize pole angle to bins 0-9
    pole_vel_want = int(np.clip(pol_vel_scaled * 10, 0, 9))  # Discretize pole angular velocity to bins 0-9

    return cart_pos_want, cart_vel_want, pole_pos_want, pole_vel_want  # Return discretized state

def calculate_reward(action, current_obs, next_obs):  # Custom reward function for better learning
    reward = 0  # Initialize reward

    # 1. Reward if pole is more vertical
    if abs(next_obs[2]) < 0.05:  # Pole very upright
        reward += 5  # High reward for vertical pole
    elif abs(next_obs[2]) < 0.1:  # Pole somewhat upright
        reward += 1  # Small reward
    else:  # Pole tilted
        reward -= 2  # Penalty for tilted pole

    # 2. Reward if cart is near center
    if abs(next_obs[0]) < 0.25:  # Cart near center
        reward += 2  # Reward for staying centered
    elif abs(next_obs[0]) < 0.5:  # Cart moderately centered
        reward += 0  # Neutral reward
    else:  # Cart far from center
        reward -= 1  # Small penalty

    # 3. Small penalty for high angular velocity (pole speed)
    if abs(next_obs[3]) > 0.75:  # High pole angular velocity
        reward -= 2  # Penalty for fast pole movement

    return reward  # Return calculated reward

    
    

alpha = 0.1             # Learning rate for Q-learning updates
gamma = 0.99            # Discount factor for future rewards
epsilon = 1.0           # Initial exploration rate
min_epsilon = 0.05  # Minimum exploration rate
epsilon_decay = 0.98  # Decay rate for exploration
episodes = 10000  # Total training episodes
max_steps = 500  # Maximum steps per episode

for episode in range(episodes):  # Main training loop
    obs, _ = env.reset()  # Reset environment for new episode
    state = discritsize(obs)  # Convert to discrete state
    total_reward = 0  # Track episode reward

    for step in range(max_steps):  # Episode step loop
        # Choose action: epsilon-greedy
        if np.random.rand() < epsilon:  # Exploration: random action
            action = env.action_space.sample()  # Sample random action
        else:  # Exploitation: best known action
            action = np.argmax(q_table[state])  # Choose action with highest Q-value

        next_obs, _, done, truncated, _ = env.step(action)  # Take action in environment
        reward=calculate_reward(action,obs,next_obs)  # Calculate custom reward
        next_state = discritsize(next_obs)  # Discretize next observation

        # Q-learning update 
        best_next_action = np.max(q_table[next_state])  # Best Q-value for next state
        q_table[state][action] += alpha * (reward + gamma * best_next_action - q_table[state][action])  # Update Q-value using Bellman equation

        state = next_state  # Update current state
        total_reward += reward  # Add to episode reward

        if done or truncated:  # Check if episode ended
            break  # Exit episode loop

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Reduce exploration over time

print("Training complete ")  # Training finished message

env=gym.make("CartPole-v1", render_mode="human")  # Create environment with visual rendering
obs, _ = env.reset()  # Reset environment for testing
state = discritsize(obs)  # Get initial discrete state
done=False  # Episode completion flag

i=1  # Step counter (unused)
while not done:  # Test loop until episode ends
    env.render()  # Display environment
    action = np.argmax(q_table[state])  # Choose best action (greedy policy)
    obs, reward, done, truncated, _ = env.step(action)  # Take action
    state = discritsize(obs)  # Update discrete state
    
    

env.close()  # Close environment