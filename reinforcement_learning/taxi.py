import numpy as np  # For numerical operations and arrays
import gymnasium as gym  # OpenAI Gym for RL environments
import matplotlib.pyplot as plt  # For visualization

env = gym.make("Taxi-v3", render_mode="rgb_array")  # Create Taxi environment with RGB rendering

def q_learning(Q, alpha, gamma, epsilon, min_epsilon, decay, num_episodes):  # Main Q-learning training loop
    print("Training...")
    actions = np.arange(0, env.action_space.n)  # Array of all possible actions
    i = 0
    success = 0  # Counter for successful taxi rides
    for i in range(num_episodes + 1):  # Training loop for specified episodes
        if i % 1000 == 0:  # Print progress every 1000 episodes
            print(f"Episode {i} / {num_episodes} | Epsilon: {epsilon:.5f} | Success (in last 1000 eps) = {success}")    # printing progress
            success = 0  # Reset success counter
        state, _ = env.reset()  # Reset environment to initial state
        completed = 0  # Episode completion flag
        while not completed:  # Continue until episode ends
            action = take_action(actions, Q, epsilon, state)    # take action
            next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status\
            completed = terminated or truncated  # Check if episode is done
            if reward == 20:  # Successful passenger delivery reward
                success += 1  # Increment success counter
            # bellman equation
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])  # Update Q-value using Bellman equation
            state = next_state  # Move to next state
            epsilon = epsilon_decay(epsilon, min_epsilon, decay)    # decay epsilon after every Q update (every step)
    return Q  # Return trained Q-table

def epsilon_decay(epsilon, min_epsilon, decay):  # Decay exploration rate over time
    if epsilon > min_epsilon:  # Only decay if above minimum
        epsilon = epsilon * decay  # Apply decay factor
    return epsilon  # Return updated epsilon

# function to take action using epsilon-greedy mechanism
def take_action(actions, Q, epsilon, state):  # Choose action using epsilon-greedy policy
    if np.random.rand() > epsilon:  # Exploit: choose best known action
        action = np.argmax(Q[state])  # Select action with highest Q-value
    else:  # Explore: choose random action
        action = np.random.randint(0, len(actions))  # Random action selection
    return action  # Return chosen action

# evaluating final learnt Q for 1 episode
def evaluate(Q):  # Test trained agent performance
    print("Evaluating...")
    #plt.ion()
    state, _ = env.reset()  # Reset environment for evaluation
    total_reward = 0  # Track cumulative reward
    completed = 0  # Episode completion flag
    while not completed:  # Continue until episode ends
        action = np.argmax(Q[state])    # take action
        next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status
        total_reward += reward  # Accumulate rewards
        completed = terminated or truncated  # Check completion status
        state = next_state  # Move to next state
        frame = env.render()    # storing the frame
        plt.imshow(frame)  # Display current frame
        plt.axis("off")  # Hide plot axes
        plt.pause(1)    # pausing for a small time to show continuous frames
    return total_reward  # Return total episode reward

Q = np.zeros((500, 6))  # Initialize Q-table with zeros (500 states, 6 actions)
Q = q_learning(Q, 0.1, 0.9999, 1.0, 0.05, 0.9995, 10000)  # Train agent with Q-learning

print(f"Final Episode Reward: {evaluate(Q)}")  # Evaluate and display final performance