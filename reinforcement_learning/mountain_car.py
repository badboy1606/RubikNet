import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0', render_mode="rgb_array")

def q_learning(alpha, gamma, epsilon, min_epsilon, decay, num_episodes):
    print("Training...")
    states = (env.observation_space.high - env.observation_space.low) * np.array([10,100])    # discretize states
    states = np.round(states, 0).astype(int) + 1
    actions = np.arange(0, env.action_space.n)
    Q = np.random.uniform(low = -1, high = 1, size = (states[0], states[1], len(actions)))    # initialize q
    i = 0
    won = 0
    for i in range(num_episodes + 1):
        if i % 1000 == 0:
            print(f"Episode {i} / {num_episodes} | Epsilon: {epsilon:.5f} | Won (in last 1000 eps) = {won}")    # printing progress
            won = 0
        state, _ = env.reset()
        state = discretize(state)
        completed = 0
        while not completed:
            action = take_action(actions, Q, epsilon, state)    # take action
            next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status
            if next_state[0] >= 0.5:
                won += 1
            next_state = discretize(next_state)
            completed = terminated or truncated
            # bellman equation
            Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
            state = next_state
            epsilon = epsilon_decay(epsilon, min_epsilon, decay)    # decay epsilon after every Q update (every step)

    return Q

def discretize(state):
    state = (state - env.observation_space.low) * np.array([10, 100])
    state = np.round(state, 0).astype(int)
    return state

def epsilon_decay(epsilon, min_epsilon, decay):
    if epsilon > min_epsilon:
        epsilon = epsilon * decay
    return epsilon

# function to take action using epsilon-greedy mechanism
def take_action(actions, Q, epsilon, state):
    if np.random.rand() > epsilon:
        action = np.argmax(Q[state[0], state[1]])
    else:
        action = np.random.randint(0, len(actions))
    return action

# evaluating final learnt Q for 1 episode
def evaluate(Q):
    state, _ = env.reset()
    state = discretize(state)
    total_reward = 0
    completed = 0
    while not completed:
        action = np.argmax(Q[state[0], state[1]])    # take action
        next_state, reward, terminated, truncated, _ = env.step(action)    # generate next state, reward and status
        state = discretize(next_state)
        total_reward += reward
        completed = terminated or truncated
        frame = env.render()    # storing the frame
        plt.imshow(frame)
        plt.axis("off")
        plt.pause(0.01)    # pausing for a small time to show continuous frames
        plt.clf()    # clear last frame
    return total_reward

Q = q_learning(0.2, 0.99, 1.0, 0.01, 0.9995, 25000)    # alpha, gamma, initial epsilon, minimum epsilon, decay rate for epsilon, number of episodes
print(f"Total Reward for one episode after training: {evaluate(Q)}")