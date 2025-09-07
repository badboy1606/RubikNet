import numpy as np
import gymnasium as gym
from collections import defaultdict

# Reinforcement Learning - Monte Carlo Method for Blackjack

env = gym.make("Blackjack-v1", sab=True)  # create Blackjack environment

def monte_carlo(n_episodes, epsilon, alpha, gamma, Q):
    for i in range(n_episodes):
        episode = generate_ep(Q, epsilon)    # generate one episode (state, action, reward)
        states, actions, rewards = zip(*episode)  # unpack episode
        epsilon *= 0.99995    # decay epsilon after every episode
        n = 0
        for state in states:
            discounted_return = 0
            k = np.arange(len(rewards[n:]))  # indices for discounting
            for j in range(len(rewards[n:])):
                discounted_return += rewards[n + j] * (gamma**k[j])    # compute discounted return
            Q[state][actions[n]] += alpha * (discounted_return - Q[state][actions[n]])    # update Q-value
            n += 1
    return Q

def generate_ep(Q, epsilon):
    state, _ = env.reset()  # reset environment to start new episode
    episode = []

    completed = 0
    while not completed:
        action = generate_action(Q, epsilon, state)  # choose action (epsilon-greedy)
        next_state, reward, terminated, truncated, _ = env.step(action)  # step in env
        completed = terminated or truncated  # check if episode finished
        episode.append((state, action, reward))  # store transition
        state = next_state
    
    return episode  # return full episode

def generate_action(Q, epsilon, state):
    action_space = [0, 1]  # 0=stick, 1=hit
    probabilities = [1 - epsilon / 2, epsilon / 2] if np.argmax(Q[state]) == 0 else [epsilon / 2, 1 - epsilon / 2]
    action = np.random.choice(action_space, p = probabilities)  # epsilon-greedy action selection
    return action

def get_win_rate(Q, n_episodes):
    # win rate = number of episodes with total_reward > 0 divided by total episodes
    episodes_won = 0
    for _ in range(n_episodes):
        state, _ = env.reset()  # reset at start of episode
        total_reward = 0
        completed = 0
        while not completed:
            action = np.argmax(Q[state]) if state in Q else np.random.choice([0, 1])  # greedy action
            next_state, reward, terminated, truncated, _ = env.step(action)
            completed = terminated or truncated
            total_reward += reward
            state = next_state
        if total_reward > 0:
            episodes_won += 1
    return (episodes_won / n_episodes) * 100  # percentage of wins

Q = defaultdict(lambda: np.zeros(2))  # initialize Q with two actions per state
print("Training...")
Q = monte_carlo(100000, 1.0, 0.4, 0.99999, Q)    # run Monte Carlo learning

print("Calculating...")
print(f"Win Rate: {get_win_rate(Q, 10000)}")    # evaluate policy performance
# Expected output: Win Rate: ~42-43% after training