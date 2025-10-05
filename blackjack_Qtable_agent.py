from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium.utils.env_checker import check_env
from collections import defaultdict
import numpy as np
from tqdm import tqdm  # Progress bar
from typing import Optional
from gymnasium import spaces
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch


class BlackjackAgent_QLearning_Qtable:
    def __init__(self, env, episodes=1000, eps=0.1, alpha=0.1, gamma=0.9):
        """
        Initialize the agent.
        - env: the Blackjack environment
        - episodes: number of training episodes
        - eps: epsilon for epsilon-greedy (exploration rate)
        - alpha: learning rate (step size for Q update)
        - gamma: discount factor
        """
        self.env = env
        self.episodes = episodes
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = spaces.Discrete(2)
        self.action = ["stick", "hit"]

        # Storage for Q-values (for Q-learning), can later swap to NN
        self.Q = {}

    def get_action(self, state):
        """
        Decide which action to take given a state.
        - With probability eps: pick random action (exploration)
        - Otherwise: pick best action based on Q-values (exploitation)
        Returns: action
        """
        if state not in self.Q:
          self.Q[state] = np.zeros(self.action_space.n)

        if np.random.random() < self.eps:
          action = np.random.choice(self.action_space.n)
        else:
          action = np.argmax(self.Q[state])

        return action

    def update_q(self, state, action, reward, next_state, done):
        """
        Perform the Q-learning update rule:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
        """
        # Initialize unseen states
        if state not in self.Q:
          self.Q[state] = np.zeros(self.action_space.n)

        if next_state not in self.Q:
          self.Q[next_state] = np.zeros(self.action_space.n)

        target = reward # Start with immediate reward

        # Add future value if game not over
        '''
        if the episode is not finished, then we also care about the future reward we might get
        `np.max(self.Q[next_state])` = best possible value I can get in the next state
        gamma helps reduce importance of far-away rewards
        add the future rewards to the immediate reward

        `self.Q[state][action]` = Q(s,a)
        This compare the current guess with the new target we have just computed
        `(target - Q(s,a))` is the error in the estimate
        multiply by learning rate `alpha` to take a small step toward fixing it
        '''
        if not done:
          target += self.gamma * np.max(self.Q[next_state])

        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def train(self):
        """
        Main training loop:
        - For each episode:
            1. Reset environment
            2. Loop through steps until terminal state:
                - Choose action
                - Take action in env
                - Get reward + next state
                - Update Q-values
        - Track performance (e.g., average reward)

        updates Q-values, uses epsilon-greedy
        """

        episode_rewards = []

        for i in range(self.episodes):
          state, info = self.env.reset()
          done = False
          total_reward = 0
          while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.update_q(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

          episode_rewards.append(total_reward)

        return episode_rewards

    def play(self, num_games=10):
        """
        Run the agent in evaluation mode (greedy policy only).
        Print/return results (e.g., wins, losses, draws).

        no updates, greedy only.
        """
        results = []

        for _ in range(num_games):

          state, info = self.env.reset()
          done = False
          total_reward = 0

          while not done:
            if state in self.Q:
              action = np.argmax(self.Q[state])
            else:
              action = self.env.action_space.sample()

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

          results.append(total_reward)

        # after all games, summarize
        print(f'Average reward: {np.mean(results):.4f}')
        print(f"Wins: {results.count(1)}, Losses: {results.count(-1)}, Draws: {results.count(0)}")

        return results
    
    def save(self, filepath="qtable_agent.pth"):
        """
        Save the learned Q-table and parameters to a file using torch.
        """
        checkpoint = {
            "Q": self.Q,
            "episodes": self.episodes,
            "eps": self.eps,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        torch.save(checkpoint, filepath, pickle_protocol=4)
        print(f"Agent saved to {filepath}")

    def load(self, filepath="qtable_agent.pth"):
        """
        Load the Q-table and parameters from a file using torch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.Q = checkpoint["Q"]
        self.episodes = checkpoint["episodes"]
        self.eps = checkpoint["eps"]
        self.alpha = checkpoint["alpha"]
        self.gamma = checkpoint["gamma"]
        print(f"Agent loaded from {filepath}")




if __name__ == "__main__":

    episodes = 5000
    eps = 0.1
    alpha = 1e-3
    gamma = 0.99
    # Create Blackjack environment
    env = gym.make("Blackjack-v1", natural = True, sab = False)  # sab=False = default rules

    # Create agent
    agent = BlackjackAgent_QLearning_Qtable(env, episodes=episodes, eps=eps, alpha=alpha, gamma=gamma)

    # Train the agent
    rewards = agent.train()
    print("Training finished.")
    agent.play(num_games=100)

    # Save the agent
    agent.save(f"blackjack_qtable_agent_episode{episodes}_eps{eps}_alpha{alpha}_gamma{gamma}.pth")


if __name__ == "__main__":

    episodes = 5000
    eps = 0.1
    alpha = 1e-3
    gamma = 0.99
    # Create Blackjack environment
    env = gym.make("Blackjack-v1", natural = True, sab = False)  # sab=False = default rules

    # Create agent
    agent = BlackjackAgent_QLearning_Qtable(env, episodes=episodes, eps=eps, alpha=alpha, gamma=gamma) 
    # Load the agent
    agent.load(f"blackjack_qtable_agent_episode{episodes}_eps{eps}_alpha{alpha}_gamma{gamma}.pth")

    # Evaluate performance after loading
    agent.play(num_games=100)

    # For visualization of training progress
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()




