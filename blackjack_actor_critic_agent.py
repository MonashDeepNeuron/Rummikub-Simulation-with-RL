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

"""# Agent - Q-Learing with Actor-Critic Method

## 🎭 Actor–Critic: The Concept

Think of the agent as having **two brains** working together:

1. **Actor (the decision maker)**

   * Learns the **policy** $\pi_\theta(a|s)$ — i.e., “what action should I take in this state?”
   * It’s usually a neural network that outputs probabilities over actions (via softmax).

2. **Critic (the evaluator)**

   * Learns the **value function** $V_w(s)$ — i.e., “how good is this state?”
   * Helps the Actor improve by providing a baseline: *Was that action better than expected?*

---

### 🔑 The flow of learning

1. Actor picks an action $a$ in state $s$.
2. Environment gives reward $r$ and next state $s'$.
3. Critic estimates the **TD error**:

   $$
   \delta = r + \gamma V_w(s') - V_w(s)
   $$

   This tells us whether things went better or worse than expected.
4. Critic updates its value network with this TD error.
5. Actor updates its policy:

   * If $\delta > 0$, the chosen action is reinforced (more likely next time).
   * If $\delta < 0$, the chosen action is weakened.

---

### Why it’s powerful

* **Actor** gives us **stochastic exploration** for free (sampling actions).
* **Critic** stabilizes training (reduces variance compared to vanilla Policy Gradient).
* Together, they balance **exploration** and **evaluation**.

---

### 📌 Key differences from Q-Learning / DQN

| Method       | Learns what?                                          | Exploration style                  |
| ------------ | ----------------------------------------------------- | ---------------------------------- |
| Q-Learning   | Action-value function $Q(s,a)$                        | ε-greedy                           |
| DQN          | Approximates $Q(s,a)$ with NN                         | ε-greedy                           |
| Actor-Critic | Actor learns policy $\pi$, Critic learns value $V(s)$ | Sampling from Actor’s distribution |

---

## Resources:
https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/ \
\
https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14 \
\
http://www.incompleteideas.net/book/RLbook2020.pdf page 353

#### Try with geeksforgeeks method
"""


# Your code - Charis

# Your code - Charis

class ActorCriticAgent:
    def __init__(self, env, episodes=1000, alpha=0.001, gamma=0.99):
        """
        env: the environment
        episodes: number of training episodes
        alpha: learning rate
        gamma: discount factor
        """
        self.env = env
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma

        obs_size = 3 # (player_sum, deler_card, usable_ace)
        '''
        state[0] = player_sum = the sum of your cards
        state[1] = the dealers' upcards (1-10)
        state[2] : Boolean = if I have an Ace
        '''
        action_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-5)

        self.critic = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # outputs scalar value estimate
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha)

        # For reward normalization (running mean/std)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero

    # =====================
    # 2. Reward Normalization
    # - Running mean/std trick
    # =====================
    def normalize_reward(self, reward):
        self.reward_count += 1
        self.reward_mean += (reward - self.reward_mean) / self.reward_count
        self.reward_var += (reward - self.reward_mean) * (reward - self.reward_mean)
        reward_std = max((self.reward_var / self.reward_count) ** 0.5, 1e-6)
        return (reward - self.reward_mean) / reward_std


    def action_space(self, state):
        '''
        Decide which action to take.
        - Actor outputs probabilities of each action.
        - Sample an action from this probability distribution.
        Returns: chosen action
        '''
        # your code
        state_t = torch.tensor([state[0], state[1], int(state[2])], dtype = torch.float32) # the int() is to transform a Boolean value into an integer
        state_t = state_t.unsqueeze(0)   # add batch dimension so nn works properly
        probs = self.actor(state_t).squeeze(0)
        dist = torch.distributions.Categorical(probs) # exploration comes naturally from sampling actions from the Actor’s probability distribution.
        action = dist.sample()

        # we need log probability of the chosen action for the policy gradient update
        return action.item(), dist.log_prob(action)


    def update(self, state, action, reward, next_state, done, log_prob):

      '''
      Core Actor–Critic update step:
      1. Critic computes TD error δ = r + γ V(s') - V(s)
      2. Critic updates its value network
      3. Actor updates its policy parameters using δ as feedback
      '''
      # your code

      # critic
      state_t = torch.tensor([state[0], state[1], int(state[2])], dtype=torch.float32)
      next_state_t = torch.tensor([next_state[0], next_state[1], int(next_state[2])], dtype=torch.float32)

      # Apply reward normalization here
      reward = self.normalize_reward(reward)

      value = self.critic(state_t)
      with torch.no_grad():
          next_value = self.critic(next_state_t)
          target = reward if done else reward + self.gamma * next_value.item()
      target = torch.tensor([target], dtype=torch.float32)

      TD_error = target - value

      # critic update
      # =====================
      # 3. Stabilizing the Critic
      # - Gradient clipping
      # - TD error squared loss
      # =====================
      critic_loss = TD_error.pow(2)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
      self.critic_optimizer.step()

      # Actor update with entropy regularization
      probs = self.actor(state_t.unsqueeze(0)).squeeze(0)
      dist = torch.distributions.Categorical(probs)
      entropy = dist.entropy()

      actor_loss = -log_prob * TD_error.detach() - 0.01 * entropy
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
      self.actor_optimizer.step()


    def train(self, log_interval=100):
        '''
        Main training loop:
        - For each episode:
            1. Reset environment
            2. Loop until terminal state:
                - Select action using Actor
                - Step environment
                - Update Actor & Critic
        '''
        # your code

        episode_rewards = []
        win_history = []   # store 1 if win, 0 otherwise

        for i in range(self.episodes):
          state, info = self.env.reset()
          done = False
          total_reward = 0
          while not done:
            action, log_prob = self.action_space(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.update(state, action, reward, next_state, done, log_prob)
            state = next_state
            total_reward += reward

          episode_rewards.append(total_reward)

          if (i+1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"[{i+1}/{self.episodes}] Average reward (last {log_interval} eps): {avg_reward:.2f}")


        return episode_rewards

    def play(self, num_games=10):
        '''
        Evaluate the trained policy:
        - Always pick action with highest probability
        - Track rewards and outcomes
        '''
        # your code
        results = []

        for _ in range(num_games):

          state, info = self.env.reset()
          done = False
          total_reward = 0

          while not done:
            # Convert state to tensor
            state_t = torch.tensor([state[0], state[1], int(state[2])], dtype=torch.float32)
            state_t = state_t.unsqueeze(0)  # Add batch dimension (so it's 1 sample of 3 features)

            # Actor network outputs probabilities
            probs = self.actor(state_t).squeeze(0) # Remove batch dimension

            # Greedy choice = argmax, not sampling
            action = torch.argmax(probs).item()

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

          results.append(total_reward)

        # after all games, summarize
        print(f'Average reward: {np.mean(results):.4f}')
        print(f"Wins: {results.count(1)}, Losses: {results.count(-1)}, Draws: {results.count(0)}")

        return results

    def save(self, filepath="actor_critic.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episodes": self.episodes,
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="actor_critic.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Model loaded from {filepath}")

    def plot_rewards(self, rewards, window=100):
        """
        Plot the reward obtained in each episode and the moving average over a given window.
        """

        plt.figure(figsize=(10, 6))


        # Raw rewards per episode
        plt.plot(rewards, label="Episode reward", color = "#13563B")


        # Moving average of rewards
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
            plt.plot(range(window-1, len(rewards)), moving_avg,
            label=f"Moving Avg (last {window} eps)", color="#E4A700", linewidth=1)

        # Add horizontal line at y=0
        plt.axhline(y=0, color='#E9D8A6', linestyle='--', linewidth=0.8)


        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Actor-Critic Training Rewards")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()







if __name__ == "__main__":
    episodes = 250000
    alpha = 1e-4
    gamma = 0.99

    env = gym.make("Blackjack-v1", natural = True, sab=False) # sab=False = default rules

    agent = ActorCriticAgent(env, episodes=episodes, alpha=alpha, gamma=gamma)

    # Train the agent
    reward = agent.train()
    print("Training finished.")

    # Save trained model
    agent.save(f"blackjack_actor_critic_episode{episodes}_alpha{alpha}_gamma{gamma}.pth")

    # Evaluate loaded agent
    agent.play(num_games=100)


if __name__ == "__main__":

    env = gym.make("Blackjack-v1", natural = True, sab = False)

    # Later, reload
    new_agent = ActorCriticAgent(env)
    new_agent.load("blackjack_actor_critic1.pth")

    # Evaluate loaded agent
    new_agent.play(num_games=1000)


