from typing import Optional
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Reward structure: +100 for reaching target, -1 per step to encourage efficiency
        reward = 100 if terminated else -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the current grid state to console."""
        print("\n" + "="*30)
        print("Grid World Game - Current State")
        print("="*30)
        
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if np.array_equal([i, j], self._agent_location):
                    row += "A "  # Agent
                elif np.array_equal([i, j], self._target_location):
                    row += "T "  # Target
                else:
                    row += ". "  # Empty space
            print(row)
        print("="*30)
        print("Legend: A=Agent, T=Target, .=Empty")
        print(f"Distance to target: {self._get_info()['distance']:.1f}")
        print()


# Q-Learning Agent for solving the grid world
class GridWorldQLearningAgent:
    """Simple Q-learning agent for grid world navigation."""
    
    def __init__(self, state_size: int, action_size: int = 4, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.9999, 
                 epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping state -> [action_values]
        self.q_table = {}
    
    def _state_to_key(self, state_dict):
        """Convert state dictionary to string key for Q-table."""
        agent = state_dict['agent']
        target = state_dict['target']
        return f"a{agent[0]}_{agent[1]}_t{target[0]}_{target[1]}"
    
    def get_action(self, state, grid_size):
        """Choose action using epsilon-greedy policy."""
        state_key = self._state_to_key(state)
        
        # Initialize state in Q-table if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-0.1, 0.1, self.action_size)
        
        # Get valid actions (don't move outside boundaries)
        valid_actions = []
        current_pos = state['agent']
        
        # Check each action: 0=right, 1=up, 2=left, 3=down
        if current_pos[0] < grid_size - 1:  # Can move right
            valid_actions.append(0)
        if current_pos[1] < grid_size - 1:  # Can move up
            valid_actions.append(1)
        if current_pos[0] > 0:  # Can move left
            valid_actions.append(2)
        if current_pos[1] > 0:  # Can move down
            valid_actions.append(3)
        
        if not valid_actions:  # Should not happen
            valid_actions = list(range(self.action_size))
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            # Choose best valid action
            valid_q_values = [self.q_table[state_key][action] for action in valid_actions]
            best_valid_action_idx = np.argmax(valid_q_values)
            return valid_actions[best_valid_action_idx]
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize states in Q-table if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-0.1, 0.1, self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.random.uniform(-0.1, 0.1, self.action_size)
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] += self.learning_rate * (
            target - self.q_table[state_key][action]
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_grid_agent(episodes: int = 1000, grid_size: int = 5):
    """Train the Q-learning agent on the grid world."""
    env = GridWorldEnv(size=grid_size)
    agent = GridWorldQLearningAgent(state_size=grid_size*grid_size)
    
    rewards_history = []
    recent_episodes = []  # Store last 3 episodes
    
    print(f"Training grid world agent for {episodes} episodes...")
    print("Grid layout:")
    env.reset()
    env.render()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = grid_size * grid_size * 2  # Prevent infinite loops
        
        episode_record = {
            'episode': episode + 1,
            'steps': [],
            'total_reward': 0,
            'success': False
        }
        
        while steps < max_steps:
            action = agent.get_action(state, grid_size)
            next_state, reward, done, _, _ = env.step(action)
            
            # Record step for episode history
            episode_record['steps'].append({
                'agent_pos': state['agent'].copy(),
                'action': action,
                'reward': reward
            })
            
            agent.update_q_table(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                episode_record['success'] = True
                break
        
        episode_record['total_reward'] = total_reward
        rewards_history.append(total_reward)
        
        # Keep only last 3 episodes
        recent_episodes.append(episode_record)
        if len(recent_episodes) > 3:
            recent_episodes.pop(0)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = len([r for r in rewards_history[-100:] if r > 0]) / 100 * 100
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.1f}, Success Rate: {success_rate:.1f}%, Epsilon: {agent.epsilon:.3f}")
    
    # Display last 3 episodes
    print("\n" + "="*50)
    print("LAST 3 EPISODES SUMMARY")
    print("="*50)
    
    for ep in recent_episodes:
        status = "SUCCESS" if ep['success'] else "FAILED"
        print(f"Episode {ep['episode']}: {status}")
        print(f"  Steps taken: {len(ep['steps'])}")
        print(f"  Total reward: {ep['total_reward']}")
        print(f"  Final position: {ep['steps'][-1]['agent_pos'] if ep['steps'] else 'N/A'}")
        print("-" * 30)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    window_size = 50
    if len(rewards_history) >= window_size:
        moving_avg = [np.mean(rewards_history[max(0, i-window_size+1):i+1]) for i in range(len(rewards_history))]
        plt.plot(moving_avg)
        plt.title(f'Moving Average Reward (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return agent, rewards_history


def test_trained_agent(agent, grid_size: int = 5, episodes: int = 3):
    """Test the trained agent."""
    env = GridWorldEnv(size=grid_size)
    agent.epsilon = 0  # No exploration during testing
    
    print(f"\nTesting trained agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = grid_size * grid_size * 2
        
        print(f"\nTest Episode {episode + 1}:")
        env.render()
        
        while steps < max_steps:
            action = agent.get_action(state, grid_size)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            
            if done:
                print(f"Success! Reached target in {steps} steps with reward {total_reward}")
                break
        
        if not done:
            print(f"Failed to reach target in {max_steps} steps")


# Main execution
if __name__ == "__main__":
    print("Enhanced Grid World Game with Q-Learning")
    print("=" * 40)
    
    # Train the agent
    trained_agent, training_rewards = train_grid_agent(episodes=20000, grid_size=6)
    
    # Test the trained agent
    test_trained_agent(trained_agent, grid_size=6, episodes=1)
    
