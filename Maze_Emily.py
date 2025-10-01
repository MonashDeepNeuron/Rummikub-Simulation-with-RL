import numpy as np
import gymnasium as gym
import pygame
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size):
        self.size = size
        self.window_size = 512
        self.render_mode = "human"

        self._agent_location = np.array([0,0])
        self._target_location = np.array([size- 1, size - 1])

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # coordinate low bound, coordinate high bound, dimensions, data type
                "goal": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1,0]),    # moving right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=1  # manhattan distance
            )
        }

    def _get_state(self):
        # convert dict observation into a simple tuple for indexing Q-table
        return tuple(self._agent_location.tolist() + self._target_location.tolist())

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     self._agent_location = np.array([0, 0])             # always bottom-left
    #     self._target_location = np.array([self.size-1, self.size-1])  # always top-right
    #     return self._get_obs(), self._get_info()


    def step(self, action):
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

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        # Reward shaping
        if terminated:
            reward = 10
        else:
            reward = -1   # step penalty

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


if __name__ == "__main__":
    env = MazeEnv(size=5)

    # Q-learning parameters
    alpha = 0.1      # learning rate
    gamma = 0.99     # discount factor
    epsilon = 1.0    # exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.05
    episodes = 5000

    # Q-table: dictionary mapping state -> action values
    Q = {}
    rewards_per_episode = []

    def get_Q(state):
        if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
        return Q[state]

    # Training loop
    for episode in range(episodes):
        obs, info = env.reset()
        state = env._get_state()
        done = False
        total_reward = 0

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(get_Q(state))

            next_obs, reward, done, truncated, info = env.step(action)
            next_state = env._get_state()

            # Ensure both current and next state exist in Q
            q_values = get_Q(state)
            next_q_values = get_Q(next_state)

            # Q-learning update
            best_next_action = np.argmax(get_Q(next_state))
            target = reward + gamma * get_Q(next_state)[best_next_action]
            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode+1}: total reward {total_reward}, epsilon {epsilon:.3f}")

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning: Rewards per Episode")
    plt.show()

    obs, info = env.reset()
    state = env._get_state()
    done = False

    while not done:
        action = np.argmax(get_Q(state))  # greedy policy
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        state = env._get_state()
