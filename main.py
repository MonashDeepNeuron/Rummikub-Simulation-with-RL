"""
Main Training Script for Rummikub RL Agent, NOT DONE

This is where you train your RL agent against the ILP baseline opponent.

Usage:
    python main.py
"""

import numpy as np
import time
from typing import List
from Rummikub_env import RummikubEnv, RummikubAction
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent import ILPOpponent


class RandomAgent:
    """
    Simple random agent for testing.
    Replace this with your actual RL agent (DQN, PPO, AlphaZero, etc.)
    """
    
    def __init__(self):
        self.name = "RandomAgent"
    
    def select_action(self, state: dict, legal_actions: List[RummikubAction]) -> RummikubAction:
        """Select a random action from legal actions."""
        return np.random.choice(legal_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update agent based on experience.
        
        This is where your RL algorithm would update:
        - DQN: Update Q-network
        - PPO: Collect trajectory and update policy
        - AlphaZero: Update value/policy network with MCTS
        """
        pass  # Random agent doesn't learn


class TrainingStats:
    """Track training statistics."""
    
    def __init__(self):
        self.episodes = 0
        self.agent_wins = 0
        self.opponent_wins = 0
        self.ties = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def record_episode(self, winner: int, agent_player: int, reward: float, length: int):
        """Record results from an episode."""
        self.episodes += 1
        self.total_reward += reward
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if winner == agent_player:
            self.agent_wins += 1
        elif winner == 1 - agent_player:
            self.opponent_wins += 1
        else:
            self.ties += 1
    
    def get_win_rate(self) -> float:
        """Calculate agent win rate."""
        if self.episodes == 0:
            return 0.0
        return self.agent_wins / self.episodes
    
    def print_summary(self, last_n: int = 100):
        """Print training summary."""
        print(f"\n{'='*70}")
        print("TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total episodes: {self.episodes}")
        print(f"Agent wins: {self.agent_wins} ({self.get_win_rate():.1%})")
        print(f"Opponent wins: {self.opponent_wins}")
        print(f"Ties: {self.ties}")
        print(f"Avg reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Avg episode length: {np.mean(self.episode_lengths):.1f} turns")
        
        if len(self.episode_rewards) >= last_n:
            recent_rewards = self.episode_rewards[-last_n:]
            recent_wins = sum(1 for i in range(-last_n, 0) 
                            if i < 0 and self.episode_rewards[i] > 0)
            print(f"\nLast {last_n} episodes:")
            print(f"  Win rate: {recent_wins/last_n:.1%}")
            print(f"  Avg reward: {np.mean(recent_rewards):.2f}")


def train_agent(agent, 
                num_episodes: int = 1000,
                opponent_objective: str = 'maximize_value_minimize_changes',
                action_gen_mode: SolverMode = SolverMode.HYBRID,
                verbose: bool = True):
    """
    Train an RL agent against the ILP baseline opponent.
    
    Args:
        agent: Your RL agent (must have select_action and learn methods)
        num_episodes: Number of training episodes
        opponent_objective: ILP opponent strategy
        action_gen_mode: Action generator mode for agent
        verbose: Print progress
    """
    
    # Setup
    env = RummikubEnv()
    env.action_generator = ActionGenerator(mode=action_gen_mode, max_ilp_calls=30)
    opponent = ILPOpponent(objective=opponent_objective)
    stats = TrainingStats()
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Agent: {agent.name}")
    print(f"Opponent: ILP ({opponent_objective})")
    print(f"Action Generator: {action_gen_mode.value}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        done = False
        
        # Randomly assign agent to player 0 or 1
        agent_player = np.random.randint(2)
        
        episode_reward = 0
        turn_count = 0
        
        # Episode loop
        while not done:
            turn_count += 1
            
            if env.current_player == agent_player:
                # Agent's turn
                legal_actions = env.get_legal_actions(agent_player)
                
                if not legal_actions:
                    print(f"WARNING: No legal actions for agent in episode {episode}")
                    break
                
                action = agent.select_action(state, legal_actions)
                next_state, reward, done, info = env.step(action)
                
                # Agent learns from this experience
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
            else:
                # Opponent's turn
                action = opponent.select_action(
                    env.player_hands[env.current_player],
                    env.table,
                    env.has_melded[env.current_player],
                    len(env.tiles_deck)
                )
                
                state, reward, done, info = env.step(action)
        
        # Record episode results
        stats.record_episode(env.winner, agent_player, episode_reward, turn_count)
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            win_rate = stats.get_win_rate()
            avg_reward = np.mean(stats.episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Win Rate: {win_rate:.1%} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Turns: {turn_count}")
        
        # Print detailed summary every 100 episodes
        if verbose and (episode + 1) % 100 == 0:
            stats.print_summary(last_n=100)
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/num_episodes:.2f}s per episode)")
    stats.print_summary(last_n=100)
    
    return stats


def evaluate_agent(agent, 
                   num_games: int = 100,
                   opponent_objective: str = 'maximize_value_minimize_changes'):
    """
    Evaluate agent performance against opponent.
    
    Returns:
        dict with evaluation metrics
    """
    env = RummikubEnv()
    env.action_generator = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=30)
    opponent = ILPOpponent(objective=opponent_objective)
    
    wins = 0
    losses = 0
    ties = 0
    
    print(f"\nEvaluating agent over {num_games} games...")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        
        # Agent always player 0 for evaluation
        agent_player = 0
        
        while not done:
            if env.current_player == agent_player:
                legal_actions = env.get_legal_actions(agent_player)
                action = agent.select_action(state, legal_actions)
            else:
                action = opponent.select_action(
                    env.player_hands[1],
                    env.table,
                    env.has_melded[1],
                    len(env.tiles_deck)
                )
            
            state, reward, done, info = env.step(action)
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner == 1 - agent_player:
            losses += 1
        else:
            ties += 1
        
        if (game + 1) % 10 == 0:
            print(f"  {game + 1}/{num_games} games completed...")
    
    win_rate = wins / num_games
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Games: {num_games}")
    print(f"Wins: {wins} ({win_rate:.1%})")
    print(f"Losses: {losses}")
    print(f"Ties: {ties}")
    print(f"{'='*70}\n")
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'ties': ties
    }


def main():
    """Main training pipeline."""
    
    print("\n" + "="*70)
    print("RUMMIKUB RL TRAINING")
    print("="*70)
    
    # Create your agent
    # TODO: Replace RandomAgent with your actual RL agent (DQN, PPO, etc.)
    agent = RandomAgent()
    
    # Training configuration
    config = {
        'num_episodes': 1000,
        'opponent_objective': 'maximize_value_minimize_changes',  # Use Model 2
        'action_gen_mode': SolverMode.HYBRID,  # Balanced speed/completeness
        'verbose': True
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    print("\nStarting training...")
    stats = train_agent(agent, **config)
    
    # Evaluate
    print("\n" + "="*70)
    print("Final evaluation against opponent...")
    results = evaluate_agent(agent, num_games=100)
    
    # Save agent (if applicable)
    # TODO: Implement agent saving
    # agent.save('trained_agent.pkl')
    # print("\nAgent saved to 'trained_agent.pkl'")
    
    print("\nTraining complete! 🎉")


if __name__ == "__main__":
    main()