# main.py
"""
Main Training Script for Rummikub RL Agent with A3C

Reward function:
Step rewards (accumulated during game):
  1. R_t = (hand_value_before - hand_value_after)
  2. Ice-breaking bonus: +20 (one-time, when agent first melds 30+ points)
  3. Drawing penalty: -5

Terminal rewards (at game end):
  4. Win by empty hand: +200 + opponent_hand_value
  5. Win by lowest hand (pool empty): +10
  6. Lose when opponent empties hand: -(200 + my_hand_value)
  7. Lose by lowest hand: -10

Episode reward = sum of step rewards + terminal reward
Terminal rewards are designed to dominate, ensuring:
  - Winning -> positive total reward
  - Losing -> negative total reward

    Running:
        Resume training from checkpoint: 
            python main.py --checkpoint checkpoint_13.pth 

        Resume training from checkpoint with custom settings:
            python main.py --checkpoint checkpoint_13.pth --workers 3 --episodes 1000

        Evaluate a checkpoint only (no training):
            python main.py --checkpoint trained_agent_final.pth --eval-only 

        Start fresh training (default):
            python main.py
"""

import numpy as np
import time
import multiprocessing as mp
from typing import List
import sys
import io

from Rummikub_env import RummikubEnv, RummikubAction
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent2 import RummikubILPSolver
from agent import ACAgent, ActorCritic, get_state_vec, get_action_vec

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


class TrainingStats:
    """Track training statistics across workers."""
    
    def __init__(self, manager):
        self.lock = manager.Lock()
        self.episodes = manager.Value('i', 0)
        self.agent_wins = manager.Value('i', 0)
        self.opponent_wins = manager.Value('i', 0)
        self.ties = manager.Value('i', 0)
        self.total_reward = manager.Value('d', 0.0)
        self.episode_rewards = manager.list()
        self.episode_lengths = manager.list()
    
    def record_episode(self, winner, agent_player, reward, length):
        with self.lock:
            self.episodes.value += 1
            self.total_reward.value += reward
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            if winner == agent_player:
                self.agent_wins.value += 1
            elif winner is not None and winner == 1 - agent_player:
                self.opponent_wins.value += 1
            else:
                self.ties.value += 1
    
    def get_win_rate(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
        return wins / eps if eps > 0 else 0.0
    
    def get_rewards_list(self):
        with self.lock:
            return list(self.episode_rewards)
    
    def print_summary(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
            opp_wins = self.opponent_wins.value
            ties = self.ties.value
            total_r = self.total_reward.value
            lengths = list(self.episode_lengths)
        
        if eps == 0:
            print("No episodes completed yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY ({eps} episodes)")
        print(f"{'='*60}")
        print(f"Agent wins: {wins} ({wins/eps:.1%})")
        print(f"Opponent wins: {opp_wins} ({opp_wins/eps:.1%})")
        print(f"Ties: {ties}")
        print(f"Avg reward: {total_r / eps:.2f}")
        if lengths:
            print(f"Avg episode length: {np.mean(lengths):.1f} turns")


class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def compute_agent_reward(env, agent_player, action, reward_from_env, done, info):
    """
    Compute the reward from the AGENT's perspective.
    
    Reward function:
    - Non-terminal steps: R_t = (hand_before - hand_after) [+ ice break +20] [- draw penalty 5]
    - Win by empty hand: R_T = 200 + opponent_hand_value
    - Win by lowest hand: R_T = +10
    - Lose by opponent empty hand: R_T = -(my_hand_value)  # Symmetric with win
    - Lose by lowest hand: R_T = -10
    
    The terminal rewards are designed to dominate intermediate rewards so that:
    - Winning always results in positive total episode reward
    - Losing always results in negative total episode reward
    """
    if done:
        winner = info.get('winner')
        win_type = info.get('win_type', '')
        
        if winner == agent_player:
            # Agent wins
            if win_type == 'emptied_hand':
                # Win by empty hand: 200 + opponent's hand value
                opp_hand_value = info.get('final_opponent_hand_value', 0)
                return 200 + opp_hand_value
            elif win_type == 'lowest_hand':
                # Win by lowest hand (tie-breaker)
                return 10
            else:
                return 10  # Default win
        elif winner is not None:
            # Agent loses (winner is opponent)
            if win_type == 'emptied_hand':
                # Lose when opponent empties hand: -(agent's hand value)
                # This makes loss penalty symmetric with win bonus
                my_hand_value = info.get('final_opponent_hand_value', 0)  # Agent's hand from winner's view
                return -my_hand_value
            elif win_type == 'lowest_hand':
                # Lose by lowest hand
                return -10
            else:
                return -10  # Default loss
        else:
            # Tie
            return 0
    
    # Not terminal - return the reward from env (for agent's own actions)
    return reward_from_env


def worker_process(worker_id, global_model, optimizer, num_episodes, config, stats):
    """Worker process for A3C training."""
    
    prefix = f"[W{worker_id}]"
    
    # Create agent
    agent = ACAgent(global_model=global_model, optimizer=optimizer, is_worker=True, use_gpu=True)
    
    print(f"{prefix} Starting on {agent.device}...")
    
    # Create environment
    with SuppressOutput():
        env = RummikubEnv()
        env.action_generator = ActionGenerator(
            mode=config['action_gen_mode'], 
            max_ilp_calls=50, 
            max_window_size=3, 
            timeout_seconds=30
        )
    
    # Create opponent
    opponent = RummikubILPSolver()
    
    print(f"{prefix} Initialized. Starting training...")
    
    episode_times = []  # Track episode durations
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        
        # Sync and reset
        agent.sync_local_to_global()
        agent.reset_hidden()
        
        # Reset environment
        with SuppressOutput():
            state = env.reset()
        
        done = False
        episode_reward = 0.0  # Accumulate ONLY agent's rewards
        turn_count = 0
        agent_player = np.random.randint(2)
        
        # Track actions
        agent_draws = 0
        agent_plays = 0
        opp_draws = 0
        opp_plays = 0
        ice_broken_turn = -1
        
        # Maximum turns to prevent infinite loops
        MAX_TURNS = 500
        
        agent.observe(state)
        
        while not done and turn_count < MAX_TURNS:
            turn_count += 1
            current_player = env.current_player
            
            if current_player == agent_player:
                # === AGENT'S TURN ===
                state_vec = get_state_vec(state)
                
                with SuppressOutput():
                    legal_actions = env.get_legal_actions(agent_player)
                
                if not legal_actions:
                    action = RummikubAction(action_type='draw')
                    action_idx = -1
                    action_vec = get_action_vec(action)
                    num_actions = 0
                    agent_draws += 1
                else:
                    action, action_idx, action_vecs_list = agent.select_action(state, legal_actions)
                    action_vec = action_vecs_list[action_idx] if 0 <= action_idx < len(action_vecs_list) else get_action_vec(action)
                    num_actions = len(legal_actions)
                    
                    if action.action_type == 'draw':
                        agent_draws += 1
                    else:
                        agent_plays += 1
                
                next_state, reward_env, done, info = env.step(action)
                
                # Compute agent's reward
                if done:
                    agent_reward = compute_agent_reward(env, agent_player, action, reward_env, done, info)
                else:
                    agent_reward = reward_env  # Use env reward for non-terminal
                
                if info.get('ice_broken') and ice_broken_turn < 0:
                    ice_broken_turn = turn_count
                
                next_state_vec = get_state_vec(next_state) if not done else None
                
                # Learn with agent's reward
                agent.learn(state_vec, action_idx, action_vec, agent_reward, next_state_vec, done, info, num_actions)
                
                # Accumulate agent's reward
                episode_reward += agent_reward
                state = next_state
                
            else:
                # === OPPONENT'S TURN ===
                state_vec = get_state_vec(state)
                
                action = opponent.solve(
                    env.player_hands[current_player],
                    env.table,
                    env.has_melded[current_player]
                )
                
                if action is None:
                    action = RummikubAction(action_type='draw')
                    opp_draws += 1
                else:
                    if action.action_type == 'draw':
                        opp_draws += 1
                    else:
                        opp_plays += 1
                
                next_state, reward_env, done, info = env.step(action)
                
                if info.get('ice_broken') and ice_broken_turn < 0:
                    ice_broken_turn = turn_count
                
                # If game ends on opponent's turn, compute agent's terminal reward
                if done:
                    agent_reward = compute_agent_reward(env, agent_player, action, reward_env, done, info)
                    episode_reward += agent_reward
                    
                    # Learn the terminal transition
                    next_state_vec = None
                    agent.learn(state_vec, -1, None, agent_reward, next_state_vec, done, info, 0)
                else:
                    # Opponent's intermediate turns: just observe, no reward for agent
                    agent.learn(state_vec, -1, None, 0, get_state_vec(next_state), done, info, 0)
                    agent.observe(next_state)
                
                state = next_state
        
        # Handle infinite loop case
        if turn_count >= MAX_TURNS:
            print(f"{prefix} WARNING: Episode {episode+1} hit max turns ({MAX_TURNS}), forcing end")
            episode_reward = -100
        
        # Record stats
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        stats.record_episode(env.winner, agent_player, episode_reward, turn_count)
        
        # Determine winner string
        if env.winner == agent_player:
            winner_str = "AGENT WIN"
        elif env.winner is not None:
            winner_str = "OPP WIN"
        else:
            winner_str = "TIE"
        
        # Print episode summary with timing
        ice_str = f"T{ice_broken_turn:3d}" if ice_broken_turn > 0 else "  -"
        print(f"{prefix} Ep {episode+1:3d}/{num_episodes} | {winner_str:9s} | "
              f"Reward:{episode_reward:7.1f} | Turns:{turn_count:3d} | "
              f"Agent(Draws:{agent_draws:2d} Play:{agent_plays:2d}) Opp(Draw:{opp_draws:2d} Play:{opp_plays:2d}) | "
              f"Ice:{ice_str} | {episode_time:.1f}s")
        
        # Detailed stats every 25 episodes
        if (episode + 1) % 25 == 0:
            win_rate = stats.get_win_rate()
            avg_time = np.mean(episode_times[-25:]) if len(episode_times) >= 25 else np.mean(episode_times)
            print(f"{prefix} === Global: {stats.episodes.value} eps, {win_rate:.1%} win rate, avg {avg_time:.1f}s/ep ===")


def save_reward_plot(rewards, filename='training_rewards.png'):
    """Save a plot of episode rewards."""
    if len(rewards) < 2:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = min(50, len(rewards) // 4) if len(rewards) > 10 else len(rewards)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-ep Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot reward distribution
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero')
    plt.axvline(x=np.mean(rewards), color='g', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved reward plot to {filename}")


def train_a3c(num_workers=4, num_episodes_per_worker=500, config=None, checkpoint_path=None):
    """
    Main A3C training function.
    
    Args:
        num_workers: Number of parallel worker processes
        num_episodes_per_worker: Episodes each worker will run
        config: Configuration dict (action_gen_mode, etc.)
        checkpoint_path: Path to checkpoint.pth to resume training from (optional)
    """
    
    if config is None:
        config = {
            'action_gen_mode': SolverMode.HYBRID,
        }
    
    manager = mp.Manager()
    stats = TrainingStats(manager)
    
    # Create global model (CPU for shared memory)
    global_model = ActorCritic()
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")
        global_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        print(f"Successfully loaded weights from {checkpoint_path}")
    
    global_model.share_memory()
    
    optimizer = optim.Adam(global_model.parameters(), lr=0.0001)
    
    print(f"\n{'='*60}")
    print("A3C TRAINING - RUMMIKUB")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}")
    print(f"Episodes per worker: {num_episodes_per_worker}")
    print(f"Total episodes: {num_workers * num_episodes_per_worker}")
    if checkpoint_path:
        print(f"Resuming from: {checkpoint_path}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print(f"GPU: Not available (using CPU)")
    
    print(f"{'='*60}")
    print("Reward Function:")
    print("  Step rewards (accumulated during game):")
    print("    - Play tiles: (hand_before - hand_after)")
    print("    - Draw: (hand_before - hand_after) - 5")
    print("    - Ice break (one-time): +20 bonus")
    print("  Terminal rewards (at game end):")
    print("    - Win (empty hand): +200 + opponent_hand_value")
    print("    - Win (lowest hand): +10")
    print("    - Lose (opp empty): -(my_hand_value)")
    print("    - Lose (lowest hand): -10")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Start workers
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process, 
            args=(worker_id, global_model, optimizer, num_episodes_per_worker, config, stats)
        )
        p.start()
        processes.append(p)
    
    # Monitor and save checkpoints
    total_expected = num_workers * num_episodes_per_worker
    last_checkpoint_eps = 0
    checkpoint_interval = 40  # Save every 40 episodes (10 per worker * 4 workers)
    
    while any(p.is_alive() for p in processes):
        time.sleep(10)  # Check every 10 seconds
        
        current_eps = stats.episodes.value
        elapsed = time.time() - start_time
        
        # Save checkpoint every ~10 episodes per worker
        if current_eps >= last_checkpoint_eps + checkpoint_interval:
            checkpoint_num = current_eps // checkpoint_interval
            torch.save(global_model.state_dict(), f'checkpoint_{checkpoint_num}.pth')
            
            # Save reward plot
            rewards = stats.get_rewards_list()
            if rewards:
                save_reward_plot(rewards, 'training_rewards.png')
            
            last_checkpoint_eps = current_eps
            
            # Print progress
            if current_eps > 0:
                eps_per_min = current_eps / (elapsed / 60)
                remaining = total_expected - current_eps
                eta_min = remaining / eps_per_min if eps_per_min > 0 else 0
                print(f"\n[MAIN] Progress: {current_eps}/{total_expected} ({100*current_eps/total_expected:.1f}%) | "
                      f"Win rate: {stats.get_win_rate():.1%} | "
                      f"ETA: {eta_min:.1f} min | Saved checkpoint_{checkpoint_num}.pth")
    
    for p in processes:
        p.join()
    
    # Final save
    torch.save(global_model.state_dict(), 'trained_agent_final.pth')
    
    # Final reward plot
    rewards = stats.get_rewards_list()
    if rewards:
        save_reward_plot(rewards, 'training_rewards_final.png')
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    stats.print_summary()
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    agent = ACAgent()
    agent.load('trained_agent_final.pth')
    evaluate_agent(agent, num_games=50)


def evaluate_agent(agent, num_games=50):
    """Evaluate agent against opponent."""
    
    with SuppressOutput():
        env = RummikubEnv()
        env.action_generator = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=50, max_window_size=3, timeout_seconds=30)
    
    opponent = RummikubILPSolver()
    
    wins = losses = ties = 0
    total_reward = 0
    
    print(f"Evaluating over {num_games} games...")
    
    for game in range(num_games):
        with SuppressOutput():
            state = env.reset()
        
        done = False
        agent_player = 0
        game_reward = 0
        
        agent.reset_hidden()
        agent.observe(state)
        
        turn = 0
        while not done and turn < 500:
            turn += 1
            if env.current_player == agent_player:
                with SuppressOutput():
                    legal_actions = env.get_legal_actions(agent_player)
                if not legal_actions:
                    action = RummikubAction(action_type='draw')
                else:
                    action, _, _ = agent.select_action(state, legal_actions)
                state, reward, done, info = env.step(action)
                
                if done:
                    game_reward = compute_agent_reward(env, agent_player, action, reward, done, info)
            else:
                action = opponent.solve(env.player_hands[env.current_player], env.table, env.has_melded[env.current_player])
                if action is None:
                    action = RummikubAction(action_type='draw')
                state, reward, done, info = env.step(action)
                agent.observe(state)
                
                if done:
                    game_reward = compute_agent_reward(env, agent_player, action, reward, done, info)
        
        total_reward += game_reward
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner is not None:
            losses += 1
        else:
            ties += 1
        
        if (game + 1) % 10 == 0:
            print(f"  {game+1}/{num_games} - W:{wins} L:{losses} T:{ties} | Avg R: {total_reward/(game+1):.1f}")
    
    print(f"\nFinal: {wins}W / {losses}L / {ties}T = {100*wins/num_games:.1f}% win rate")
    print(f"Average reward: {total_reward/num_games:.1f}")
    
    return {'wins': wins, 'losses': losses, 'ties': ties, 'win_rate': wins/num_games}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='A3C Training for Rummikub')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint.pth to resume training from')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--episodes', '-e', type=int, default=500,
                        help='Episodes per worker (default: 500)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate the checkpoint, no training')
    
    args = parser.parse_args()
    
    if args.eval_only:
        if args.checkpoint is None:
            print("Error: --eval-only requires --checkpoint")
            return
        print(f"Evaluating checkpoint: {args.checkpoint}")
        agent = ACAgent()
        agent.load(args.checkpoint)
        evaluate_agent(agent, num_games=100)
    else:
        train_a3c(
            num_workers=args.workers,
            num_episodes_per_worker=args.episodes,
            checkpoint_path=args.checkpoint
        )


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()