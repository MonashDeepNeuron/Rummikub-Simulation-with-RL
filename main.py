"""
Main Training Script for Rummikub RL Agent with A3C

REWARD SYSTEM (v2 - No large constants):
=========================================
Intermediate Rewards (per turn):
  - Base play: 2.0 × (hand_before - hand_after)
  - Draw: 2.0 × (hand_before - hand_after) - 5.0
  - Ice break bonus: +30.0 (one-time)
  - Manipulation bonus: +10.0
  - 4+ tiles bonus: +5.0
  - Extension bonus: +3.0
  - Large hand penalty: -2.0 (if 20+ tiles)

Terminal Rewards (NO LARGE CONSTANTS):
  - Win by empty hand: +opponent_hand_value
  - Lose by opponent empty: -my_hand_value  
  - Pool empty: ±(difference in hand values)

Usage:
    python main.py                              # Fresh training
    python main.py --checkpoint ckpt.pth        # Resume training
    python main.py --checkpoint ckpt.pth --eval-only  # Evaluate only
"""

import numpy as np
import time
import multiprocessing as mp
from typing import List
from datetime import datetime
import sys
import io

from Rummikub_env import RummikubEnv, RummikubAction
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent2 import RummikubILPSolver
from agent import ACAgent, ActorCritic, get_state_vec, get_action_vec

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Config:
    """Training configuration. Reward params are in RummikubEnv class."""
    
    # Network architecture
    HIDDEN_SIZE = 512
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.1
    USE_LAYER_NORM = True
    
    # Training parameters
    LEARNING_RATE = 0.0003  
    WEIGHT_DECAY = 1e-5
    EXPLORATION_PROB = 0.05
    GAMMA = 0.99
    ENTROPY_COEF = 0.02
    VALUE_COEF = 0.5
    BATCH_SIZE = 64
    GRAD_CLIP = 0.5
    
    # Timeout parameters
    MAX_TURNS = 150
    ACTION_GEN_TIMEOUT = 10.0      # Passed to ActionGenerator (internal timeout)
    OPPONENT_TIMEOUT = 5.0         # Passed to RummikubILPSolver (internal timeout)
    MAX_EPISODE_TIME = 180.0       # 3 minutes max per episode (safety net)
    
    # Action generator settings
    ACTION_GEN_MODE = SolverMode.HYBRID
    MAX_ILP_CALLS = 20
    MAX_WINDOW_SIZE = 2
    
    # Checkpoint settings
    CHECKPOINT_INTERVAL = 10       # Save every N episodes total


def get_timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y.%m.%d %H:%M:%S")


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
        self.terminal_rewards = manager.list()
        self.episode_lengths = manager.list()
        # Track win/loss for each episode (1=win, 0=loss, 0.5=tie)
        self.episode_results = manager.list()
    
    def record_episode(self, winner, agent_player, reward, terminal_reward, length):
        with self.lock:
            self.episodes.value += 1
            self.total_reward.value += reward
            self.episode_rewards.append(reward)
            self.terminal_rewards.append(terminal_reward)
            self.episode_lengths.append(length)
            
            if winner == agent_player:
                self.agent_wins.value += 1
                self.episode_results.append(1.0)  # Win
            elif winner is not None and winner == 1 - agent_player:
                self.opponent_wins.value += 1
                self.episode_results.append(0.0)  # Loss
            else:
                self.ties.value += 1
                self.episode_results.append(0.5)  # Tie
    
    def get_win_rate(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
        return wins / eps if eps > 0 else 0.0
    
    def get_rewards_list(self):
        with self.lock:
            return list(self.episode_rewards), list(self.terminal_rewards)
    
    def get_episode_results(self):
        """Get list of episode results (1=win, 0=loss, 0.5=tie)."""
        with self.lock:
            return list(self.episode_results)
    
    def print_summary(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
            opp_wins = self.opponent_wins.value
            ties = self.ties.value
            total_r = self.total_reward.value
            lengths = list(self.episode_lengths)
        
        if eps == 0:
            print(f"No episodes completed yet. | [{get_timestamp()}]")
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


def plot_winning_rate(episode_results, filename='winning_rate.png', window_size=10, ma_window=50):
    """
    Plot winning rate over episodes with moving average.
    
    Args:
        episode_results: List of episode results (1=win, 0=loss, 0.5=tie)
        filename: Output filename
        window_size: Number of episodes per data point (default: 10)
        ma_window: Moving average window in episodes (default: 50)
    """
    if len(episode_results) < window_size:
        return
    
    # Calculate winning rate for each window
    num_windows = len(episode_results) // window_size
    if num_windows == 0:
        return
    
    win_rates = []
    x_values = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_results = episode_results[start_idx:end_idx]
        
        # Win rate: wins / total (treating ties as 0.5 wins)
        win_rate = sum(window_results) / len(window_results)
        win_rates.append(win_rate)
        x_values.append(end_idx)  # Episode number at end of window
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw winning rate
    ax.plot(x_values, win_rates, 'b-', alpha=0.4, linewidth=1, label=f'Win Rate (per {window_size} eps)')
    ax.scatter(x_values, win_rates, c='blue', s=20, alpha=0.5)
    
    # Calculate and plot moving average
    ma_windows = ma_window // window_size  # Number of data points for MA
    if len(win_rates) >= ma_windows and ma_windows > 1:
        ma = np.convolve(win_rates, np.ones(ma_windows)/ma_windows, mode='valid')
        ma_x = x_values[ma_windows-1:]  # Align MA with end of window
        ax.plot(ma_x, ma, 'r-', linewidth=2.5, label=f'{ma_window}-ep Moving Avg')
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (Random)')
    ax.axhline(y=0.2, color='orange', linestyle=':', alpha=0.5, label='20% (Baseline)')
    
    # Current overall win rate
    overall_win_rate = sum(episode_results) / len(episode_results)
    ax.axhline(y=overall_win_rate, color='green', linestyle='--', alpha=0.7, 
               label=f'Overall: {overall_win_rate:.1%}')
    
    # Formatting
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title(f'Agent Win Rate Over Training\n(Window: {window_size} episodes, MA: {ma_window} episodes)', 
                 fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(x_values) * 1.02)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add current stats text box
    latest_win_rate = win_rates[-1] if win_rates else 0
    stats_text = f'Episodes: {len(episode_results)}\nLatest {window_size}-ep: {latest_win_rate:.1%}\nOverall: {overall_win_rate:.1%}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved winning rate plot to {filename}")


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


def compute_agent_reward(agent_player, acting_player, done, info):
    """
    Compute the TERMINAL reward from the AGENT's perspective.
    
    CRITICAL: info['final_my_hand_value'] and info['final_opponent_hand_value']
    are from the ACTING PLAYER's perspective, not necessarily the agent's.
    We must convert them to the agent's perspective.
    
    Args:
        agent_player: Which player (0 or 1) is the agent
        acting_player: Which player was acting when game ended
        done: Whether game is over
        info: Info dict from env.step()
    
    Returns:
        Terminal reward from agent's perspective
    """
    if not done:
        return 0
    
    winner = info.get('winner')
    win_type = info.get('win_type', '')
    
    # Get hand values from info (from ACTING player's perspective)
    acting_hand = info.get('final_my_hand_value', 0)
    other_hand = info.get('final_opponent_hand_value', 0)
    
    # Convert to AGENT's perspective
    if acting_player == agent_player:
        # Agent was acting: my=agent, opponent=opponent
        agent_hand = acting_hand
        opponent_hand = other_hand
    else:
        # Opponent was acting: my=opponent, opponent=agent
        agent_hand = other_hand
        opponent_hand = acting_hand
    
    if winner is None:
        # Tie
        return 0
    
    if winner == agent_player:
        # Agent wins
        if win_type == 'emptied_hand':
            # Agent emptied hand: +opponent's hand value
            return opponent_hand
        elif win_type == 'lowest_hand':
            # Agent has lower hand: +(opponent_hand - agent_hand)
            return opponent_hand - agent_hand
        else:
            return 10  # Default win
    else:
        # Agent loses
        if win_type == 'emptied_hand':
            # Opponent emptied hand: -agent's hand value
            return -agent_hand
        elif win_type == 'lowest_hand':
            # Agent has higher hand: -(agent_hand - opponent_hand)
            return -(agent_hand - opponent_hand)
        else:
            return -10  # Default loss


def worker_process(worker_id, global_model, optimizer, num_episodes, config, stats):
    """Worker process for A3C training."""
    
    prefix = f"[W{worker_id}]"
    
    # Create agent with Config settings
    agent = ACAgent(
        global_model=global_model, 
        optimizer=optimizer, 
        is_worker=True, 
        use_gpu=True,
        gamma=Config.GAMMA,
        entropy_coef=Config.ENTROPY_COEF,
        value_coef=Config.VALUE_COEF,
        batch_size=Config.BATCH_SIZE,
        grad_clip=Config.GRAD_CLIP,
        exploration_prob=Config.EXPLORATION_PROB
    )
    
    print(f"{prefix} Starting on {agent.device}...")
    
    # Create environment with Config settings
    with SuppressOutput():
        env = RummikubEnv()
        env.action_generator = ActionGenerator(
            mode=Config.ACTION_GEN_MODE, 
            max_ilp_calls=Config.MAX_ILP_CALLS, 
            max_window_size=Config.MAX_WINDOW_SIZE, 
            timeout_seconds=Config.ACTION_GEN_TIMEOUT
        )
    
    # Create opponent with Config settings
    opponent = RummikubILPSolver(timeout=Config.OPPONENT_TIMEOUT)
    
    print(f"{prefix} Initialized. Starting training...")
    
    episode_times = []
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        
        # Sync and reset
        agent.sync_local_to_global()
        agent.reset_hidden()
        
        # Reset environment
        with SuppressOutput():
            state = env.reset()
        
        done = False
        episode_reward = 0.0
        terminal_reward = 0.0
        turn_count = 0
        agent_player = np.random.randint(2)
        
        # Track actions
        agent_draws = 0
        agent_plays = 0
        opp_draws = 0
        opp_plays = 0
        ice_broken_turn = -1
        timeout_occurred = False
        
        # Initial observation
        with SuppressOutput():
            legal_actions = env.get_legal_actions(env.current_player) if env.current_player == agent_player else []
        agent.observe(state, legal_actions)
        
        while not done and turn_count < Config.MAX_TURNS:
            # Check episode time limit
            if time.time() - episode_start_time > Config.MAX_EPISODE_TIME:
                timeout_occurred = True
                break
                
            turn_count += 1
            current_player = env.current_player
            
            if current_player == agent_player:
                # === AGENT'S TURN ===
                with SuppressOutput():
                    legal_actions = env.get_legal_actions(agent_player)
                
                state_vec = get_state_vec(state, legal_actions)
                
                if not legal_actions:
                    action = RummikubAction(action_type='draw')
                    action_idx = -1
                    action_vec = get_action_vec(action)
                    all_action_vecs = []
                    num_actions = 0
                    agent_draws += 1
                else:
                    action, action_idx, all_action_vecs = agent.select_action(state, legal_actions)
                    action_vec = all_action_vecs[action_idx] if 0 <= action_idx < len(all_action_vecs) else get_action_vec(action)
                    num_actions = len(legal_actions)
                    
                    if action.action_type == 'draw':
                        agent_draws += 1
                    else:
                        agent_plays += 1
                
                # Execute action
                next_state, reward_env, done, info = env.step(action)
                
                if info.get('ice_broken') and ice_broken_turn < 0:
                    ice_broken_turn = turn_count
                
                # Get agent's reward
                if done:
                    # Agent was acting when game ended
                    term_r = compute_agent_reward(agent_player, agent_player, done, info)
                    agent_reward = reward_env + term_r  # Intermediate + terminal
                    terminal_reward = term_r
                else:
                    agent_reward = reward_env
                
                next_state_vec = get_state_vec(next_state) if not done else None
                
                # Learn with ALL action vectors for proper policy gradient
                agent.learn(state_vec, action_idx, action_vec, agent_reward, 
                           next_state_vec, done, info, num_actions, all_action_vecs)
                
                episode_reward += agent_reward
                state = next_state
                
            else:
                # === OPPONENT'S TURN ===
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
                    # Opponent was acting when game ended
                    acting_player = 1 - agent_player
                    term_r = compute_agent_reward(agent_player, acting_player, done, info)
                    episode_reward += term_r
                    terminal_reward = term_r
                    
                    # Store terminal transition
                    state_vec = get_state_vec(state)
                    agent.learn(state_vec, -1, None, term_r, None, done, info, 0, [])
                else:
                    # Observe opponent's action (no reward for agent)
                    with SuppressOutput():
                        next_legal = env.get_legal_actions(agent_player) if env.current_player == agent_player else []
                    agent.observe(next_state, next_legal)
                
                state = next_state
        
        # Handle timeout/max turns
        if turn_count >= Config.MAX_TURNS or timeout_occurred:
            if not timeout_occurred:
                print(f"{prefix} WARNING: Episode {episode+1} hit max turns ({Config.MAX_TURNS}) | [{get_timestamp()}]")
            else:
                print(f"{prefix} WARNING: Episode {episode+1} hit time limit ({Config.MAX_EPISODE_TIME}s) | [{get_timestamp()}]")
            timeout_occurred = True
            terminal_reward = -50
            episode_reward += terminal_reward
        
        # Record stats
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        stats.record_episode(env.winner, agent_player, episode_reward, terminal_reward, turn_count)
        
        # Determine winner string
        if env.winner == agent_player:
            winner_str = "AGENT WIN"
        elif env.winner is not None:
            winner_str = "OPP WIN"
        else:
            winner_str = "TIE"
        
        # Print episode summary
        ice_str = f"T{ice_broken_turn:3d}" if ice_broken_turn > 0 else "  -"
        timeout_str = " TIMEOUT" if timeout_occurred else ""
        
        print(f"{prefix} Ep {episode+1:3d}/{num_episodes} | {winner_str:9s} | "
              f"R:{episode_reward:7.1f} (term:{terminal_reward:7.1f}) | "
              f"T:{turn_count:3d} | A(D:{agent_draws:2d} P:{agent_plays:2d}) O(D:{opp_draws:2d} P:{opp_plays:2d}) | "
              f"Ice:{ice_str} | {episode_time:.1f}s{timeout_str} | [{get_timestamp()}]")
        
        # Global stats every 25 episodes
        if (episode + 1) % 25 == 0:
            win_rate = stats.get_win_rate()
            avg_time = np.mean(episode_times[-25:]) if len(episode_times) >= 25 else np.mean(episode_times)
            print(f"{prefix} === Global: {stats.episodes.value} eps, {win_rate:.1%} win rate, avg {avg_time:.1f}s/ep === [{get_timestamp()}]")


def save_reward_plot(rewards, terminal_rewards, filename='training_rewards.png'):
    """Save a 2x2 plot of episode rewards."""
    if len(rewards) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Total reward line
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, label='Episode Reward', color='blue')
    window = min(50, len(rewards) // 4) if len(rewards) > 10 else max(1, len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, 
                label=f'{window}-ep Moving Avg')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Total reward histogram
    ax2 = axes[0, 1]
    ax2.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax2.axvline(x=0, color='r', linestyle='--', label='Zero')
    ax2.axvline(x=np.mean(rewards), color='g', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Total Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Terminal reward line
    ax3 = axes[1, 0]
    ax3.plot(terminal_rewards, alpha=0.3, label='Terminal Reward', color='green')
    if window > 1 and len(terminal_rewards) >= window:
        term_avg = np.convolve(terminal_rewards, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(terminal_rewards)), term_avg, 'r-', linewidth=2,
                label=f'{window}-ep Moving Avg')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Terminal Reward')
    ax3.set_title('Terminal Rewards (No Large Constants)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Terminal reward histogram
    ax4 = axes[1, 1]
    ax4.hist(terminal_rewards, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax4.axvline(x=0, color='r', linestyle='--', label='Zero')
    ax4.axvline(x=np.mean(terminal_rewards), color='purple', linestyle='--', 
               label=f'Mean: {np.mean(terminal_rewards):.1f}')
    ax4.set_xlabel('Terminal Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Terminal Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved reward plot to {filename}")


def train_a3c(num_workers=4, num_episodes_per_worker=500, config=None, checkpoint_path=None):
    """Main A3C training function."""
    
    # Config dict is no longer needed - using Config class
    
    manager = mp.Manager()
    stats = TrainingStats(manager)
    
    # Create global model with Config settings (CPU for shared memory)
    global_model = ActorCritic(
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        dropout=Config.DROPOUT,
        use_layer_norm=Config.USE_LAYER_NORM
    )
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")
        global_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        print(f"Successfully loaded weights from {checkpoint_path}")
    
    global_model.share_memory()
    
    # Create optimizer with Config settings
    optimizer = optim.Adam(
        global_model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    print(f"\n{'='*60}")
    print("A3C TRAINING - RUMMIKUB (v2 Rewards)")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}")
    print(f"Episodes per worker: {num_episodes_per_worker}")
    print(f"Total episodes: {num_workers * num_episodes_per_worker}")
    if checkpoint_path:
        print(f"Resuming from: {checkpoint_path}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"GPU: Not available (using CPU)")
    
    print(f"\n--- Config Settings ---")
    print(f"Network: hidden={Config.HIDDEN_SIZE}, layers={Config.NUM_LSTM_LAYERS}, "
          f"dropout={Config.DROPOUT}, layer_norm={Config.USE_LAYER_NORM}")
    print(f"Training: lr={Config.LEARNING_RATE}, gamma={Config.GAMMA}, "
          f"entropy={Config.ENTROPY_COEF}, batch={Config.BATCH_SIZE}")
    print(f"Timeouts: max_turns={Config.MAX_TURNS}, action_gen={Config.ACTION_GEN_TIMEOUT}s, "
          f"opponent={Config.OPPONENT_TIMEOUT}s, episode={Config.MAX_EPISODE_TIME}s")
    print(f"Action Gen: mode={Config.ACTION_GEN_MODE.value}, ilp_calls={Config.MAX_ILP_CALLS}, "
          f"window={Config.MAX_WINDOW_SIZE}")
    
    print(f"\n--- Reward System v2 (No Large Constants) ---")
    print("  Intermediate:")
    print("    - Play: 2.0 × hand_change")
    print("    - Draw: 2.0 × hand_change - 5")
    print("    - Ice break: +30, Manipulation: +10")
    print("    - 4+ tiles: +5, Extension: +3")
    print("    - Large hand (20+): -2")
    print("  Terminal:")
    print("    - Win empty: +opp_hand_value")
    print("    - Lose empty: -my_hand_value")
    print("    - Pool empty: ±(hand difference)")
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
    checkpoint_interval = Config.CHECKPOINT_INTERVAL
    
    while any(p.is_alive() for p in processes):
        time.sleep(10)
        
        current_eps = stats.episodes.value
        elapsed = time.time() - start_time
        
        if current_eps >= last_checkpoint_eps + checkpoint_interval:
            checkpoint_num = current_eps // checkpoint_interval
            torch.save(global_model.state_dict(), f'checkpoint_{checkpoint_num}.pth')
            
            # Save reward plot
            rewards, terminal_rewards = stats.get_rewards_list()
            if rewards:
                save_reward_plot(rewards, terminal_rewards, 'training_rewards.png')
            
            # Save winning rate plot
            episode_results = stats.get_episode_results()
            if len(episode_results) >= 10:
                plot_winning_rate(episode_results, 'winning_rate.png', window_size=10, ma_window=50)
            
            last_checkpoint_eps = current_eps
            
            if current_eps > 0:
                eps_per_min = current_eps / (elapsed / 60)
                remaining = total_expected - current_eps
                eta_min = remaining / eps_per_min if eps_per_min > 0 else 0
                print(f"\n[MAIN] Progress: {current_eps}/{total_expected} ({100*current_eps/total_expected:.1f}%) | "
                      f"Win rate: {stats.get_win_rate():.1%} | "
                      f"ETA: {eta_min:.1f} min | Saved checkpoint_{checkpoint_num}.pth | [{get_timestamp()}]")
    
    for p in processes:
        p.join()
    
    # Final save
    torch.save(global_model.state_dict(), 'trained_agent_final.pth')
    
    rewards, terminal_rewards = stats.get_rewards_list()
    if rewards:
        save_reward_plot(rewards, terminal_rewards, 'training_rewards_final.png')
    
    # Final winning rate plot
    episode_results = stats.get_episode_results()
    if len(episode_results) >= 10:
        plot_winning_rate(episode_results, 'winning_rate_final.png', window_size=10, ma_window=50)
    
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
        env.action_generator = ActionGenerator(
            mode=Config.ACTION_GEN_MODE, 
            max_ilp_calls=Config.MAX_ILP_CALLS, 
            max_window_size=Config.MAX_WINDOW_SIZE, 
            timeout_seconds=Config.ACTION_GEN_TIMEOUT
        )
    
    opponent = RummikubILPSolver(timeout=Config.OPPONENT_TIMEOUT)
    
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
        
        with SuppressOutput():
            legal_actions = env.get_legal_actions(agent_player) if env.current_player == agent_player else []
        agent.observe(state, legal_actions)
        
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
                game_reward += reward
                
                if done:
                    # Agent was acting
                    game_reward += compute_agent_reward(agent_player, agent_player, done, info)
            else:
                action = opponent.solve(env.player_hands[env.current_player], 
                                       env.table, env.has_melded[env.current_player])
                if action is None:
                    action = RummikubAction(action_type='draw')
                state, reward, done, info = env.step(action)
                
                with SuppressOutput():
                    next_legal = env.get_legal_actions(agent_player) if env.current_player == agent_player else []
                agent.observe(state, next_legal)
                
                if done:
                    # Opponent was acting
                    game_reward += compute_agent_reward(agent_player, 1 - agent_player, done, info)
        
        total_reward += game_reward
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner is not None:
            losses += 1
        else:
            ties += 1
        
        if (game + 1) % 10 == 0:
            print(f"  {game+1}/{num_games} - W:{wins} L:{losses} T:{ties} | Avg R: {total_reward/(game+1):.1f} | [{get_timestamp()}]")
    
    print(f"\nFinal: {wins}W / {losses}L / {ties}T = {100*wins/num_games:.1f}% win rate")
    print(f"Average reward: {total_reward/num_games:.1f} | [{get_timestamp()}]")
    
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