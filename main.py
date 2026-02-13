# main.py
"""
Main Training Script for Rummikub RL Agent with A3C

REWARDS ARE COMPUTED IN Rummikub_env.py - NOT HERE!

The environment's step() function returns:
    info['reward_for_player_0'] - reward from player 0's perspective
    info['reward_for_player_1'] - reward from player 1's perspective

This script simply uses info['reward_for_player_{agent_player}']

TIMEOUTS:
    - ActionGenerator has internal timeout (timeout_seconds parameter)
    - RummikubILPSolver has internal timeout (time_limit_seconds parameter)
    - MAX_EPISODE_TIME is a safety net checked at start of each turn

Running:
    python main.py
    python main.py --checkpoint checkpoint_13.pth
    python main.py --checkpoint checkpoint_13.pth --workers 3 --episodes 1000
    python main.py --checkpoint trained_agent_final.pth --eval-only
"""

import numpy as np
import time
import multiprocessing as mp
from typing import List
import sys
import io
from datetime import datetime
import traceback

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
    CHECKPOINT_INTERVAL = 10       # Save every 10 episodes total


def get_timestamp():
    return datetime.now().strftime("%Y.%m.%d %H:%M:%S")


class TrainingStats:
    def __init__(self, manager):
        self.lock = manager.Lock()
        self.episodes = manager.Value('i', 0)
        self.agent_wins = manager.Value('i', 0)
        self.opponent_wins = manager.Value('i', 0)
        self.ties = manager.Value('i', 0)
        self.total_reward = manager.Value('d', 0.0)
        self.episode_rewards = manager.list()
        self.terminal_rewards = manager.list()  # NEW: track terminal rewards
        self.episode_lengths = manager.list()
        self.timeouts = manager.Value('i', 0)
    
    def record_episode(self, winner, agent_player, reward, terminal_reward, length, timed_out=False):
        with self.lock:
            self.episodes.value += 1
            self.total_reward.value += reward
            self.episode_rewards.append(reward)
            self.terminal_rewards.append(terminal_reward)  # NEW
            self.episode_lengths.append(length)
            if timed_out:
                self.timeouts.value += 1
            if winner == agent_player:
                self.agent_wins.value += 1
            elif winner is not None:
                self.opponent_wins.value += 1
            else:
                self.ties.value += 1
    
    def get_win_rate(self):
        with self.lock:
            return self.agent_wins.value / max(1, self.episodes.value)
    
    def get_rewards_list(self):
        with self.lock:
            return list(self.episode_rewards), list(self.terminal_rewards)
    
    def print_summary(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
            opp_wins = self.opponent_wins.value
            ties = self.ties.value
            total_r = self.total_reward.value
            lengths = list(self.episode_lengths)
            timeouts = self.timeouts.value
            term_rewards = list(self.terminal_rewards)
        
        if eps == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY ({eps} episodes)")
        print(f"{'='*60}")
        print(f"Agent wins: {wins} ({wins/eps:.1%})")
        print(f"Opponent wins: {opp_wins} ({opp_wins/eps:.1%})")
        print(f"Ties: {ties}, Timeouts: {timeouts}")
        print(f"Avg total reward: {total_r / eps:.2f}")
        if term_rewards:
            print(f"Avg terminal reward: {np.mean(term_rewards):.2f}")
        if lengths:
            print(f"Avg episode length: {np.mean(lengths):.1f} turns")


class SuppressOutput:
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._stdout, self._stderr


def worker_process(worker_id, global_model, optimizer, num_episodes, config_dict, stats):
    """Worker process for A3C training."""
    
    prefix = f"[W{worker_id}]"
    
    agent = ACAgent(global_model=global_model, optimizer=optimizer, is_worker=True, use_gpu=True)
    
    print(f"{prefix} Starting on {agent.device}... [{get_timestamp()}]")
    
    with SuppressOutput():
        env = RummikubEnv()
        # ActionGenerator handles its own timeout internally
        env.action_generator = ActionGenerator(
            mode=Config.ACTION_GEN_MODE, 
            max_ilp_calls=Config.MAX_ILP_CALLS, 
            max_window_size=Config.MAX_WINDOW_SIZE, 
            timeout_seconds=Config.ACTION_GEN_TIMEOUT
        )
    
    # RummikubILPSolver handles its own timeout internally
    opponent = RummikubILPSolver(time_limit_seconds=Config.OPPONENT_TIMEOUT)
    
    print(f"{prefix} Initialized. [{get_timestamp()}]")
    
    episode_times = []
    
    for episode in range(num_episodes):
        episode_start = time.time()
        timed_out = False
        
        # Initialize variables BEFORE try block to prevent UnboundLocalError
        agent_player = 0
        episode_reward = 0.0
        terminal_reward = 0.0
        turn_count = 0
        agent_draws = agent_plays = opp_draws = opp_plays = 0
        ice_broken_turn = -1
        
        try:
            agent.sync_local_to_global()
            agent.reset_hidden()
            
            with SuppressOutput():
                state = env.reset()
            
            done = False
            agent_player = np.random.randint(2)  # Randomize after init
            
            agent.observe(state)
            
            while not done and turn_count < Config.MAX_TURNS:
                # Safety net: check episode timeout at start of each turn
                elapsed = time.time() - episode_start
                if elapsed > Config.MAX_EPISODE_TIME:
                    print(f"{prefix} Ep {episode+1} TIMEOUT after {elapsed:.1f}s at turn {turn_count} [{get_timestamp()}]")
                    timed_out = True
                    break
                
                turn_count += 1
                current_player = env.current_player
                
                if current_player == agent_player:
                    # === AGENT'S TURN ===
                    state_vec = get_state_vec(state)
                    
                    # ActionGenerator has internal timeout
                    try:
                        with SuppressOutput():
                            legal_actions = env.get_legal_actions(agent_player)
                    except Exception as e:
                        print(f"{prefix} Error getting actions: {e}")
                        legal_actions = []
                    
                    if not legal_actions:
                        action = RummikubAction(action_type='draw')
                        action_idx = -1
                        action_vecs_list = []  # No actions available
                        num_actions = 0
                        agent_draws += 1
                    else:
                        # select_action returns ALL action vectors for proper learning
                        action, action_idx, action_vecs_list = agent.select_action(state, legal_actions)
                        num_actions = len(legal_actions)
                        if action.action_type == 'draw':
                            agent_draws += 1
                        else:
                            agent_plays += 1
                    
                    next_state, _, done, info = env.step(action)
                    
                    if info.get('ice_broken') and ice_broken_turn < 0:
                        ice_broken_turn = turn_count
                    
                    # GET REWARD FROM ENVIRONMENT
                    agent_reward = info[f'reward_for_player_{agent_player}']
                    
                    if done:
                        # terminal_reward = the FULL reward for this terminal turn
                        # For agent winning: intermediate + 300 + opp_hand (always > 300)
                        terminal_reward = agent_reward
                    
                    next_state_vec = get_state_vec(next_state) if not done else None
                    # Pass ALL action vectors for proper actor loss computation
                    agent.learn(state_vec, action_idx, action_vecs_list, agent_reward, next_state_vec, done, info, num_actions)
                    episode_reward += agent_reward
                    state = next_state
                    
                else:
                    # === OPPONENT'S TURN ===
                    state_vec = get_state_vec(state)
                    
                    # RummikubILPSolver has internal timeout
                    try:
                        action = opponent.solve(
                            env.player_hands[current_player],
                            env.table,
                            env.has_melded[current_player]
                        )
                    except Exception as e:
                        print(f"{prefix} Opponent error: {e}")
                        action = None
                    
                    if action is None:
                        action = RummikubAction(action_type='draw')
                        opp_draws += 1
                    else:
                        if action.action_type == 'draw':
                            opp_draws += 1
                        else:
                            opp_plays += 1
                    
                    next_state, _, done, info = env.step(action)
                    
                    if info.get('ice_broken') and ice_broken_turn < 0:
                        ice_broken_turn = turn_count
                    
                    # GET REWARD FROM ENVIRONMENT
                    agent_reward = info[f'reward_for_player_{agent_player}']
                    
                    if done:
                        terminal_reward = agent_reward
                        episode_reward += agent_reward
                        agent.learn(state_vec, -1, None, agent_reward, None, done, info, 0)
                    else:
                        agent.learn(state_vec, -1, None, 0, get_state_vec(next_state), done, info, 0)
                        agent.observe(next_state)
                    
                    state = next_state
            
            if turn_count >= Config.MAX_TURNS:
                timed_out = True
            
            if timed_out and not done:
                episode_reward -= 100
                terminal_reward = -100
            
        except Exception as e:
            print(f"{prefix} Ep {episode+1} ERROR: {e} [{get_timestamp()}]")
            traceback.print_exc()
            episode_reward = -200
            timed_out = True
            terminal_reward = -200
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        winner = env.winner
        stats.record_episode(winner, agent_player, episode_reward, terminal_reward, turn_count, timed_out)
        
        if winner == agent_player:
            winner_str = "AGENT WIN"
        elif winner is not None:
            winner_str = "OPP WIN"
        else:
            winner_str = "TIE/TO"
        
        ice_str = f"T{ice_broken_turn:3d}" if ice_broken_turn > 0 else "  -"
        timeout_str = " [TO]" if timed_out else ""
        
        print(f"{prefix} Ep {episode+1:3d}/{num_episodes} | {winner_str:9s} | "
              f"R:{episode_reward:7.1f} (term:{terminal_reward:7.1f}) | "
              f"T:{turn_count:3d} | A(D:{agent_draws:2d} P:{agent_plays:2d}) O(D:{opp_draws:2d} P:{opp_plays:2d}) | "
              f"Ice:{ice_str} | {episode_time:.1f}s{timeout_str} | [{get_timestamp()}]")
        
        if (episode + 1) % 25 == 0:
            win_rate = stats.get_win_rate()
            avg_time = np.mean(episode_times[-25:]) if len(episode_times) >= 25 else np.mean(episode_times)
            print(f"{prefix} === Win: {win_rate:.1%}, TO: {stats.timeouts.value}, Avg: {avg_time:.1f}s/ep === [{get_timestamp()}]")


def save_reward_plot(total_rewards, terminal_rewards, filename='training_rewards.png'):
    """
    Save a 2x2 training progress graph.
    
    Row 1: Total episode rewards (line graph + histogram)
    Row 2: Terminal rewards (line graph + histogram)
    """
    if len(total_rewards) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calculate moving average window
    window = min(50, len(total_rewards) // 4) if len(total_rewards) > 10 else max(1, len(total_rewards))
    
    # =========================================
    # Row 1, Column 1: Total Reward Line Graph
    # =========================================
    ax = axes[0, 0]
    ax.plot(total_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if window > 1:
        ma = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(total_rewards)), ma, 'b-', lw=2, label=f'{window}-ep Moving Avg')
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Episode Rewards')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # =========================================
    # Row 1, Column 2: Total Reward Histogram
    # =========================================
    ax = axes[0, 1]
    ax.hist(total_rewards, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(0, color='black', ls='--', lw=2, label='Zero')
    mean_total = np.mean(total_rewards)
    ax.axvline(mean_total, color='red', ls='-', lw=2, label=f'Mean: {mean_total:.1f}')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Reward Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # =========================================
    # Row 2, Column 1: Terminal Reward Line Graph
    # =========================================
    ax = axes[1, 0]
    ax.plot(terminal_rewards, alpha=0.3, color='green', label='Terminal Reward')
    if window > 1 and len(terminal_rewards) >= window:
        ma = np.convolve(terminal_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(terminal_rewards)), ma, 'g-', lw=2, label=f'{window}-ep Moving Avg')
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.axhline(300, color='blue', ls=':', alpha=0.5, label='Win threshold (300)')
    ax.axhline(-200, color='red', ls=':', alpha=0.5, label='Lose threshold (-200)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Terminal Reward')
    ax.set_title('Terminal Rewards (Win: >300, Lose: <-200)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # =========================================
    # Row 2, Column 2: Terminal Reward Histogram
    # =========================================
    ax = axes[1, 1]
    ax.hist(terminal_rewards, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(0, color='black', ls='--', lw=2, label='Zero')
    mean_term = np.mean(terminal_rewards)
    ax.axvline(mean_term, color='red', ls='-', lw=2, label=f'Mean: {mean_term:.1f}')
    ax.set_xlabel('Terminal Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Terminal Reward Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_a3c(num_workers=4, num_episodes_per_worker=500, checkpoint_path=None):
    manager = mp.Manager()
    stats = TrainingStats(manager)
    
    global_model = ActorCritic(
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        dropout=Config.DROPOUT,
        use_layer_norm=Config.USE_LAYER_NORM
    )
    
    if checkpoint_path:
        try:
            global_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
            print(f"Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Unexpected key" in str(e):
                print(f"\n{'='*60}")
                print("WARNING: Checkpoint architecture mismatch!")
                print("The checkpoint was trained with a different model architecture.")
                print("Starting with FRESH weights (checkpoint ignored).")
                print("Delete old checkpoints before training.")
                print(f"{'='*60}\n")
            else:
                print(f"Could not load checkpoint: {e}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    global_model.share_memory()
    
    optimizer = optim.Adam(global_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    print(f"\n{'='*60}")
    print(f"A3C TRAINING [{get_timestamp()}]")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}, Episodes/worker: {num_episodes_per_worker}")
    print(f"Total episodes: {num_workers * num_episodes_per_worker}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"\nTimeouts (internal to solvers):")
    print(f"  ActionGenerator: {Config.ACTION_GEN_TIMEOUT}s")
    print(f"  RummikubILPSolver: {Config.OPPONENT_TIMEOUT}s")
    print(f"  Max episode (safety): {Config.MAX_EPISODE_TIME}s ({Config.MAX_EPISODE_TIME/60:.1f} min)")
    print(f"\nReward config (from RummikubEnv):")
    print(f"  LOSE_EMPTY_HAND = {RummikubEnv.REWARD_LOSE_EMPTY_HAND}")
    print(f"  When opp wins: Agent gets {RummikubEnv.REWARD_LOSE_EMPTY_HAND} - agent_hand (ALWAYS < -200)")
    print(f"\nCheckpoint: Every {Config.CHECKPOINT_INTERVAL} episodes total")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    processes = []
    for wid in range(num_workers):
        p = mp.Process(target=worker_process, args=(wid, global_model, optimizer, num_episodes_per_worker, {}, stats))
        p.start()
        processes.append(p)
    
    total_expected = num_workers * num_episodes_per_worker
    last_ckpt = 0
    ckpt_interval = Config.CHECKPOINT_INTERVAL
    
    while any(p.is_alive() for p in processes):
        time.sleep(5)
        
        current_eps = stats.episodes.value
        
        if current_eps >= last_ckpt + ckpt_interval:
            ckpt_num = current_eps // ckpt_interval
            ckpt_filename = f'checkpoint_{ckpt_num}.pth'
            torch.save(global_model.state_dict(), ckpt_filename)
            
            total_rewards, terminal_rewards = stats.get_rewards_list()
            if total_rewards:
                save_reward_plot(total_rewards, terminal_rewards)
                pos = sum(1 for r in total_rewards if r > 0) / len(total_rewards)
                neg = sum(1 for r in total_rewards if r < 0) / len(total_rewards)
                avg_term = np.mean(terminal_rewards) if terminal_rewards else 0
            else:
                pos = neg = avg_term = 0
            
            last_ckpt = current_eps
            
            elapsed = time.time() - start_time
            eps_per_min = current_eps / (elapsed / 60) if elapsed > 0 else 0
            eta = (total_expected - current_eps) / eps_per_min if eps_per_min > 0 else 0
            
            print(f"\n[MAIN] {current_eps}/{total_expected} | Win: {stats.get_win_rate():.1%} | "
                  f"Pos/Neg: {pos:.1%}/{neg:.1%} | AvgTerm: {avg_term:.1f} | TO: {stats.timeouts.value} | "
                  f"ETA: {eta:.1f}m | Saved: {ckpt_filename} [{get_timestamp()}]")
    
    for p in processes:
        p.join()
    
    torch.save(global_model.state_dict(), 'trained_agent_final.pth')
    
    total_rewards, terminal_rewards = stats.get_rewards_list()
    if total_rewards:
        save_reward_plot(total_rewards, terminal_rewards, 'training_rewards_final.png')
    
    print(f"\n{'='*60}")
    print(f"COMPLETE [{get_timestamp()}] - {(time.time() - start_time)/60:.1f} min")
    stats.print_summary()
    
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    # Create ACAgent with matching architecture for loading checkpoint
    agent = ACAgent(
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LSTM_LAYERS,
        dropout=Config.DROPOUT,
        use_layer_norm=Config.USE_LAYER_NORM
    )
    agent.load('trained_agent_final.pth')
    evaluate_agent(agent, 50)


def evaluate_agent(agent, num_games=50):
    with SuppressOutput():
        env = RummikubEnv()
        env.action_generator = ActionGenerator(
            mode=Config.ACTION_GEN_MODE, 
            max_ilp_calls=Config.MAX_ILP_CALLS, 
            max_window_size=Config.MAX_WINDOW_SIZE, 
            timeout_seconds=Config.ACTION_GEN_TIMEOUT
        )
    
    opponent = RummikubILPSolver(time_limit_seconds=Config.OPPONENT_TIMEOUT)
    
    wins = losses = ties = 0
    total_reward = 0
    positive = negative = 0
    
    print(f"Evaluating {num_games} games... [{get_timestamp()}]")
    
    for game in range(num_games):
        with SuppressOutput():
            state = env.reset()
        
        done = False
        agent_player = 0
        game_reward = 0
        
        agent.reset_hidden()
        agent.observe(state)
        
        turn = 0
        game_start = time.time()
        
        while not done and turn < Config.MAX_TURNS:
            if time.time() - game_start > Config.MAX_EPISODE_TIME:
                break
            
            turn += 1
            if env.current_player == agent_player:
                try:
                    with SuppressOutput():
                        legal = env.get_legal_actions(agent_player)
                except:
                    legal = []
                
                action = agent.select_action(state, legal)[0] if legal else RummikubAction(action_type='draw')
                state, _, done, info = env.step(action)
                game_reward += info[f'reward_for_player_{agent_player}']
            else:
                try:
                    action = opponent.solve(env.player_hands[env.current_player], env.table, env.has_melded[env.current_player])
                except:
                    action = None
                
                if action is None:
                    action = RummikubAction(action_type='draw')
                state, _, done, info = env.step(action)
                game_reward += info[f'reward_for_player_{agent_player}']
                agent.observe(state)
        
        total_reward += game_reward
        
        if game_reward > 0:
            positive += 1
        elif game_reward < 0:
            negative += 1
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner is not None:
            losses += 1
        else:
            ties += 1
        
        if (game + 1) % 10 == 0:
            print(f"  {game+1}/{num_games} - W:{wins} L:{losses} T:{ties} | "
                  f"Avg: {total_reward/(game+1):.1f} | +/-: {positive}/{negative}")
    
    print(f"\nFinal: {wins}W/{losses}L/{ties}T = {100*wins/num_games:.1f}% win")
    print(f"Avg reward: {total_reward/num_games:.1f}")
    print(f"Positive/Negative: {positive}/{negative}")
    print(f"Verification: Wins should have positive, Losses should have negative")
    
    return {'wins': wins, 'losses': losses, 'win_rate': wins/num_games}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='A3C Training for Rummikub')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--episodes', '-e', type=int, default=500,
                        help='Episodes per worker (default: 500)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate, no training')
    
    args = parser.parse_args()
    
    if args.eval_only:
        if not args.checkpoint:
            print("--eval-only requires --checkpoint")
            return
        print(f"Evaluating: {args.checkpoint} [{get_timestamp()}]")
        # Create ACAgent with matching architecture for loading checkpoint
        agent = ACAgent(
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LSTM_LAYERS,
            dropout=Config.DROPOUT,
            use_layer_norm=Config.USE_LAYER_NORM
        )
        agent.load(args.checkpoint)
        evaluate_agent(agent, 100)
    else:
        train_a3c(args.workers, args.episodes, args.checkpoint)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()