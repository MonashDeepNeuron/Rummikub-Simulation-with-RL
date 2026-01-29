# main.py
"""
Main Training Script for Rummikub RL Agent with A3C
"""

import numpy as np
import time
import multiprocessing as mp
from typing import List
import pickle
import sys
import io

from Rummikub_env import RummikubEnv, RummikubAction
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent2 import RummikubILPSolver
from agent import ACAgent, ActorCritic, get_state_vec, get_action_vec

import torch
import torch.optim as optim


class TrainingStats:
    """Track training statistics across workers."""
    
    def __init__(self, manager):
        # Use manager.Lock() for explicit synchronization (Python 3.13 compatible)
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
            elif winner == 1 - agent_player:
                self.opponent_wins.value += 1
            else:
                self.ties.value += 1
    
    def get_win_rate(self):
        with self.lock:
            eps = self.episodes.value
            wins = self.agent_wins.value
        return wins / eps if eps > 0 else 0.0
    
    def print_summary(self, last_n=100):
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
        print(f"Opponent wins: {opp_wins}")
        print(f"Ties: {ties}")
        print(f"Avg reward: {total_r / eps:.2f}")
        
        if len(lengths) > 0:
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


def worker_process(worker_id, global_model, optimizer, num_episodes, config, stats):
    """Worker process for A3C training."""
    
    # Prefix for all prints from this worker
    prefix = f"[W{worker_id}]"
    
    print(f"{prefix} Starting worker...")
    
    # Create agent (CPU only for A3C)
    agent = ACAgent(global_model=global_model, optimizer=optimizer, is_worker=True)
    
    # Create environment with suppressed output
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
    
    for episode in range(num_episodes):
        # Sync and reset
        agent.sync_local_to_global()
        agent.reset_hidden()
        
        # Reset environment (suppress debug prints)
        with SuppressOutput():
            state = env.reset()
        
        done = False
        episode_reward = 0
        turn_count = 0
        agent_player = np.random.randint(2)
        
        # Track actions
        agent_draws = 0
        agent_plays = 0
        opp_draws = 0
        opp_plays = 0
        ice_broken_turn = -1
        
        agent.observe(state)
        
        while not done:
            turn_count += 1
            current_player = env.current_player
            
            if current_player == agent_player:
                # === AGENT'S TURN ===
                state_vec = get_state_vec(state)
                
                # Suppress action generator prints
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
                
                next_state, reward, done, info = env.step(action)
                
                if info.get('ice_broken') and ice_broken_turn < 0:
                    ice_broken_turn = turn_count
                
                next_state_vec = get_state_vec(next_state) if not done else None
                
                agent.learn(state_vec, action_idx, action_vec, reward, next_state_vec, done, info, num_actions)
                episode_reward += reward
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
                
                next_state, reward_opp, done, info = env.step(action)
                
                if info.get('ice_broken') and ice_broken_turn < 0:
                    ice_broken_turn = turn_count
                
                next_state_vec = get_state_vec(next_state) if not done else None
                
                agent.learn(state_vec, -1, None, -reward_opp, next_state_vec, done, info, 0)
                
                if not done:
                    agent.observe(next_state)
                
                state = next_state
        
        # Record stats
        stats.record_episode(env.winner, agent_player, episode_reward, turn_count)
        
        # Determine winner
        if env.winner == agent_player:
            winner_str = "AGENT WIN"
        elif env.winner is not None:
            winner_str = "OPP WIN"
        else:
            winner_str = "TIE"
        
        # Print episode summary
        print(f"{prefix} Ep {episode+1:3d}/{num_episodes} | {winner_str:9s} | "
              f"R:{episode_reward:6.1f} | T:{turn_count:3d} | "
              f"Agent(D:{agent_draws:2d} P:{agent_plays:2d}) Opp(D:{opp_draws:2d} P:{opp_plays:2d}) | "
              f"Ice:T{ice_broken_turn if ice_broken_turn > 0 else '-':>3}")
        
        # Detailed stats every 25 episodes
        if (episode + 1) % 25 == 0:
            win_rate = stats.get_win_rate()
            print(f"{prefix} === Global stats: {stats.episodes.value} eps, {win_rate:.1%} win rate ===")


def train_a3c(num_workers=4, num_episodes_per_worker=500, config=None):
    """Main A3C training function."""
    
    if config is None:
        config = {
            'action_gen_mode': SolverMode.HYBRID,
        }
    
    manager = mp.Manager()
    stats = TrainingStats(manager)
    
    # Create global model (CPU for shared memory)
    global_model = ActorCritic()
    global_model.share_memory()
    
    optimizer = optim.Adam(global_model.parameters(), lr=0.0001)
    
    print(f"\n{'='*60}")
    print("A3C TRAINING - RUMMIKUB")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}")
    print(f"Episodes per worker: {num_episodes_per_worker}")
    print(f"Total episodes: {num_workers * num_episodes_per_worker}")
    print(f"Device: CPU (required for A3C shared memory)")
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
    
    # Monitor
    total_expected = num_workers * num_episodes_per_worker
    checkpoint_count = 0
    
    while any(p.is_alive() for p in processes):
        time.sleep(30)
        
        # Save checkpoint
        torch.save(global_model.state_dict(), f'checkpoint_{checkpoint_count}.pth')
        checkpoint_count += 1
        
        # Progress update
        current = stats.episodes.value
        elapsed = time.time() - start_time
        if current > 0:
            eps_per_min = current / (elapsed / 60)
            remaining = total_expected - current
            eta_min = remaining / eps_per_min if eps_per_min > 0 else 0
            print(f"\n[MAIN] Progress: {current}/{total_expected} ({100*current/total_expected:.1f}%) | "
                  f"Win rate: {stats.get_win_rate():.1%} | "
                  f"ETA: {eta_min:.1f} min")
    
    for p in processes:
        p.join()
    
    # Final save
    torch.save(global_model.state_dict(), 'trained_agent_final.pth')
    
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
    
    print(f"Evaluating over {num_games} games...")
    
    for game in range(num_games):
        with SuppressOutput():
            state = env.reset()
        
        done = False
        agent_player = 0
        agent.reset_hidden()
        agent.observe(state)
        
        while not done:
            if env.current_player == agent_player:
                with SuppressOutput():
                    legal_actions = env.get_legal_actions(agent_player)
                if not legal_actions:
                    action = RummikubAction(action_type='draw')
                else:
                    action, _, _ = agent.select_action(state, legal_actions)
                state, _, done, _ = env.step(action)
            else:
                action = opponent.solve(env.player_hands[env.current_player], env.table, env.has_melded[env.current_player])
                if action is None:
                    action = RummikubAction(action_type='draw')
                state, _, done, _ = env.step(action)
                agent.observe(state)
        
        if env.winner == agent_player:
            wins += 1
        elif env.winner is not None:
            losses += 1
        else:
            ties += 1
        
        if (game + 1) % 10 == 0:
            print(f"  {game+1}/{num_games} - W:{wins} L:{losses} T:{ties}")
    
    print(f"\nFinal: {wins}W / {losses}L / {ties}T = {100*wins/num_games:.1f}% win rate")
    
    return {'wins': wins, 'losses': losses, 'ties': ties, 'win_rate': wins/num_games}


def main():
    train_a3c(num_workers=4, num_episodes_per_worker=500)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()