# basic_strategy_bot.py
from __future__ import annotations
import gymnasium as gym
import numpy as np

STAND = 0
HIT = 1

def basic_strategy_action(state: tuple[int, int, bool]) -> int:
    """
    Deterministic basic-strategy policy for Single-Deck, S17 (dealer stands on soft 17),
    translated to the limited action set of Gym's Blackjack-v1 (hit / stick only).

    state = (player_sum, dealer_upcard, usable_ace)
    """
    player, dealer, usable_ace = state

    # Soft totals (user has ace)
    if usable_ace:

        if player in (12, 13, 14, 15, 16, 17):
            return HIT

        if player == 18:
            if dealer in (9, 10):
                return HIT
            else:
                return STAND
            
        if player >= 19:
            return STAND

    # Hard totals (no ace)
    else:
        if player in (4, 5, 6, 7, 8, 9, 10, 11):
            return HIT
        
        if player == 12:
            if dealer in (2, 3) or dealer in (7, 8, 9, 10, 11):
                return HIT
            else: 
                return STAND
            
        if player in (13, 14, 15, 16):
            if dealer in (7, 8, 9 , 10, 11):
                return HIT
            else: 
                return STAND
            
        if player >= 17:
            return STAND
                   
class BasicStrategyAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def play(self, num_games):
        results = []
        for _ in range(num_games):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = basic_strategy_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            results.append(total_reward)

        wins = results.count(1.0)
        losses = results.count(-1.0)
        pushes = results.count(0.0)
        avg = float(np.mean(results))
        return {"avg_reward": avg, "wins": wins, "losses": losses, "pushes": pushes}

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    agent = BasicStrategyAgent(env)
    stats = agent.play(num_games=100)
    print(
        f"Avg reward: {stats['avg_reward']:.4f} "
        f"(W:{stats['wins']} L:{stats['losses']} P:{stats['pushes']})"
    )
