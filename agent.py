# ac_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict
from Rummikub_env import RummikubAction, TileType, Color, RummikubEnv

def get_state_vec(state: Dict) -> np.ndarray:
    hand_counts = np.zeros(53)
    for t in state['my_hand']:
        if t.tile_type == TileType.JOKER:
            hand_counts[52] += 1
        else:
            index = t.color.value * 13 + (t.number - 1)
            hand_counts[index] += 1
    table_counts = np.zeros(53)
    for s in state['table']:
        for t in s.tiles:
            if t.tile_type == TileType.JOKER:
                table_counts[52] += 1
            else:
                index = t.color.value * 13 + (t.number - 1)
                table_counts[index] += 1
    opp_count = state['opponent_tile_count'] / 30.0
    pool_size = state['pool_size'] / 80.0
    has_melded = 1.0 if state['has_melded'][state['current_player']] else 0.0
    opp_has_melded = 1.0 if state['has_melded'][1 - state['current_player']] else 0.0
    vec = np.concatenate((hand_counts, table_counts, [opp_count, pool_size, has_melded, opp_has_melded]))
    return vec

def get_action_vec(action: RummikubAction) -> np.ndarray:
    played_counts = np.zeros(53)
    for t in action.tiles:
        if t.tile_type == TileType.JOKER:
            played_counts[52] += 1
        else:
            index = t.color.value * 13 + (t.number - 1)
            played_counts[index] += 1
    flag_draw = 1.0 if action.action_type == 'draw' else 0.0
    return np.concatenate((played_counts, [flag_draw]))

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.lstm = nn.LSTM(110, 128, batch_first=False)
        self.actor_head = nn.Sequential(
            nn.Linear(128 + 54, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_vec, action_vecs, hidden):
        input_lstm = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out, new_hidden = self.lstm(input_lstm, hidden)
        out = out.squeeze(0).squeeze(0)           # shape [128]

        value = self.critic_head(out).squeeze(-1)  # shape []
        
        if action_vecs is None:
            return value, new_hidden
        
        action_inputs = [torch.cat((out, torch.tensor(a_vec, dtype=torch.float32))) 
                        for a_vec in action_vecs]
        logits = torch.stack([self.actor_head(a_input) for a_input in action_inputs]).squeeze(1)
        return value, logits, new_hidden



class ACAgent:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))

        self.name = "ACAgent"

        # Buffers for last transition (agent's turn)
        self.last_value      = None
        self.last_log_prob   = None
        self.last_logits     = None

        # For opponent's turn value estimation
        self.last_opponent_value = None

        # Safety / reset flags
        self.last_state_value    = None   # mostly unused now

    def pre_opponent_turn(self, state: Dict):
        """Called just before opponent acts → capture value estimate"""
        state_vec = get_state_vec(state)
        with torch.no_grad():
            value, new_hidden = self.net(state_vec, None, self.hidden)
        self.last_opponent_value = value
        self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())

    def select_action(self, state: Dict, legal_actions: List[RummikubAction]) -> RummikubAction:
        if not legal_actions:
            return RummikubAction(action_type='draw') #Instead of raise, fallback to draw (safety net)

        state_vec = get_state_vec(state)
        action_vecs = [get_action_vec(a) for a in legal_actions]

        value, logits, new_hidden = self.net(state_vec, action_vecs, self.hidden)

        # Detach hidden to prevent long-term graph accumulation
        self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())

        # Sample
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()

        # Save for learning
        self.last_value     = value
        self.last_log_prob  = dist.log_prob(idx)
        self.last_logits    = logits

        return legal_actions[int(idx.item())]

    def learn(self, state, action, reward, next_state, done, info):
        """Improved learn method - handles forced draw case safely"""
        reward = float(reward)

        # Next state value estimation
        if next_state is not None:
            next_state_vec = get_state_vec(next_state)
            with torch.no_grad():
                next_value, new_hidden = self.net(next_state_vec, None, self.hidden)
            self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        else:
            next_value = torch.tensor(0.0, dtype=torch.float32)

        # Joker penalty + terminal shaping (only if agent lost the episode)
        if done and next_state is not None:
            jokers = sum(1 for t in next_state.get("my_hand", []) 
                        if getattr(t, 'tile_type', None) == TileType.JOKER)
            lost = False
            if 'win_type' in info:
                if info['win_type'] == 'emptied_hand':
                    if len(next_state.get("my_hand", [])) > 0:
                        # Opponent emptied hand → agent lost
                        lost = True
                elif info['win_type'] == 'lowest_hand':
                    my_value = info.get('final_my_hand_value', 0)
                    opp_value = info.get('final_opponent_hand_value', 0)
                    if my_value > opp_value:
                        # Agent has higher (worse) hand value → lost
                        lost = True
                # For 'tie', lost=False
            if lost:
                reward -= 30.0 * jokers  # Extra penalty only on loss

        target = torch.tensor(reward, dtype=torch.float32)
        if not done:
            target += 0.99 * next_value

        if state is not None and self.last_value is not None:
            # Normal agent's turn (policy + critic update)
            advantage = target - self.last_value
            actor_loss = -self.last_log_prob * advantage.detach()
            critic_loss = F.mse_loss(self.last_value, target)
            entropy = -torch.sum(F.softmax(self.last_logits, dim=0) * 
                                F.log_softmax(self.last_logits, dim=0))
        else:
            # Opponent's turn OR forced draw → only critic update, no actor loss
            actor_loss = torch.tensor(0.0)
            if self.last_opponent_value is not None:
                critic_loss = F.mse_loss(self.last_opponent_value, target)
            else:
                critic_loss = torch.tensor(0.0)
            entropy = torch.tensor(0.0)

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # Added gradient clipping
            self.optimizer.step()

        # Reset memory at end of episode
        if done:
            self.hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
            self.last_value = None
            self.last_log_prob = None
            self.last_logits = None
            self.last_opponent_value = None

    def observe(self, state: Dict):
        """Optional: update hidden state without action selection"""
        state_vec = get_state_vec(state)
        with torch.no_grad():
            _, new_hidden = self.net(state_vec, None, self.hidden)
        self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()