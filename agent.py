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
        input_lstm = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,110]
        out, new_hidden = self.lstm(input_lstm, hidden)
        out = out.squeeze(0).squeeze(0)
        value = self.critic_head(out)
        if action_vecs is None:
            return value, new_hidden
        action_inputs = [torch.cat((out, torch.tensor(a_vec, dtype=torch.float32))) for a_vec in action_vecs]
        logits = torch.stack([self.actor_head(a_input) for a_input in action_inputs]).squeeze(1)
        return value, logits, new_hidden

class ACAgent:
    def __init__(self):
        self.net = ActorCritic()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
        self.name = "ACAgent"
        self.last_value = None
        self.last_log_prob = None
        self.last_logits = None
        self.last_next_value = None

    def select_action(self, state: Dict, legal_actions: List[RummikubAction]) -> RummikubAction:
        state_vec = get_state_vec(state)
        action_vecs = [get_action_vec(a) for a in legal_actions]
        with torch.no_grad():
            value, logits, new_hidden = self.net(state_vec, action_vecs, self.hidden)
        self.hidden = new_hidden
        self.last_value = value
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        idx = np.random.choice(len(legal_actions), p=probs)
        self.last_log_prob = torch.log_softmax(logits, dim=0)[idx]
        self.last_logits = logits
        return legal_actions[idx]

    def learn(self, state, action, reward, next_state, done, info):
        if next_state is not None:
            next_state_vec = get_state_vec(next_state)
            with torch.no_grad():
                next_value, new_hidden = self.net(next_state_vec, None, self.hidden)
            self.last_next_value = next_value
        else:
            next_value = torch.tensor(0.0)

        self.hidden = new_hidden if next_state is not None else self.hidden

        if state is None:
            # Opponent turn
            reward = 0.0
            if done:
                my_hand_value = sum(t.get_value() for t in info.get('final_my_hand_value', next_state['my_hand'] if next_state else info.get('hand_value_after', 0)))
                if 'win_type' in info:
                    if info['win_type'] == 'emptied_hand':
                        reward = -my_hand_value
                    elif info['win_type'] == 'lowest_hand':
                        if info['winner'] == next_state['current_player'] if next_state else self.current_player:  # Need to know agent_player, but assume from info
                            # To make it work, assume 'winner' == 0 for agent (since agent_player random, but for simplicity, perhaps set agent_player=0 in main
                            # For now, use if info['winner'] == next_state['current_player'] but since last actor is opp, if winner = opp, reward = -10 for agent
                            reward = -10 if info['winner'] == 1 - next_state['current_player'] else 10  # flip
                    elif info['win_type'] == 'tie':
                        reward = 0
        # Joker penalty
        if done:
            if next_state is None:
                jokers = 0
            else:
                jokers = sum(
                    1
                    for t in next_state["my_hand"]
                    if t.tile_type == TileType.JOKER
                )

            reward += -30 * jokers

        target = torch.tensor(reward) if done else torch.tensor(reward) + 0.99 * next_value

        if state is None:
            advantage = target - self.last_next_value
            actor_loss = torch.tensor(0.0)
            critic_loss = F.mse_loss(self.last_next_value, target)
            entropy = torch.tensor(0.0)
        else:
            advantage = target - self.last_value
            actor_loss = -self.last_log_prob * advantage.detach()
            critic_loss = F.mse_loss(self.last_value, target)
            entropy = -torch.sum(torch.softmax(self.last_logits, dim=0) * torch.log_softmax(self.last_logits, dim=0))

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))

    def observe(self, state):
        state_vec = get_state_vec(state)
        with torch.no_grad():
            _, new_hidden = self.net(state_vec, None, self.hidden)
        self.hidden = new_hidden

    def save(self, path):
        torch.save(self.net.state_dict(), path)