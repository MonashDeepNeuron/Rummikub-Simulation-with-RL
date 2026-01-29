import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from Rummikub_env import RummikubEnv, RummikubAction, TileType, Color
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent2 import RummikubILPSolver

# Store numpy arrays to avoid gradient issues
Transition = namedtuple('Transition', (
    'state_vec',      # numpy array
    'action_idx',     # int or None
    'action_vec',     # numpy array or None  
    'reward',         # float
    'next_state_vec', # numpy array or None
    'done',           # bool
    'info',           # dict
    'num_actions'     # int
))


def get_state_vec(state: Dict) -> np.ndarray:
    """Convert game state to feature vector."""
    hand = state['my_hand']
    if len(hand) == 0:
        hand_counts = np.zeros(53, dtype=np.float32)
    else:
        hand_types = np.array([t.tile_type.value for t in hand], dtype=np.int64)
        hand_colors = np.array([t.color.value if t.color else 99 for t in hand], dtype=np.int64)
        hand_numbers = np.array([t.number if t.number else 99 for t in hand], dtype=np.int64)
        
        hand_counts = np.zeros(53, dtype=np.float32)
        joker_mask = hand_types == TileType.JOKER.value
        hand_counts[52] = np.sum(joker_mask)
        if np.any(~joker_mask):
            non_joker_indices = (hand_colors * 13 + (hand_numbers - 1))[~joker_mask]
            np.add.at(hand_counts, non_joker_indices, 1)
    
    all_table_tiles = []
    for s in state['table']:
        all_table_tiles.extend(s.tiles)
    
    if len(all_table_tiles) == 0:
        table_counts = np.zeros(53, dtype=np.float32)
    else:
        table_types = np.array([t.tile_type.value for t in all_table_tiles], dtype=np.int64)
        table_colors = np.array([t.color.value if t.color else 99 for t in all_table_tiles], dtype=np.int64)
        table_numbers = np.array([t.number if t.number else 99 for t in all_table_tiles], dtype=np.int64)
        
        table_counts = np.zeros(53, dtype=np.float32)
        table_joker_mask = table_types == TileType.JOKER.value
        table_counts[52] = np.sum(table_joker_mask)
        if np.any(~table_joker_mask):
            table_non_joker_indices = (table_colors * 13 + (table_numbers - 1))[~table_joker_mask]
            np.add.at(table_counts, table_non_joker_indices, 1)
    
    opp_count = state['opponent_tile_count'] / 30.0
    pool_size = state['pool_size'] / 80.0
    has_melded = 1.0 if state['has_melded'][state['current_player']] else 0.0
    opp_has_melded = 1.0 if state['has_melded'][1 - state['current_player']] else 0.0
    
    vec = np.concatenate((hand_counts, table_counts, [opp_count, pool_size, has_melded, opp_has_melded]))
    return vec.astype(np.float32)


def get_action_vec(action: RummikubAction) -> np.ndarray:
    """Convert action to feature vector."""
    tiles = action.tiles if action.tiles else []
    if len(tiles) == 0:
        played_counts = np.zeros(53, dtype=np.float32)
    else:
        types = np.array([t.tile_type == TileType.JOKER for t in tiles])
        colors = np.array([t.color.value if t.color else 99 for t in tiles], dtype=np.int64)
        numbers = np.array([t.number if t.number else 99 for t in tiles], dtype=np.int64)
        
        played_counts = np.zeros(53, dtype=np.float32)
        joker_mask = types
        played_counts[52] = np.sum(joker_mask)
        if np.any(~joker_mask):
            non_joker_indices = (colors * 13 + (numbers - 1))[~joker_mask]
            np.add.at(played_counts, non_joker_indices, 1)
    
    flag_draw = 1.0 if action.action_type == 'draw' else 0.0
    return np.concatenate((played_counts, [flag_draw])).astype(np.float32)


class ActorCritic(nn.Module):
    """Actor-Critic network with LSTM."""
    
    def __init__(self, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(110, hidden_size, batch_first=True)
        
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + 54, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state_vecs, action_vecs_list=None, hiddens=None):
        batch_size = state_vecs.size(0)
        
        if hiddens is None:
            hiddens = (torch.zeros(1, batch_size, self.hidden_size, device=state_vecs.device),
                       torch.zeros(1, batch_size, self.hidden_size, device=state_vecs.device))
        
        out, new_hiddens = self.lstm(state_vecs, hiddens)
        out = out[:, -1, :]
        
        values = self.critic_head(out).squeeze(-1)
        
        if action_vecs_list is None:
            return values, new_hiddens
        
        logits_list = []
        for b in range(batch_size):
            if action_vecs_list[b] is None or len(action_vecs_list[b]) == 0:
                logits_list.append(None)
                continue
            
            if isinstance(action_vecs_list[b], list):
                action_vecs_stacked = torch.stack(action_vecs_list[b])
            else:
                action_vecs_stacked = action_vecs_list[b]
            
            num_actions = action_vecs_stacked.size(0)
            state_repeated = out[b].unsqueeze(0).expand(num_actions, -1)
            action_inputs = torch.cat([state_repeated, action_vecs_stacked], dim=1)
            
            logits = self.actor_head(action_inputs).squeeze(-1)
            logits_list.append(logits)
            
        return values, logits_list, new_hiddens


class ACAgent:
    """A3C Agent - Uses GPU for computation, CPU for shared global model."""
    
    def __init__(self, global_model=None, optimizer=None, is_worker=False, use_gpu=True):
        # Use GPU for local computation if available
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.local_net = ActorCritic().to(self.device)
        self.global_model = global_model  # Always on CPU for shared memory
        self.optimizer = optimizer
        self.is_worker = is_worker
        
        self.hidden = None
        self.reset_hidden()
        
        self.name = "ACAgent"
        self.buffer: List[Transition] = []
        self.batch_size = 64  # Larger batch for GPU efficiency
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        if not is_worker and optimizer is None:
            self.optimizer = optim.Adam(self.local_net.parameters(), lr=0.001)

    def reset_hidden(self):
        self.hidden = (
            torch.zeros(1, 1, self.local_net.hidden_size, device=self.device),
            torch.zeros(1, 1, self.local_net.hidden_size, device=self.device)
        )

    def sync_local_to_global(self):
        """Copy global model (CPU) weights to local model (GPU/CPU)."""
        if self.global_model is not None:
            # Load CPU state dict to local device
            state_dict = self.global_model.state_dict()
            # Move to local device
            local_state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            self.local_net.load_state_dict(local_state_dict)

    def select_action(self, state: Dict, legal_actions: List[RummikubAction]) -> Tuple[RummikubAction, int, List[np.ndarray]]:
        """Select action. Returns (action, action_index, action_vecs_numpy)."""
        if not legal_actions:
            return RummikubAction(action_type='draw'), -1, []
        
        state_vec_np = get_state_vec(state)
        state_vec = torch.from_numpy(state_vec_np).to(self.device).unsqueeze(0).unsqueeze(0)
        
        action_vecs_np = [get_action_vec(a) for a in legal_actions]
        action_vecs = [torch.from_numpy(av).to(self.device) for av in action_vecs_np]
        
        with torch.no_grad():
            _, logits_list, new_hidden = self.local_net(state_vec, [action_vecs], self.hidden)
            self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        
        logits = logits_list[0]
        
        if logits is None or logits.numel() == 0:
            return RummikubAction(action_type='draw'), -1, []
        
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample().item()
        
        return legal_actions[idx], idx, action_vecs_np

    def store_transition(self, state_vec, action_idx, action_vec, reward, next_state_vec, done, info, num_actions):
        trans = Transition(state_vec, action_idx, action_vec, reward, next_state_vec, done, info, num_actions)
        self.buffer.append(trans)

    def learn(self, state_vec, action_idx, action_vec, reward, next_state_vec, done, info, num_actions):
        self.store_transition(state_vec, action_idx, action_vec, reward, next_state_vec, done, info, num_actions)
        
        if len(self.buffer) >= self.batch_size or done:
            self._update_global()

    def _update_global(self):
        if not self.buffer or self.global_model is None:
            self.buffer = []
            return
        
        self.sync_local_to_global()
        
        batch_size = len(self.buffer)
        
        state_vecs = np.stack([t.state_vec for t in self.buffer])
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float32, device=self.device)
        
        state_vecs_t = torch.from_numpy(state_vecs).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            values_detached, _ = self.local_net(state_vecs_t, None)
        
        next_values = torch.zeros(batch_size, device=self.device)
        for i, trans in enumerate(self.buffer):
            if trans.next_state_vec is not None and not trans.done:
                next_state_t = torch.from_numpy(trans.next_state_vec).to(self.device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    nv, _ = self.local_net(next_state_t, None)
                    next_values[i] = nv.squeeze()
        
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        values_with_grad, _ = self.local_net(state_vecs_t, None)
        
        advantages = (targets - values_detached).detach()
        
        actor_loss = torch.tensor(0.0, device=self.device)
        num_actor_samples = 0
        
        for i, trans in enumerate(self.buffer):
            if trans.action_idx is None or trans.action_idx < 0 or trans.action_vec is None:
                continue
            if trans.num_actions <= 0:
                continue
            
            state_t = torch.from_numpy(trans.state_vec).to(self.device).unsqueeze(0).unsqueeze(0)
            action_t = torch.from_numpy(trans.action_vec).to(self.device).unsqueeze(0)
            
            temp_hidden = (torch.zeros(1, 1, self.local_net.hidden_size, device=self.device),
                          torch.zeros(1, 1, self.local_net.hidden_size, device=self.device))
            
            _, logits_list, _ = self.local_net(state_t, [[action_t.squeeze(0)]], temp_hidden)
            
            if logits_list[0] is not None and logits_list[0].numel() > 0:
                log_prob = F.log_softmax(logits_list[0], dim=0)[0]
                actor_loss = actor_loss - log_prob * advantages[i]
                num_actor_samples += 1
        
        if num_actor_samples > 0:
            actor_loss = actor_loss / num_actor_samples
        
        critic_loss = F.mse_loss(values_with_grad, targets)
        
        loss = actor_loss + self.value_coef * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 0.5)
        
        # Copy gradients to global model (GPU -> CPU)
        for local_param, global_param in zip(self.local_net.parameters(), self.global_model.parameters()):
            if local_param.grad is not None:
                # Move gradient to CPU before assigning to global model
                cpu_grad = local_param.grad.cpu()
                if global_param.grad is None:
                    global_param.grad = cpu_grad.clone()
                else:
                    global_param.grad.copy_(cpu_grad)
        
        self.optimizer.step()
        
        self.buffer = []

    def observe(self, state: Dict):
        state_vec_np = get_state_vec(state)
        state_vec = torch.from_numpy(state_vec_np).to(self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, new_hidden = self.local_net(state_vec, None, self.hidden)
        self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())

    def save(self, path: str):
        model = self.global_model if self.global_model else self.local_net
        torch.save(model.state_dict(), path)

    def load(self, path: str):
        model = self.global_model if self.global_model else self.local_net
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()