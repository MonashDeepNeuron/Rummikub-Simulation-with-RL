"""
Rummikub RL Agent with Actor-Critic Architecture

ENHANCED STATE VECTOR (114 features):
- Hand tile counts: 53 (52 regular + 1 joker)
- Table tile counts: 53
- Opponent tile count: 1 (normalized /30)
- Pool size: 1 (normalized /80)
- Has melded (me): 1
- Has melded (opp): 1
- NEW: Num table runs: 1 (normalized /10)
- NEW: Num table groups: 1 (normalized /10)
- NEW: Legal plays count: 1 (normalized /50)
- NEW: Hand potential value: 1 (normalized /100)

ENHANCED ACTION VECTOR (57 features):
- Tiles played counts: 53
- Is draw: 1
- NEW: Set type encoding: 1 (0=draw, 0.33=run, 0.67=group, 1=mixed)
- NEW: Is extension: 1
- NEW: Meld value: 1 (normalized /50)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import namedtuple
from Rummikub_env import RummikubEnv, RummikubAction, TileType, Color

# State: 114 features, Action: 57 features
STATE_DIM = 114
ACTION_DIM = 57

# Store numpy arrays to avoid gradient issues
Transition = namedtuple('Transition', (
    'state_vec',      # numpy array (114,)
    'action_idx',     # int or None
    'action_vec',     # numpy array (57,) or None  
    'reward',         # float
    'next_state_vec', # numpy array (114,) or None
    'done',           # bool
    'info',           # dict
    'num_actions',    # int
    'all_action_vecs' # List of numpy arrays - ALL legal actions for policy gradient
))


def get_state_vec(state: Dict, legal_actions: List[RummikubAction] = None) -> np.ndarray:
    """
    Convert game state to 114-dim feature vector.
    
    Features:
    - 0-52: Hand tile counts (53)
    - 53-105: Table tile counts (53)
    - 106: Opponent tile count (normalized)
    - 107: Pool size (normalized)
    - 108: Has melded (me)
    - 109: Has melded (opp)
    - 110: Num table runs (normalized)
    - 111: Num table groups (normalized)
    - 112: Legal plays count (normalized)
    - 113: Hand potential value (normalized)
    """
    hand = state['my_hand']
    table = state['table']
    
    # Hand tile counts (53 features)
    hand_counts = np.zeros(53, dtype=np.float32)
    for t in hand:
        if t.tile_type == TileType.JOKER:
            hand_counts[52] += 1
        else:
            idx = t.color.value * 13 + (t.number - 1)
            hand_counts[idx] += 1
    
    # Table tile counts (53 features)
    table_counts = np.zeros(53, dtype=np.float32)
    num_runs = 0
    num_groups = 0
    for tile_set in table:
        if tile_set.set_type == 'run':
            num_runs += 1
        else:
            num_groups += 1
        for t in tile_set.tiles:
            if t.tile_type == TileType.JOKER:
                table_counts[52] += 1
            else:
                idx = t.color.value * 13 + (t.number - 1)
                table_counts[idx] += 1
    
    # Basic features
    opp_count = state['opponent_tile_count'] / 30.0
    pool_size = state['pool_size'] / 80.0
    has_melded = 1.0 if state['has_melded'][state['current_player']] else 0.0
    opp_has_melded = 1.0 if state['has_melded'][1 - state['current_player']] else 0.0
    
    # NEW: Table structure features
    num_runs_norm = num_runs / 10.0
    num_groups_norm = num_groups / 10.0
    
    # NEW: Legal plays count (requires legal_actions)
    if legal_actions is not None:
        play_actions = [a for a in legal_actions if a.action_type != 'draw']
        legal_plays_count = len(play_actions) / 50.0
    else:
        legal_plays_count = 0.0
    
    # NEW: Hand potential value (sum of hand tile values)
    hand_value = sum(t.get_value() for t in hand)
    hand_potential = hand_value / 100.0
    
    # Concatenate all features (114 total)
    vec = np.concatenate([
        hand_counts,        # 53
        table_counts,       # 53
        [opp_count],        # 1
        [pool_size],        # 1
        [has_melded],       # 1
        [opp_has_melded],   # 1
        [num_runs_norm],    # 1 NEW
        [num_groups_norm],  # 1 NEW
        [legal_plays_count],# 1 NEW
        [hand_potential]    # 1 NEW
    ])
    
    return vec.astype(np.float32)


def get_action_vec(action: RummikubAction) -> np.ndarray:
    """
    Convert action to 57-dim feature vector.
    
    Features:
    - 0-52: Tiles played counts (53)
    - 53: Is draw (1)
    - 54: Set type encoding (1) - 0=draw, 0.33=run, 0.67=group, 1=mixed
    - 55: Is extension (1)
    - 56: Meld value normalized (1)
    """
    tiles = action.tiles if action.tiles else []
    
    # Tiles played counts (53 features)
    played_counts = np.zeros(53, dtype=np.float32)
    for t in tiles:
        if t.tile_type == TileType.JOKER:
            played_counts[52] += 1
        else:
            idx = t.color.value * 13 + (t.number - 1)
            played_counts[idx] += 1
    
    # Is draw
    flag_draw = 1.0 if action.action_type == 'draw' else 0.0
    
    # NEW: Set type encoding
    if action.action_type == 'draw':
        set_type_enc = 0.0
    elif not action.set_types:
        set_type_enc = 0.5  # Unknown
    else:
        has_run = 'run' in action.set_types
        has_group = 'group' in action.set_types
        if has_run and has_group:
            set_type_enc = 1.0  # Mixed
        elif has_run:
            set_type_enc = 0.33  # Run only
        else:
            set_type_enc = 0.67  # Group only
    
    # NEW: Is extension
    is_extension = 1.0 if action.is_extension else 0.0
    
    # NEW: Meld value normalized
    meld_value_norm = action.meld_value / 50.0
    
    # Concatenate (57 total)
    vec = np.concatenate([
        played_counts,      # 53
        [flag_draw],        # 1
        [set_type_enc],     # 1 NEW
        [is_extension],     # 1 NEW
        [meld_value_norm]   # 1 NEW
    ])
    
    return vec.astype(np.float32)


class ActorCritic(nn.Module):
    """Actor-Critic network with LSTM for Rummikub."""
    
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, 
                 hidden_size=512, num_layers=2, dropout=0.1, use_layer_norm=True):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(state_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Identity()
        
        # Actor head: scores each action
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Critic head: estimates state value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state_vecs, action_vecs_list=None, hiddens=None):
        """
        Forward pass.
        
        Args:
            state_vecs: (batch, seq_len, state_dim) or (batch, state_dim)
            action_vecs_list: List of action vectors per batch item, or None
            hiddens: LSTM hidden state tuple
            
        Returns:
            values: State values (batch,)
            logits_list: List of action logits per batch item (if action_vecs_list provided)
            new_hiddens: Updated LSTM hidden state
        """
        batch_size = state_vecs.size(0)
        
        # Ensure 3D input for LSTM
        if state_vecs.dim() == 2:
            state_vecs = state_vecs.unsqueeze(1)
        
        if hiddens is None:
            hiddens = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=state_vecs.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=state_vecs.device)
            )
        
        out, new_hiddens = self.lstm(state_vecs, hiddens)
        out = out[:, -1, :]  # Take last timestep
        out = self.layer_norm(out)
        
        # Critic: state value
        values = self.critic_head(out).squeeze(-1)
        
        if action_vecs_list is None:
            return values, new_hiddens
        
        # Actor: score each action
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
    """Actor-Critic Agent for Rummikub with A3C support."""
    
    def __init__(self, global_model=None, optimizer=None, is_worker=False, use_gpu=True,
                 gamma=0.99, entropy_coef=0.02, value_coef=0.5, batch_size=64, 
                 grad_clip=0.5, exploration_prob=0.05):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.local_net = ActorCritic().to(self.device)
        self.global_model = global_model
        self.optimizer = optimizer
        self.is_worker = is_worker
        
        self.hidden = None
        self.reset_hidden()
        
        self.name = "ACAgent"
        self.buffer: List[Transition] = []
        
        # Hyperparameters (from Config or defaults)
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.exploration_prob = exploration_prob
        
        if not is_worker and optimizer is None:
            self.optimizer = optim.Adam(self.local_net.parameters(), lr=0.0003)

    def reset_hidden(self):
        self.hidden = (
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device),
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device)
        )

    def sync_local_to_global(self):
        """Copy global model weights to local model."""
        if self.global_model is not None:
            state_dict = self.global_model.state_dict()
            local_state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            self.local_net.load_state_dict(local_state_dict)

    def select_action(self, state: Dict, legal_actions: List[RummikubAction]) -> Tuple[RummikubAction, int, List[np.ndarray]]:
        """
        Select action using policy network with epsilon-greedy exploration.
        
        Returns:
            action: Selected RummikubAction
            action_idx: Index of selected action
            all_action_vecs: ALL action vectors (for proper policy gradient)
        """
        if not legal_actions:
            return RummikubAction(action_type='draw'), -1, []
        
        # Get state vector with legal actions info
        state_vec_np = get_state_vec(state, legal_actions)
        state_vec = torch.from_numpy(state_vec_np).to(self.device).unsqueeze(0).unsqueeze(0)
        
        # Get ALL action vectors (critical for proper policy gradient!)
        all_action_vecs_np = [get_action_vec(a) for a in legal_actions]
        all_action_vecs = [torch.from_numpy(av).to(self.device) for av in all_action_vecs_np]
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_prob:
            idx = np.random.randint(len(legal_actions))
            return legal_actions[idx], idx, all_action_vecs_np
        
        with torch.no_grad():
            _, logits_list, new_hidden = self.local_net(state_vec, [all_action_vecs], self.hidden)
            self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        
        logits = logits_list[0]
        
        if logits is None or logits.numel() == 0:
            return RummikubAction(action_type='draw'), -1, []
        
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample().item()
        
        return legal_actions[idx], idx, all_action_vecs_np

    def store_transition(self, state_vec, action_idx, action_vec, reward, 
                        next_state_vec, done, info, num_actions, all_action_vecs=None):
        """Store transition with ALL action vectors for proper gradient computation."""
        trans = Transition(
            state_vec, action_idx, action_vec, reward, 
            next_state_vec, done, info, num_actions,
            all_action_vecs if all_action_vecs is not None else []
        )
        self.buffer.append(trans)

    def learn(self, state_vec, action_idx, action_vec, reward, next_state_vec, 
              done, info, num_actions, all_action_vecs=None):
        """Store transition and update if buffer is full."""
        self.store_transition(state_vec, action_idx, action_vec, reward, 
                             next_state_vec, done, info, num_actions, all_action_vecs)
        
        if len(self.buffer) >= self.batch_size or done:
            self._update_global()

    def _update_global(self):
        """Compute loss and update global model."""
        if not self.buffer or self.global_model is None:
            self.buffer = []
            return
        
        self.sync_local_to_global()
        
        batch_size = len(self.buffer)
        
        # Stack states
        state_vecs = np.stack([t.state_vec for t in self.buffer])
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float32, device=self.device)
        
        state_vecs_t = torch.from_numpy(state_vecs).to(self.device).unsqueeze(1)
        
        # Compute values (detached for advantage)
        with torch.no_grad():
            values_detached, _ = self.local_net(state_vecs_t, None)
        
        # Compute next state values for TD target
        next_values = torch.zeros(batch_size, device=self.device)
        for i, trans in enumerate(self.buffer):
            if trans.next_state_vec is not None and not trans.done:
                next_state_t = torch.from_numpy(trans.next_state_vec).to(self.device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    nv, _ = self.local_net(next_state_t, None)
                    next_values[i] = nv.squeeze()
        
        # TD targets
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Compute advantages (normalized)
        advantages = targets - values_detached
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute values with gradient for critic loss
        values_with_grad, _ = self.local_net(state_vecs_t, None)
        
        # Actor loss with proper policy gradient over ALL actions
        actor_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        num_actor_samples = 0
        
        for i, trans in enumerate(self.buffer):
            if trans.action_idx is None or trans.action_idx < 0:
                continue
            if not trans.all_action_vecs or len(trans.all_action_vecs) == 0:
                continue
            
            # Get all action vectors for this transition
            all_action_vecs = [torch.from_numpy(av).to(self.device) for av in trans.all_action_vecs]
            
            state_t = torch.from_numpy(trans.state_vec).to(self.device).unsqueeze(0).unsqueeze(0)
            
            temp_hidden = (
                torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device),
                torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device)
            )
            
            _, logits_list, _ = self.local_net(state_t, [all_action_vecs], temp_hidden)
            
            if logits_list[0] is not None and logits_list[0].numel() > 0:
                logits = logits_list[0]
                log_probs = F.log_softmax(logits, dim=0)
                probs = F.softmax(logits, dim=0)
                
                # Policy gradient: -log_prob * advantage
                log_prob = log_probs[trans.action_idx]
                actor_loss = actor_loss - log_prob * advantages[i]
                
                # Entropy bonus for exploration
                entropy = -(probs * log_probs).sum()
                entropy_loss = entropy_loss - entropy
                
                num_actor_samples += 1
        
        if num_actor_samples > 0:
            actor_loss = actor_loss / num_actor_samples
            entropy_loss = entropy_loss / num_actor_samples
        
        # Critic loss
        critic_loss = F.mse_loss(values_with_grad, targets)
        
        # Total loss
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.grad_clip)
        
        # Copy gradients to global model
        for local_param, global_param in zip(self.local_net.parameters(), self.global_model.parameters()):
            if local_param.grad is not None:
                cpu_grad = local_param.grad.cpu()
                if global_param.grad is None:
                    global_param.grad = cpu_grad.clone()
                else:
                    global_param.grad.copy_(cpu_grad)
        
        self.optimizer.step()
        
        self.buffer = []

    def observe(self, state: Dict, legal_actions: List[RummikubAction] = None):
        """Update LSTM hidden state by observing a state."""
        state_vec_np = get_state_vec(state, legal_actions)
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