"""
Rummikub RL Agent with Actor-Critic Architecture (GPU Optimized)

OPTIMIZATIONS:
- Batched next_value computation (no per-transition loops)
- Pre-allocated tensors and reduced CPU-GPU transfers
- Non-blocking data transfers
- Optional mixed precision training (AMP)
- TF32 enabled for Ampere+ GPUs

STATE VECTOR (114 features):
- Hand tile counts: 53, Table tile counts: 53
- Opponent/pool/meld features: 8

ACTION VECTOR (57 features):
- Tiles played counts: 53
- Draw/set_type/extension/meld_value: 4
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
    """Convert game state to 114-dim feature vector."""
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
    
    # Table structure features
    num_runs_norm = num_runs / 10.0
    num_groups_norm = num_groups / 10.0
    
    # Legal plays count
    if legal_actions is not None:
        play_actions = [a for a in legal_actions if a.action_type != 'draw']
        legal_plays_count = len(play_actions) / 50.0
    else:
        legal_plays_count = 0.0
    
    # Hand potential value
    hand_value = sum(t.get_value() for t in hand)
    hand_potential = hand_value / 100.0
    
    # Concatenate all features (114 total)
    vec = np.concatenate([
        hand_counts, table_counts,
        [opp_count, pool_size, has_melded, opp_has_melded,
         num_runs_norm, num_groups_norm, legal_plays_count, hand_potential]
    ])
    
    return vec.astype(np.float32)


def get_action_vec(action: RummikubAction) -> np.ndarray:
    """Convert action to 57-dim feature vector."""
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
    
    # Set type encoding
    if action.action_type == 'draw':
        set_type_enc = 0.0
    elif not action.set_types:
        set_type_enc = 0.5
    else:
        has_run = 'run' in action.set_types
        has_group = 'group' in action.set_types
        if has_run and has_group:
            set_type_enc = 1.0
        elif has_run:
            set_type_enc = 0.33
        else:
            set_type_enc = 0.67
    
    # Is extension
    is_extension = 1.0 if action.is_extension else 0.0
    
    # Meld value normalized
    meld_value_norm = action.meld_value / 50.0
    
    # Concatenate (57 total)
    vec = np.concatenate([
        played_counts,
        [flag_draw, set_type_enc, is_extension, meld_value_norm]
    ])
    
    return vec.astype(np.float32)


class ActorCritic(nn.Module):
    """Actor-Critic network with LSTM (GPU Optimized)."""
    
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, 
                 hidden_size=512, num_layers=2, dropout=0.1, use_layer_norm=True):
        super(ActorCritic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(state_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state_vecs, action_vecs_list=None, hiddens=None):
        """Forward pass."""
        batch_size = state_vecs.size(0)
        device = state_vecs.device
        
        if hiddens is None:
            hiddens = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            )
        
        lstm_out, new_hiddens = self.lstm(state_vecs, hiddens)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.layer_norm(lstm_out)
        
        values = self.critic_head(lstm_out).squeeze(-1)
        
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
            state_expanded = lstm_out[b].unsqueeze(0).expand(num_actions, -1)
            combined = torch.cat([state_expanded, action_vecs_stacked], dim=1)
            logits = self.actor_head(combined).squeeze(-1)
            logits_list.append(logits)
        
        return values, logits_list, new_hiddens
    
    def forward_critic_batch(self, state_vecs):
        """Efficient batched critic-only forward pass."""
        batch_size = state_vecs.size(0)
        device = state_vecs.device
        
        hiddens = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )
        
        lstm_out, _ = self.lstm(state_vecs, hiddens)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.layer_norm(lstm_out)
        
        values = self.critic_head(lstm_out).squeeze(-1)
        return values


class ACAgent:
    """Actor-Critic Agent with GPU-optimized training."""
    
    def __init__(self, global_model=None, optimizer=None, is_worker=False, use_gpu=True,
                 gamma=0.99, entropy_coef=0.02, value_coef=0.5, batch_size=64, 
                 grad_clip=0.5, exploration_prob=0.05, use_amp=False):
        # Device selection with GPU optimizations
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Enable TF32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
        else:
            self.device = torch.device('cpu')
        
        self.local_net = ActorCritic().to(self.device)
        self.global_model = global_model
        self.optimizer = optimizer
        self.is_worker = is_worker
        
        # Mixed precision training
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.hidden = None
        self.reset_hidden()
        
        self.name = "ACAgent"
        self.buffer: List[Transition] = []
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.exploration_prob = exploration_prob
        
        if not is_worker and optimizer is None:
            self.optimizer = optim.Adam(self.local_net.parameters(), lr=0.0003)

    def reset_hidden(self):
        """Reset LSTM hidden state."""
        self.hidden = (
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device),
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, device=self.device)
        )
    
    def _get_temp_hidden(self):
        """Create fresh zero hidden states for LSTM (no gradients needed)."""
        # Always create fresh tensors to avoid in-place modification conflicts
        # during backpropagation. Using requires_grad=False since we don't need
        # gradients through hidden states for our actor computation.
        return (
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, 
                       device=self.device, requires_grad=False),
            torch.zeros(self.local_net.num_layers, 1, self.local_net.hidden_size, 
                       device=self.device, requires_grad=False)
        )

    def sync_local_to_global(self):
        """Copy global model weights to local model (CPU -> GPU)."""
        if self.global_model is not None:
            state_dict = self.global_model.state_dict()
            local_state_dict = {k: v.to(self.device, non_blocking=True) for k, v in state_dict.items()}
            self.local_net.load_state_dict(local_state_dict)

    def select_action(self, state: Dict, legal_actions: List[RummikubAction]) -> Tuple[RummikubAction, int, List[np.ndarray]]:
        """Select action using policy network with epsilon-greedy exploration."""
        if not legal_actions:
            return RummikubAction(action_type='draw'), -1, []
        
        state_vec_np = get_state_vec(state, legal_actions)
        all_action_vecs_np = [get_action_vec(a) for a in legal_actions]
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_prob:
            idx = np.random.randint(len(legal_actions))
            return legal_actions[idx], idx, all_action_vecs_np
        
        # Move to GPU with non-blocking transfer
        state_vec = torch.from_numpy(state_vec_np).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
        all_action_vecs = [torch.from_numpy(av).to(self.device, non_blocking=True) for av in all_action_vecs_np]
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    _, logits_list, new_hidden = self.local_net(state_vec, [all_action_vecs], self.hidden)
            else:
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
        """Store transition."""
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
        """GPU-optimized loss computation and global model update."""
        if not self.buffer or self.global_model is None:
            self.buffer = []
            return
        
        self.sync_local_to_global()
        batch_size = len(self.buffer)
        
        # ===== BATCHED DATA PREPARATION =====
        state_vecs = np.stack([t.state_vec for t in self.buffer])
        rewards_np = np.array([t.reward for t in self.buffer], dtype=np.float32)
        dones_np = np.array([t.done for t in self.buffer], dtype=np.float32)
        
        # Move to GPU with non-blocking transfers
        state_vecs_t = torch.from_numpy(state_vecs).to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards_np).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones_np).to(self.device, non_blocking=True)
        
        # ===== BATCHED NEXT STATE VALUES (KEY OPTIMIZATION) =====
        next_state_indices = []
        next_states_list = []
        for i, trans in enumerate(self.buffer):
            if trans.next_state_vec is not None and not trans.done:
                next_state_indices.append(i)
                next_states_list.append(trans.next_state_vec)
        
        next_values = torch.zeros(batch_size, device=self.device)
        
        if next_states_list:
            # Batch all next states together - single GPU forward pass
            next_states_np = np.stack(next_states_list)
            next_states_t = torch.from_numpy(next_states_np).to(self.device, non_blocking=True).unsqueeze(1)
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        next_vals = self.local_net.forward_critic_batch(next_states_t)
                else:
                    next_vals = self.local_net.forward_critic_batch(next_states_t)
            
            # Scatter results to correct positions
            for j, idx in enumerate(next_state_indices):
                next_values[idx] = next_vals[j]
        
        # ===== COMPUTE VALUES AND ADVANTAGES =====
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    values_detached = self.local_net.forward_critic_batch(state_vecs_t)
            else:
                values_detached = self.local_net.forward_critic_batch(state_vecs_t)
        
        # TD targets
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Normalized advantages
        advantages = targets - values_detached
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ===== FORWARD PASS WITH GRADIENTS =====
        if self.use_amp:
            with torch.cuda.amp.autocast():
                values_with_grad = self.local_net.forward_critic_batch(state_vecs_t)
        else:
            values_with_grad = self.local_net.forward_critic_batch(state_vecs_t)
        
        # ===== ACTOR LOSS =====
        actor_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        num_actor_samples = 0
        
        for i, trans in enumerate(self.buffer):
            if trans.action_idx is None or trans.action_idx < 0:
                continue
            if not trans.all_action_vecs or len(trans.all_action_vecs) == 0:
                continue
            
            all_action_vecs = [torch.from_numpy(av).to(self.device, non_blocking=True) 
                              for av in trans.all_action_vecs]
            
            state_t = state_vecs_t[i:i+1]  # Slice for efficiency
            
            # Create fresh hidden states for each iteration to avoid gradient conflicts
            temp_hidden = self._get_temp_hidden()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    _, logits_list, _ = self.local_net(state_t, [all_action_vecs], temp_hidden)
            else:
                _, logits_list, _ = self.local_net(state_t, [all_action_vecs], temp_hidden)
            
            if logits_list[0] is not None and logits_list[0].numel() > 0:
                logits = logits_list[0]
                log_probs = F.log_softmax(logits, dim=0)
                probs = F.softmax(logits, dim=0)
                
                log_prob = log_probs[trans.action_idx]
                actor_loss = actor_loss - log_prob * advantages[i]
                
                entropy = -(probs * log_probs).sum()
                entropy_loss = entropy_loss - entropy
                
                num_actor_samples += 1
        
        if num_actor_samples > 0:
            actor_loss = actor_loss / num_actor_samples
            entropy_loss = entropy_loss / num_actor_samples
        
        # ===== CRITIC LOSS =====
        critic_loss = F.mse_loss(values_with_grad, targets)
        
        # ===== TOTAL LOSS AND BACKWARD =====
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.grad_clip)
            
            for local_param, global_param in zip(self.local_net.parameters(), self.global_model.parameters()):
                if local_param.grad is not None:
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.cpu().clone()
                    else:
                        global_param.grad.copy_(local_param.grad.cpu())
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), self.grad_clip)
            
            for local_param, global_param in zip(self.local_net.parameters(), self.global_model.parameters()):
                if local_param.grad is not None:
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.cpu().clone()
                    else:
                        global_param.grad.copy_(local_param.grad.cpu())
            
            self.optimizer.step()
        
        self.buffer = []

    def observe(self, state: Dict, legal_actions: List[RummikubAction] = None):
        """Update LSTM hidden state by observing a state."""
        state_vec_np = get_state_vec(state, legal_actions)
        state_vec = torch.from_numpy(state_vec_np).to(self.device, non_blocking=True).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    _, new_hidden = self.local_net(state_vec, None, self.hidden)
            else:
                _, new_hidden = self.local_net(state_vec, None, self.hidden)
        self.hidden = (new_hidden[0].detach(), new_hidden[1].detach())

    def save(self, path: str):
        """Save model weights."""
        model = self.global_model if self.global_model else self.local_net
        torch.save(model.state_dict(), path)

    def load(self, path: str):
        """Load model weights."""
        model = self.global_model if self.global_model else self.local_net
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()