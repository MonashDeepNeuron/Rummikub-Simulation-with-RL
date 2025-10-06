"""
Interactive Human vs AI Blackjack Game
Standalone script that loads trained agents and provides side-by-side gameplay comparison

Configure your agent paths below in the AGENT_CONFIGS section.
Simply run: python interactive_blackjack_game.py
"""

# ================================
# AGENT CONFIGURATION SECTION
# ================================
AGENT_CONFIGS = {
    # Enhanced DQN Agent Configuration (10,000 episodes optimized)
    "dqn": {
        "enabled": True,  # Set to False to disable this agent
        "model_path": "dqn_agent.pth",  # Enhanced model with LayerNorm + Dropout
        "display_name": "DQN Agent"
    },
    
    # Actor-Critic Agent Configuration
    "actor_critic": {
        "enabled": True,  # Now enabled with correct model path
        "model_path": "actor_critic_agent.pth",
        "display_name": "Actor-Critic Agent"
    },
    
    # Temporal Difference Search Agent Configuration
    "td_search": {
        "enabled": True,  # Disabled until you have a proper TD Search model
        "model_path": "td_search_agent.pth",  # Use proper TD model path
        "display_name": "TD Search Agent"
    },
    
    # Q-Table Agent Configuration
    "qtable": {
        "enabled": True,  # Now enabled with correct model path
        "model_path": "qtable_agent.pth",
        "display_name": "Q-Table Agent"
    },

    # Basic Strategy (Optimal Policy) Agent
    "basic_strategy": {
        "enabled": True,  # Enable it
        "model_path": None,  # No model file needed
        "display_name": "Basic Strategy Agent"
    },

}

import pickle
import sys
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    print("Error: tkinter not available")
    GUI_AVAILABLE = False

# Optional matplotlib for basic GUI fallback
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import modern UI components
try:
    from ui_components import (
        ModernColors, PlayerSection, ModernButton, WinRateChart,
        format_cards_display, get_card_color
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Warning: ui_components not available, using basic styling")
    UI_COMPONENTS_AVAILABLE = False


class DQNNetwork(nn.Module):
    """Enhanced Deep Q-Network"""
    def __init__(self, state_size=3, action_size=2):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)


class BlackjackAgent_DQN:
    """DQN Agent for blackjack"""
    def __init__(self, model_path=None):
        self.state_size = 3
        self.action_size = 2
        self.q_network = DQNNetwork(self.state_size, self.action_size)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def state_to_tensor(self, state):
        """Convert state to normalized tensor"""
        player_sum = (state[0] - 12) / 9.0
        dealer_card = (state[1] - 6) / 5.0
        usable_ace = float(state[2])
        
        return torch.tensor([player_sum, dealer_card, usable_ace], dtype=torch.float32).unsqueeze(0)
    
    def get_action(self, state, training=False):
        """Get action from trained network"""
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def load_model(self, model_path):
        """Load trained model - handles both old and new checkpoint formats"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle new checkpoint format (with full training state)
            if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            # Handle old format (direct state dict)
            else:
                self.q_network.load_state_dict(checkpoint)
                
            self.q_network.eval()
            print(f"✅ DQN model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading DQN model: {e}")
    
    def save_model(self, model_path):
        """Save trained model"""
        torch.save(self.q_network.state_dict(), model_path)


class BlackjackAgent_ActorCritic:
    """Actor-Critic Agent for loading and inference"""
    def __init__(self, model_path=None):
        # Initialize actor network (same architecture as in training)
        obs_size = 3
        action_size = 2
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def get_action(self, state, training=False):
        """Get action from Actor-Critic model"""
        try:
            state_tensor = torch.tensor([state[0], state[1], int(state[2])], 
                                      dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = self.actor(state_tensor).squeeze(0)
                # Greedy action selection
                return torch.argmax(action_probs).item()
        except Exception as e:
            print(f"❌ Error in Actor-Critic inference: {e}")
            return np.random.choice(2)
    
    def load_model(self, model_path):
        """Load Actor-Critic model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle checkpoint dictionary format
            if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
            else:
                # Fallback for direct model save
                self.actor.load_state_dict(checkpoint)
            
            self.actor.eval()
            print(f"✅ Actor-Critic model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Actor-Critic model: {e}")
    
    def save_model(self, model_path):
        """Save Actor-Critic model"""
        torch.save(self.actor.state_dict(), model_path)


class BlackjackAgent_TDSearch:
    """Temporal Difference Search Agent for loading and inference"""
    def __init__(self, model_path=None):
        self.model_data = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def get_action(self, state, training=False):
        """Get action from TD Search model"""
        if self.model_data is None:
            return np.random.choice(2)  # Random if no model loaded
        
        try:
            if isinstance(self.model_data, dict):
                if state in self.model_data:
                    action_values = self.model_data[state]
                    return np.argmax(action_values)
                else:
                    return np.random.choice(2)
            elif hasattr(self.model_data, 'predict'):
                state_array = np.array([state[0], state[1], int(state[2])]).reshape(1, -1)
                action_values = self.model_data.predict(state_array)
                return np.argmax(action_values)
            else:
                state_tensor = torch.tensor([state[0], state[1], int(state[2])], 
                                          dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_values = self.model_data(state_tensor)
                    return action_values.argmax().item()
        except Exception as e:
            print(f"❌ Error in TD Search inference: {e}")
            return np.random.choice(2)
    
    def load_model(self, model_path):
        """Load TD Search model"""
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                self.model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                if hasattr(self.model_data, 'eval'):
                    self.model_data.eval()
            else:
                try:
                    with open(model_path, 'rb') as f:
                        self.model_data = pickle.load(f)
                except:
                    self.model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    if hasattr(self.model_data, 'eval'):
                        self.model_data.eval()
            
            print(f"✅ TD Search model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading TD Search model: {e}")
    
    def save_model(self, model_path):
        """Save TD Search model"""
        if self.model_data:
            if model_path.endswith('.pkl'):
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model_data, f)
            else:
                torch.save(self.model_data, model_path)

class BlackjackAgent_QTable:
    """Q-Table Agent for loading and inference"""
    def __init__(self, model_path=None):
        self.q_table = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def get_action(self, state, training=False):
        """Get action from Q-table"""
        if self.q_table is None:
            return np.random.choice(2)
        
        try:
            # Q-table uses state as key
            if state in self.q_table:
                return np.argmax(self.q_table[state])
            else:
                # If state not in table, return random action
                return np.random.choice(2)
        except Exception as e:
            print(f"❌ Error in Q-Table inference: {e}")
            return np.random.choice(2)
    
    def load_model(self, model_path):
        """Load Q-table model"""
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
            elif model_path.endswith('.pth') or model_path.endswith('.pt'):
                data = torch.load(model_path, map_location='cpu', weights_only=False)
            else:
                # Try pickle first, then torch
                try:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                except:
                    data = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle checkpoint dictionary format (contains 'Q' key)
            if isinstance(data, dict) and 'Q' in data:
                self.q_table = data['Q']
            else:
                # Direct Q-table save
                self.q_table = data
            
            print(f"✅ Q-Table model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading Q-Table model: {e}")
    
    def save_model(self, model_path):
        """Save Q-table model"""
        if self.q_table:
            if model_path.endswith('.pkl'):
                with open(model_path, 'wb') as f:
                    pickle.dump(self.q_table, f)
            else:
                torch.save(self.q_table, model_path)


class BlackjackAgent_BasicStrategy:
    """Rule-based Basic Strategy Agent (Optimal policy for single-deck S17)"""
    def __init__(self, model_path=None):
        pass  # No model needed

    def get_action(self, state, training=False):
        """
        Map environment state (player_sum, dealer_upcard, usable_ace)
        to deterministic action using your basic_strategy_action function.
        """
        from basic_strategy_bot import basic_strategy_action
        return basic_strategy_action(state)


class GUIBlackjackGame:
    """GUI version using tkinter"""
    
    def __init__(self, agents):
        self.agents = agents  # Dictionary of agents
        self.root = tk.Tk()
        self.root.title("Human vs Multi-AI Blackjack")
        self.root.geometry("1400x900")
        if UI_COMPONENTS_AVAILABLE:
            self.root.configure(bg=ModernColors.MAIN_BG)
        else:
            self.root.configure(bg="lightgray")
        
        # Game state
        self.human_env = None
        self.agent_envs = {}
        self.human_state = None
        self.agent_states = {}
        self.human_done = False
        self.agent_done = {}
        self.human_action_pending = False
        
        # Statistics
        self.human_wins = 0
        self.human_losses = 0
        self.human_draws = 0
        self.agent_stats = {}
        self.games_played = 0
        
        # Initialize agent statistics
        for agent_name in agents:
            self.agent_stats[agent_name] = {'wins': 0, 'losses': 0, 'draws': 0}
            self.agent_done[agent_name] = False
        
        self.setup_gui()
    
    def _is_natural_blackjack(self, state, is_first_hand=True):
        """Check if state is natural blackjack (21 with first 2 cards)"""
        player_sum = state[0]
        usable_ace = state[2]
        
        return is_first_hand and player_sum == 21 and usable_ace
    
    def _is_natural_blackjack_from_cards(self, cards):
        """Check if hand is natural blackjack (Ace + 10-value card)"""
        if len(cards) != 2:
            return False
        
        has_ace = 1 in cards
        has_ten = any(card == 10 for card in cards)
        
        return has_ace and has_ten
    
    def setup_gui(self):
        """Setup the GUI elements using modern components"""
        if UI_COMPONENTS_AVAILABLE:
            self._setup_modern_gui()
        else:
            self._setup_basic_gui()
    
    def _setup_modern_gui(self):
        """Setup modern GUI"""
        title_label = tk.Label(self.root, text="🎯 Human vs AI Blackjack", 
                              font=("Segoe UI", 18, "bold"),
                              fg=ModernColors.WHITE, bg=ModernColors.MAIN_BG)
        title_label.pack(pady=10)
        
        self.dealer_section = PlayerSection(self.root, "🎰 DEALER", player_type="dealer")
        self.dealer_section.pack(fill=tk.X, padx=15, pady=8)
        
        self.human_section = PlayerSection(self.root, "👤 HUMAN PLAYER", player_type="human")
        self.human_section.pack(fill=tk.X, padx=15, pady=8)
        
        agents_frame = tk.Frame(self.root, bg=ModernColors.MAIN_BG)
        agents_frame.pack(fill=tk.X, padx=15, pady=8)
        
        self.agent_sections = {}
        for agent_name in self.agents.keys():
            display_name = AGENT_CONFIGS.get(agent_name, {}).get('display_name', f"🤖 {agent_name.upper()}")
            clean_name = display_name.replace("🧠 ", "").replace("🎭 ", "").replace("🔍 ", "").replace("📊 ", "")
            
            agent_section = PlayerSection(agents_frame, f"🤖 {clean_name}", 
                                        player_type="ai", compact=True)
            agent_section.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
            self.agent_sections[agent_name] = agent_section
        
        button_frame = tk.Frame(self.root, bg=ModernColors.MAIN_BG)
        button_frame.pack(pady=15)
        
        self.hit_button = ModernButton(button_frame, "🎯 HIT", command=self.human_hit,
                                      state=tk.DISABLED, button_type="hit")
        self.hit_button.pack(side=tk.LEFT, padx=8)
        
        self.stick_button = ModernButton(button_frame, "✋ STICK", command=self.human_stick,
                                        state=tk.DISABLED, button_type="stick")
        self.stick_button.pack(side=tk.LEFT, padx=8)
        
        self.new_game_button = ModernButton(button_frame, "🔄 NEW GAME", command=self.start_new_game,
                                           button_type="new")
        self.new_game_button.pack(side=tk.LEFT, padx=8)
        
        self.chart = WinRateChart(self.root)
        self.chart.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.win_history = {"Human": [], "Human_losses": [], "Games": []}
        for agent_name in self.agents.keys():
            display_name = AGENT_CONFIGS.get(agent_name, {}).get('display_name', f"🤖 {agent_name.upper()}")
            self.win_history[display_name] = []
            self.win_history[f"{display_name}_losses"] = []
    
    def _setup_basic_gui(self):
        """Fallback basic GUI setup if ui_components not available"""
        # Title
        title_label = tk.Label(self.root, text="🎯 Human vs AI Blackjack", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        dealer_frame = tk.Frame(self.root, relief=tk.RAISED, borderwidth=3, bg="darkgreen")
        dealer_frame.pack(pady=15, padx=30, fill=tk.X)
        
        tk.Label(dealer_frame, text="🎰 DEALER", font=("Arial", 18, "bold"), 
                bg="darkgreen", fg="white").pack(pady=10)
        
        dealer_info_frame = tk.Frame(dealer_frame, bg="darkgreen")
        dealer_info_frame.pack(pady=5)
        
        self.dealer_card_label = tk.Label(dealer_info_frame, text="Showing: -", 
                                         font=("Arial", 14, "bold"), 
                                         bg="darkgreen", fg="white")
        self.dealer_card_label.pack(side=tk.LEFT, padx=20)
        
        self.dealer_final_label = tk.Label(dealer_info_frame, text="Final: -", 
                                          font=("Arial", 14, "bold"), 
                                          bg="darkgreen", fg="white")
        self.dealer_final_label.pack(side=tk.LEFT, padx=20)
        
        human_frame = tk.Frame(self.root, relief=tk.RAISED, borderwidth=4, bg="lightblue")
        human_frame.pack(pady=10, padx=40, fill=tk.X)
        
        tk.Label(human_frame, text="👤 HUMAN PLAYER", font=("Arial", 16, "bold"), 
                bg="lightblue").pack(pady=8)
        
        # Human game info
        human_info_frame = tk.Frame(human_frame, bg="lightblue")
        human_info_frame.pack(pady=5)
        
        self.human_sum_label = tk.Label(human_info_frame, text="Sum: -", 
                                       font=("Arial", 14, "bold"), bg="lightblue")
        self.human_sum_label.pack(side=tk.LEFT, padx=15)
        
        self.human_status_label = tk.Label(human_frame, text="Ready", 
                                          font=("Arial", 12, "bold"), bg="lightblue")
        self.human_status_label.pack(pady=3)
        
        self.human_result_label = tk.Label(human_frame, text="", 
                                          font=("Arial", 14, "bold"), bg="lightblue")
        self.human_result_label.pack(pady=2)
        
        # Human stats
        human_tally_frame = tk.Frame(human_frame, relief=tk.SUNKEN, borderwidth=1, bg="white") 
        human_tally_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(human_tally_frame, text="📊 Record:", font=("Arial", 11, "bold"), 
                bg="white").pack(side=tk.LEFT, padx=5)
        
        self.human_wins_label = tk.Label(human_tally_frame, text="W: 0", 
                                        font=("Arial", 11), bg="white", fg="green")
        self.human_wins_label.pack(side=tk.LEFT, padx=8)
        
        self.human_losses_label = tk.Label(human_tally_frame, text="L: 0", 
                                          font=("Arial", 11), bg="white", fg="red")
        self.human_losses_label.pack(side=tk.LEFT, padx=8)
        
        self.human_draws_label = tk.Label(human_tally_frame, text="D: 0", 
                                         font=("Arial", 11), bg="white", fg="blue")
        self.human_draws_label.pack(side=tk.LEFT, padx=8)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.hit_button = tk.Button(button_frame, text="🎯 HIT", command=self.human_hit,
                                   state=tk.DISABLED, bg="lightgreen", font=("Arial", 12))
        self.hit_button.pack(side=tk.LEFT, padx=10)
        
        self.stick_button = tk.Button(button_frame, text="🛑 STICK", command=self.human_stick,
                                     state=tk.DISABLED, bg="lightcoral", font=("Arial", 12))
        self.stick_button.pack(side=tk.LEFT, padx=10)
        
        self.new_game_button = tk.Button(button_frame, text="🎮 NEW GAME", command=self.start_new_game,
                                        bg="lightblue", font=("Arial", 12))
        self.new_game_button.pack(side=tk.LEFT, padx=10)
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        agents_container = tk.Frame(main_frame)
        agents_container.pack(pady=10, padx=20, fill=tk.X)
        
        self.agent_sum_labels = {}
        self.agent_status_labels = {}
        self.agent_result_labels = {}
        
        for agent_name in self.agents.keys():
            agent_frame = tk.Frame(agents_container, relief=tk.RAISED, borderwidth=2, bg="#34495e")
            agent_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
            
            display_name = AGENT_CONFIGS.get(agent_name, {}).get('display_name', f"🤖 {agent_name.upper()}")
            tk.Label(agent_frame, text=display_name, font=("Arial", 12, "bold"), 
                    bg="#34495e", fg="white").pack(pady=5)
            
            self.agent_sum_labels[agent_name] = tk.Label(agent_frame, text="Sum: -", 
                                                        font=("Arial", 11), bg="#34495e", fg="white")
            self.agent_sum_labels[agent_name].pack(pady=2)
            
            self.agent_status_labels[agent_name] = tk.Label(agent_frame, text="Ready", 
                                                           font=("Arial", 10), bg="#34495e", fg="white")
            self.agent_status_labels[agent_name].pack(pady=2)
            
            self.agent_result_labels[agent_name] = tk.Label(agent_frame, text="", 
                                                           font=("Arial", 11, "bold"), bg="#34495e", fg="white")
            self.agent_result_labels[agent_name].pack(pady=1)
            

        

        
        # Win Rate Graph - larger and more prominent
        graph_frame = tk.Frame(self.root)
        graph_frame.pack(pady=15, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(graph_frame, text="� Win Rate Over Time", font=("Arial", 12, "bold")).pack()
        
        # Create matplotlib figure - bigger size
        self.fig = Figure(figsize=(12, 6), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Wins Over Time")
        self.ax.set_xlabel("Game Number")
        self.ax.set_ylabel("Cumulative Wins")
        self.ax.grid(True, alpha=0.3)
        
        # Canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize win tracking
        self.win_history = {"Human": [], "Human_losses": [], "Games": []}
        for agent_name in self.agents.keys():
            display_name = AGENT_CONFIGS.get(agent_name, {}).get('display_name', f"🤖 {agent_name.upper()}")
            self.win_history[display_name] = []
            self.win_history[f"{display_name}_losses"] = []
    
    def update_graph(self):
        """Update the win rate graph"""
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'chart'):
            self.chart.update_chart(self.win_history)
        elif hasattr(self, 'ax'):
            self.ax.clear()
            self.ax.set_title("Wins Over Time")
            self.ax.set_xlabel("Game Number")
            self.ax.set_ylabel("Cumulative Wins")
            self.ax.grid(True, alpha=0.3)
            
            # Add theoretical win probability reference line (42% of total games played)
            if len(self.win_history.get("Games", [])) > 0:
                max_games = max(self.win_history["Games"])
                theoretical_wins = [g * 0.42 for g in self.win_history["Games"]]
                self.ax.plot(self.win_history["Games"], theoretical_wins, 
                           color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                           label='Theoretical (42%)')
            
            if len(self.win_history["Games"]) > 0:
                # Define colors for up to 7 entities (Human, Dealer, 5 agents)
                colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                
                # Plot human wins
                self.ax.plot(self.win_history["Games"], self.win_history["Human"], 
                            label="Human", marker='o', linewidth=2, color=colors[0])
                
                # Plot dealer wins
                self.ax.plot(self.win_history["Games"], self.win_history["Dealer"], 
                            label="Dealer", marker='^', linewidth=2, color=colors[1])
                
                # Plot agent wins using display names
                color_index = 2  # Start from third color
                for player_name, wins in self.win_history.items():
                    if player_name not in ["Games", "Human", "Dealer"]:
                        self.ax.plot(self.win_history["Games"], wins, 
                                    label=player_name, marker='s', linewidth=2, 
                                    color=colors[color_index % len(colors)])
                        color_index += 1
                
                self.ax.legend()
                self.ax.set_xlim(0, max(self.win_history["Games"]) + 1)
            
            if hasattr(self, 'canvas'):
                self.canvas.draw()
    
    def start_new_game(self):
        """Start a new game"""
        # Generate random seed for identical hands
        seed = np.random.randint(0, 100000)
        
        # Close previous environments
        if self.human_env:
            self.human_env.close()
        for env in self.agent_envs.values():
            env.close()
        
        # Create new environments
        self.human_env = gym.make("Blackjack-v1", natural=True, sab=False)
        self.agent_envs = {}
        self.agent_states = {}
        
        for agent_name in self.agents:
            self.agent_envs[agent_name] = gym.make("Blackjack-v1", natural=True, sab=False)
            self.agent_states[agent_name], _ = self.agent_envs[agent_name].reset(seed=seed)
            self.agent_done[agent_name] = False
        
        # Reset with same seed
        self.human_state, _ = self.human_env.reset(seed=seed)
        
        # Get initial cards from environment
        self.human_cards = list(self.human_env.unwrapped.player)
        
        # Reset game state
        self.human_done = False
        self.human_action_pending = True
        self.games_played += 1
        
        # Check for natural blackjack
        has_blackjack = self._is_natural_blackjack(self.human_state, is_first_hand=True)
        
        if has_blackjack:
            # Auto-stick on blackjack
            self.human_state, human_reward, terminated, truncated, _ = self.human_env.step(0)
            self.human_done = True
            self.human_action_pending = False
            
            # Accept the result from the environment - if it's a draw, dealer also had 21
            if human_reward == 0:
                print("🤝 Human: Draw - Dealer also has 21")
            
            # Update UI for blackjack
            self.hit_button.config(state=tk.DISABLED)
            self.stick_button.config(state=tk.DISABLED)
            self._update_human_display(status="🎉 NATURAL BLACKJACK! (Auto-stick)", cards=self.human_cards)
            
            # Update dealer display with final cards for blackjack resolution
            try:
                dealer_cards = list(self.human_env.unwrapped.dealer)
                if len(dealer_cards) > 1:
                    self._update_dealer_display(dealer_cards=dealer_cards, show_hidden=False)
            except:
                pass
            
            # Complete the game
            self.root.after(1000, lambda: self._complete_agent_turn(human_reward))
        else:
            # Update UI for normal play
            self.hit_button.config(state=tk.NORMAL)
            self.stick_button.config(state=tk.NORMAL)
            self._update_human_display(status="Your Turn!")
        
        self.new_game_button.config(state=tk.DISABLED)
        
        # Clear dealer display first
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'dealer_section'):
            self.dealer_section.clear_cards()
        
        # Update dealer display with initial cards
        try:
            # Get dealer's initial cards (showing card + hidden card)
            dealer_cards = list(self.human_env.unwrapped.dealer)
            self._update_dealer_display(showing=self.human_state[1], dealer_cards=dealer_cards, show_hidden=True)
        except:
            # Fallback if we can't access dealer cards
            self._update_dealer_display(showing=self.human_state[1])
        
        # Update human display with cards
        self._update_human_display(
            sum_val=self.human_state[0],
            cards=self.human_cards
        )
        
        # Update agent displays
        for agent_name in self.agents:
            state = self.agent_states[agent_name]
            self._update_agent_display(
                agent_name,
                sum_val=state[0],
                status="Ready"
            )

    
    def human_hit(self):
        """Handle human hit action"""
        if not self.human_action_pending or self.human_done:
            return
        
        self.human_state, human_reward, terminated, truncated, _ = self.human_env.step(1)
        self.human_done = terminated or truncated
        
        # Update human cards after hit
        self.human_cards = list(self.human_env.unwrapped.player)
        
        # Update display after hit
        self._update_human_display(
            sum_val=self.human_state[0],
            cards=self.human_cards
        )
        
        if self.human_done:
            self.human_action_pending = False
            self.hit_button.config(state=tk.DISABLED)
            self.stick_button.config(state=tk.DISABLED)
            self._update_human_display(status="Finished")
            self.root.after(1000, lambda: self._complete_agent_turn(human_reward))
    
    def human_stick(self):
        """Handle human stick action"""
        if not self.human_action_pending or self.human_done:
            return
        
        self.human_state, human_reward, terminated, truncated, _ = self.human_env.step(0)
        self.human_done = terminated or truncated
        self.human_action_pending = False
        
        self.hit_button.config(state=tk.DISABLED)
        self.stick_button.config(state=tk.DISABLED)
        self._update_human_display(
            sum_val=self.human_state[0],
            cards=self.human_cards,
            status="Finished"
        )
        
        self.root.after(1000, lambda: self._complete_agent_turn(human_reward))
    
    def _update_dealer_display(self, showing=None, final_sum=None, cards_display=None, dealer_cards=None, show_hidden=False):
        """Update dealer display - works with both modern and basic GUI"""
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'dealer_section'):
            if dealer_cards is not None and len(dealer_cards) >= 2:
                self.dealer_section.clear_cards()
                # Show first card (dealer's showing card)
                first_card = dealer_cards[0]
                rank = self._format_card_rank(first_card)
                suit = self._get_card_suit(0)
                color = get_card_color(0)
                self.dealer_section.add_card(rank, suit, color)
                
                if show_hidden:
                    # Add hidden card and show only visible card sum
                    self.dealer_section.add_card_back()
                    self.dealer_section.update_sum(first_card)  # Only show visible card value
                else:
                    # Show all remaining cards (final state)
                    for i, card in enumerate(dealer_cards[1:], 1):
                        rank = self._format_card_rank(card)
                        suit = self._get_card_suit(i)
                        color = get_card_color(i)
                        self.dealer_section.add_card(rank, suit, color)
                    # Show full hand sum
                    dealer_sum = self._calculate_hand_value(dealer_cards)
                    self.dealer_section.update_sum(dealer_sum)
            elif showing is not None:
                self.dealer_section.clear_cards()
                # Add showing card and hidden card
                rank = self._format_card_rank(showing)
                suit = self._get_card_suit(0)
                color = get_card_color(0)
                self.dealer_section.add_card(rank, suit, color)
                self.dealer_section.add_card_back()
                # Show only the visible card value
                self.dealer_section.update_sum(showing)
            
            if final_sum is not None:
                self.dealer_section.update_sum(final_sum)
        else:
            # Basic GUI fallback
            if hasattr(self, 'dealer_card_label') and showing is not None:
                self.dealer_card_label.config(text=f"Showing: {showing}")
            if hasattr(self, 'dealer_final_label') and final_sum is not None:
                self.dealer_final_label.config(text=f"Final: {final_sum}")
    
    def _update_human_display(self, sum_val=None, status=None, result=None, cards=None):
        """Update human display - works with both modern and basic GUI"""
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'human_section'):
            if cards is not None:
                self.human_section.clear_cards()
                for i, card in enumerate(cards):
                    rank = self._format_card_rank(card)
                    suit = self._get_card_suit(i)
                    color = get_card_color(i)
                    self.human_section.add_card(rank, suit, color)
            if sum_val is not None:
                self.human_section.update_sum(sum_val)
            if status is not None:
                self.human_section.update_status(status)
            if result is not None:
                self.human_section.update_status("", result)
        else:
            if hasattr(self, 'human_sum_label') and sum_val is not None:
                self.human_sum_label.config(text=f"Sum: {sum_val}")
            if hasattr(self, 'human_status_label') and status is not None:
                self.human_status_label.config(text=status)
            if hasattr(self, 'human_result_label') and result is not None:
                result_text = "🎉 WIN" if result == 1 else "😔 LOSE" if result == -1 else "🤝 DRAW"
                result_color = "green" if result == 1 else "red" if result == -1 else "blue"
                self.human_result_label.config(text=result_text, fg=result_color)
    
    def _update_human_record(self):
        """Update human win/loss record"""
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'human_section'):
            self.human_section.update_record(self.human_wins, self.human_losses, self.human_draws)
        else:
            # Basic GUI fallback
            if hasattr(self, 'human_wins_label'):
                self.human_wins_label.config(text=f"W: {self.human_wins}")
            if hasattr(self, 'human_losses_label'):
                self.human_losses_label.config(text=f"L: {self.human_losses}")
            if hasattr(self, 'human_draws_label'):
                self.human_draws_label.config(text=f"D: {self.human_draws}")
    
    def _update_agent_display(self, agent_name, sum_val=None, status=None, result=None):
        """Update agent display - works with both modern and basic GUI"""
        if UI_COMPONENTS_AVAILABLE and hasattr(self, 'agent_sections') and agent_name in self.agent_sections:
            section = self.agent_sections[agent_name]
            if sum_val is not None:
                section.update_sum(sum_val)
            if status is not None and result is None:
                section.update_status(status)
            if result is not None:
                # Ensure result is properly converted to int for comparison
                result = int(result)
                # Use update_status with reward parameter
                section.update_status("", reward=result)
        else:
            if agent_name in self.agent_sum_labels and sum_val is not None:
                self.agent_sum_labels[agent_name].config(text=f"Sum: {sum_val}")
            if agent_name in self.agent_status_labels and status is not None:
                self.agent_status_labels[agent_name].config(text=status)
            if agent_name in self.agent_result_labels and result is not None:
                # Ensure proper int comparison
                result = int(result)
                result_text = "WIN" if result == 1 else "LOSE" if result == -1 else "DRAW"
                self.agent_result_labels[agent_name].config(text=result_text)

    def _complete_agent_turn(self, human_reward):
        """Let all agents complete their turns"""
        # Update all agent status
        for agent_name in self.agents:
            self._update_agent_display(agent_name, status="Playing...")
        
        self.agent_rewards = {}
        self.current_agent_index = 0
        self.agent_names_list = list(self.agents.keys())
        
        def next_agent_step():
            if self.current_agent_index < len(self.agent_names_list):
                agent_name = self.agent_names_list[self.current_agent_index]
                
                if not self.agent_done[agent_name]:
                    # Check for natural blackjack first
                    current_state = self.agent_states[agent_name]
                    if self._is_natural_blackjack(current_state, is_first_hand=True):
                        # Auto-stick on blackjack
                        state, reward, terminated, truncated, _ = self.agent_envs[agent_name].step(0)  # Stick
                        self._update_agent_display(agent_name, status="🎉 BLACKJACK!")
                    else:
                        # Normal agent play
                        action = self.agents[agent_name].get_action(current_state, training=False)
                        action_name = "HIT" if action == 1 else "STICK"
                        self._update_agent_display(agent_name, status=f"{action_name}")
                        state, reward, terminated, truncated, _ = self.agent_envs[agent_name].step(action)
                    
                    self.agent_states[agent_name] = state
                    self.agent_done[agent_name] = terminated or truncated
                    
                    # Update display
                    self._update_agent_display(agent_name, 
                                             sum_val=state[0])
                    
                    if self.agent_done[agent_name]:
                        # Store reward (will be recalculated in _finish_game for consistency)
                        self.agent_rewards[agent_name] = int(reward)
                        self._update_agent_display(agent_name, status="Finished")
                        self.current_agent_index += 1
                    
                    self.root.after(400, next_agent_step)  # Reduced from 800ms to 400ms
                else:
                    self.current_agent_index += 1
                    self.root.after(50, next_agent_step)  # Reduced from 100ms to 50ms
            else:
                # All agents finished
                self._finish_game(human_reward, self.agent_rewards)
        
        next_agent_step()
    
    def _finish_game(self, human_reward, agent_rewards):
        """Finish the game and update statistics"""
        # Get and display dealer's final sum
        try:
            # Find an environment where a player didn't bust to get the complete dealer hand
            dealer_env = self.human_env  # Default to human env
            
            # If human busted, check if any agent didn't bust and use their environment
            if self.human_state[0] > 21:  # Human busted
                for agent_name in agent_rewards.keys():
                    if self.agent_states[agent_name][0] <= 21:  # This agent didn't bust
                        dealer_env = self.agent_envs[agent_name]
                        break
            
            dealer_cards = dealer_env.unwrapped.dealer
            dealer_sum = self._calculate_hand_value(dealer_cards)
            
            # Check if any player (human or agents) didn't bust (sum <= 21)
            any_player_still_in = (self.human_state[0] <= 21)  # Human didn't bust
            for agent_name in agent_rewards.keys():
                if self.agent_states[agent_name][0] <= 21:  # Agent didn't bust
                    any_player_still_in = True
                    break
            
            # Only show dealer's final sum if they completed their hand
            if any_player_still_in or len(dealer_cards) > 2:
                self._update_dealer_display(final_sum=dealer_sum, dealer_cards=dealer_cards, show_hidden=False)
            else:
                self._update_dealer_display(final_sum="(didn't complete)")
        except Exception as e:
            self._update_dealer_display(final_sum="Unknown")
            dealer_sum = 0
            dealer_cards = []
        
        dealer_has_natural = self._is_natural_blackjack_from_cards(dealer_cards) if dealer_cards else False
        human_cards = list(self.human_env.unwrapped.player)
        human_has_natural = self._is_natural_blackjack_from_cards(human_cards)
        
        human_sum = self.human_state[0]
        if human_sum > 21:
            human_reward = -1
        elif dealer_sum > 21:
            human_reward = 1
        elif human_sum == 21 and dealer_sum == 21:
            if human_has_natural and not dealer_has_natural:
                human_reward = 1
            elif dealer_has_natural and not human_has_natural:
                human_reward = -1
            else:
                human_reward = 0
        elif human_sum > dealer_sum:
            human_reward = 1
        elif human_sum < dealer_sum:
            human_reward = -1
        else:
            human_reward = 0
        corrected_agent_rewards = {}
        for agent_name in agent_rewards.keys():
            agent_sum = self.agent_states[agent_name][0]
            agent_cards = list(self.agent_envs[agent_name].unwrapped.player)
            agent_has_natural = self._is_natural_blackjack_from_cards(agent_cards)
            
            if agent_sum > 21:
                corrected_agent_rewards[agent_name] = -1
            elif dealer_sum > 21:
                corrected_agent_rewards[agent_name] = 1
            elif agent_sum == 21 and dealer_sum == 21:
                if agent_has_natural and not dealer_has_natural:
                    corrected_agent_rewards[agent_name] = 1
                elif dealer_has_natural and not agent_has_natural:
                    corrected_agent_rewards[agent_name] = -1
                else:
                    corrected_agent_rewards[agent_name] = 0
            elif agent_sum > dealer_sum:
                corrected_agent_rewards[agent_name] = 1
            elif agent_sum < dealer_sum:
                corrected_agent_rewards[agent_name] = -1
            else:
                corrected_agent_rewards[agent_name] = 0
        
        if human_reward == 1:
            self.human_wins += 1
        elif human_reward == -1:
            self.human_losses += 1
        else:
            self.human_draws += 1
        
        self._update_human_record()
        self._update_human_display(result=human_reward)
        
        for agent_name, reward in corrected_agent_rewards.items():
            if reward == 1:
                self.agent_stats[agent_name]['wins'] += 1
            elif reward == -1:
                self.agent_stats[agent_name]['losses'] += 1
            else:
                self.agent_stats[agent_name]['draws'] += 1
            
            if UI_COMPONENTS_AVAILABLE and hasattr(self, 'agent_sections') and agent_name in self.agent_sections:
                stats = self.agent_stats[agent_name]
                self.agent_sections[agent_name].update_record(stats['wins'], stats['losses'], stats['draws'])
            
            self._update_agent_display(agent_name, result=reward)
        
        # Update win history for graph
        self.win_history["Games"].append(self.games_played)
        self.win_history["Human"].append(self.human_wins)
        self.win_history["Human_losses"].append(self.human_losses)
        
        for agent_name in self.agents.keys():
            display_name = AGENT_CONFIGS.get(agent_name, {}).get('display_name', f"🤖 {agent_name.upper()}")
            if display_name not in self.win_history:
                self.win_history[display_name] = []
            if f"{display_name}_losses" not in self.win_history:
                self.win_history[f"{display_name}_losses"] = []
            self.win_history[display_name].append(self.agent_stats[agent_name]['wins'])
            self.win_history[f"{display_name}_losses"].append(self.agent_stats[agent_name]['losses'])
        
        # Update graph
        self.update_graph()
        
        # Enable new game button
        self.new_game_button.config(state=tk.NORMAL)
    
    def _calculate_hand_value(self, cards):
        """Calculate the value of a hand of cards"""
        if not cards:
            return 0
            
        total = sum(cards)
        aces = cards.count(1)  # Aces are represented as 1
        
        # Convert aces from 1 to 11 if it doesn't bust
        while total <= 11 and aces > 0:
            total += 10  # Convert ace from 1 to 11
            aces -= 1
            
        return total
    
    def _format_card_rank(self, card_value):
        """Format card value to display rank"""
        card_names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        return card_names.get(card_value, str(card_value))
    
    def _get_card_suit(self, card_index):
        """Get suit symbol for card based on its position"""
        suits = ['♠', '♥', '♦', '♣']
        return suits[card_index % len(suits)]
    
    def _format_cards(self, cards):
        """Format cards for display with suits"""
        if not cards:
            return "-"
        
        suits = ['♠', '♥', '♦', '♣']
        card_names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        
        formatted_cards = []
        for i, card in enumerate(cards):
            suit = suits[i % len(suits)]
            name = card_names.get(card, str(card))
            formatted_cards.append(f"{name}{suit}")
        
        return " ".join(formatted_cards)
    

    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted")
        finally:
            if self.human_env:
                self.human_env.close()
            for env in self.agent_envs.values():
                env.close()


def load_agent(agent_type, model_path):
    """Load agent based on type and model path"""
    agent_map = {
        'dqn': BlackjackAgent_DQN,
        'deep': BlackjackAgent_DQN,
        'actor_critic': BlackjackAgent_ActorCritic,
        'ac': BlackjackAgent_ActorCritic,
        'td_search': BlackjackAgent_TDSearch,
        'tds': BlackjackAgent_TDSearch,
        'qtable': BlackjackAgent_QTable,
        'q_table': BlackjackAgent_QTable,
        'basic_strategy': BlackjackAgent_BasicStrategy,
        'bs': BlackjackAgent_BasicStrategy
    }
    
    agent_class = agent_map.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_class(model_path)

def load_agents_from_config():
    """Load agents based on the configuration"""
    agents = {}
    
    for agent_type, config in AGENT_CONFIGS.items():
        if not config['enabled']:
            continue

        model_path = config.get('model_path')

        # Allow agents that don't need a file (like basic_strategy)
        if model_path is None:
            agent = load_agent(agent_type, model_path)
            agents[agent_type] = agent
            print(f"✅ {config['display_name']} loaded (no model file required)")
            continue

        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found for {agent_type}: {model_path}")
            continue
        
        try:
            agent = load_agent(agent_type, model_path)
            agents[agent_type] = agent
        except Exception as e:
            print(f"Error loading {agent_type}: {e}")
    
    return agents


def main():
    """Main function"""
    if not GUI_AVAILABLE:
        print("GUI not available! Please install tkinter.")
        sys.exit(1)
    
    agents = load_agents_from_config()
    
    if not agents:
        print("No agents loaded! Check AGENT_CONFIGS.")
        sys.exit(1)
    
    game = GUIBlackjackGame(agents)
    game.run()


if __name__ == "__main__":
    main()
