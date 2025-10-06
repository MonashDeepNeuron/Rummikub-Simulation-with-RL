"""
UI Components for Blackjack Game
Modern, reusable tkinter components for the interactive blackjack game
Optimized for multiple agents (4-5) displayed side by side on one screen
"""

import tkinter as tk
from tkinter import ttk
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ModernColors:
    """Color palette for the modern UI"""
    # Backgrounds
    MAIN_BG = "#1e3c72"
    DEALER_BG = "#0f7a3f"
    DEALER_BG_LIGHT = "#128a4a"
    HUMAN_BG = "#f8f9fa"
    HUMAN_BG_LIGHT = "#ffffff"
    AI_BG = "#2c3e50"
    AI_BG_LIGHT = "#34495e"
    
    # Text colors
    WHITE = "#ffffff"
    DARK = "#2c3e50"
    GRAY = "#6b7280"
    LIGHT_GRAY = "#9ca3af"
    
    # Status colors
    WIN_GREEN = "#4ade80"
    WIN_GREEN_DARK = "#065f46"
    LOSS_RED = "#f87171"
    LOSS_RED_DARK = "#7f1d1d"
    DRAW_YELLOW = "#fbbf24"
    DRAW_YELLOW_DARK = "#78350f"
    PLAY_BLUE = "#60a5fa"
    PLAY_BLUE_DARK = "#1e3a8a"
    
    # Button colors
    BTN_HIT = "#4ade80"
    BTN_HIT_TEXT = "#065f46"
    BTN_STICK = "#f87171"
    BTN_STICK_TEXT = "#7f1d1d"
    BTN_NEW = "#60a5fa"
    BTN_NEW_TEXT = "#1e3a8a"
    
    # Accents
    CARD_BG = "#ffffff"
    CARD_RED = "#dc2626"
    CARD_BLACK = "#1f2937"
    INFO_BOX_BG = "#e5e7eb"


class ModernCard(tk.Frame):
    """A modern playing card component"""
    
    def __init__(self, parent, rank="?", suit="", color="black", **kwargs):
        super().__init__(parent, bg=ModernColors.CARD_BG, relief=tk.RAISED, 
                        borderwidth=1, width=45, height=65, **kwargs)
        self.pack_propagate(False)
        
        # Card color
        card_color = ModernColors.CARD_RED if color == "red" else ModernColors.CARD_BLACK
        
        # Rank label
        rank_label = tk.Label(self, text=rank, font=("Arial", 14, "bold"),
                             fg=card_color, bg=ModernColors.CARD_BG)
        rank_label.pack(expand=True, pady=(3, 0))
        
        # Suit label
        if suit:
            suit_label = tk.Label(self, text=suit, font=("Arial", 11),
                                 fg=card_color, bg=ModernColors.CARD_BG)
            suit_label.pack(pady=(0, 3))


class CardBack(tk.Frame):
    """A card back (hidden card) component"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernColors.PLAY_BLUE, relief=tk.RAISED,
                        borderwidth=1, width=45, height=65, **kwargs)
        self.pack_propagate(False)
        
        label = tk.Label(self, text="?", font=("Arial", 24, "bold"),
                        fg=ModernColors.WHITE, bg=ModernColors.PLAY_BLUE)
        label.pack(expand=True)


class InfoBox(tk.Frame):
    """An information display box - compact version"""
    
    def __init__(self, parent, label_text, value_text="-", bg=None, fg=None, compact=False, **kwargs):
        # Determine background color
        if bg is None:
            bg = ModernColors.INFO_BOX_BG
        
        super().__init__(parent, bg=bg, relief=tk.FLAT, borderwidth=0, **kwargs)
        
        # Label color
        label_fg = fg if fg else ModernColors.GRAY
        value_fg = fg if fg else ModernColors.DARK
        
        # Adjust sizes based on compact mode
        label_size = 7 if compact else 9
        value_size = 11 if compact else 16
        pady_top = 4 if compact else 8
        pady_bottom = 3 if compact else 8
        
        # Label
        self.label = tk.Label(self, text=label_text, font=("Segoe UI", label_size),
                             fg=label_fg, bg=bg)
        self.label.pack(pady=(pady_top, 0))
        
        # Value
        self.value = tk.Label(self, text=value_text, font=("Segoe UI", value_size, "bold"),
                             fg=value_fg, bg=bg)
        self.value.pack(pady=(2, pady_bottom))
    
    def update_value(self, new_value):
        """Update the displayed value"""
        self.value.config(text=str(new_value))


class StatusBadge(tk.Label):
    """A status badge (WIN/LOSE/DRAW) - compact version"""
    
    def __init__(self, parent, status="", compact=False, **kwargs):
        font_size = 9 if compact else 12
        super().__init__(parent, text="", font=("Segoe UI", font_size, "bold"),
                        relief=tk.FLAT, **kwargs)
        if status:
            self.set_status(status)
    
    def set_status(self, status, reward=None):
        """Update status badge"""
        if reward is not None:
            if reward == 1:
                status = "WIN"
            elif reward == -1:
                status = "LOSE"
            else:
                status = "DRAW"
        
        status = status.upper()
        
        if status == "WIN":
            self.config(text=f" {status} ", bg=ModernColors.WIN_GREEN,
                       fg=ModernColors.WIN_GREEN_DARK)
        elif status == "LOSE":
            self.config(text=f" {status} ", bg=ModernColors.LOSS_RED,
                       fg=ModernColors.LOSS_RED_DARK)
        elif status == "DRAW":
            self.config(text=f" {status} ", bg=ModernColors.DRAW_YELLOW,
                       fg=ModernColors.DRAW_YELLOW_DARK)
        elif status == "PLAYING":
            self.config(text=f" {status} ", bg=ModernColors.PLAY_BLUE,
                       fg=ModernColors.PLAY_BLUE_DARK)
        else:
            self.config(text=f" {status} ", bg=ModernColors.INFO_BOX_BG,
                       fg=ModernColors.DARK)


class RecordDisplay(tk.Frame):
    """Win/Loss/Draw record display - compact version"""
    
    def __init__(self, parent, wins=0, losses=0, draws=0, bg=None, compact=False, **kwargs):
        super().__init__(parent, bg=bg if bg else ModernColors.HUMAN_BG, **kwargs)
        
        font_size = 8 if compact else 10
        padding = 3 if compact else 5
        inner_padding = 6 if compact else 10
        
        # Wins
        self.wins_frame = tk.Frame(self, bg=ModernColors.INFO_BOX_BG, relief=tk.FLAT)
        self.wins_frame.pack(side=tk.LEFT, padx=padding)
        self.wins_label = tk.Label(self.wins_frame, text=f"W:{wins}",
                                   font=("Segoe UI", font_size, "bold"),
                                   fg=ModernColors.WIN_GREEN,
                                   bg=ModernColors.INFO_BOX_BG, padx=inner_padding, pady=3)
        self.wins_label.pack()
        
        # Losses
        self.losses_frame = tk.Frame(self, bg=ModernColors.INFO_BOX_BG, relief=tk.FLAT)
        self.losses_frame.pack(side=tk.LEFT, padx=padding)
        self.losses_label = tk.Label(self.losses_frame, text=f"L:{losses}",
                                     font=("Segoe UI", font_size, "bold"),
                                     fg=ModernColors.LOSS_RED,
                                     bg=ModernColors.INFO_BOX_BG, padx=inner_padding, pady=3)
        self.losses_label.pack()
        
        # Draws
        self.draws_frame = tk.Frame(self, bg=ModernColors.INFO_BOX_BG, relief=tk.FLAT)
        self.draws_frame.pack(side=tk.LEFT, padx=padding)
        self.draws_label = tk.Label(self.draws_frame, text=f"D:{draws}",
                                    font=("Segoe UI", font_size, "bold"),
                                    fg=ModernColors.DRAW_YELLOW,
                                    bg=ModernColors.INFO_BOX_BG, padx=inner_padding, pady=3)
        self.draws_label.pack()
    
    def update(self, wins, losses, draws):
        """Update the record display"""
        self.wins_label.config(text=f"W:{wins}")
        self.losses_label.config(text=f"L:{losses}")
        self.draws_label.config(text=f"D:{draws}")


class PlayerSection(tk.Frame):
    """A complete player section with all info - optimized for multiple agents"""
    
    def __init__(self, parent, title, player_type="human", compact=False, **kwargs):
        # Determine colors based on player type
        if player_type == "dealer":
            bg = ModernColors.DEALER_BG
            fg = ModernColors.WHITE
            info_bg = ModernColors.DEALER_BG_LIGHT
        elif player_type == "human":
            bg = ModernColors.HUMAN_BG
            fg = ModernColors.DARK
            info_bg = ModernColors.INFO_BOX_BG
        else:  # AI
            bg = ModernColors.AI_BG
            fg = ModernColors.WHITE
            info_bg = ModernColors.AI_BG_LIGHT
        
        border_width = 2 if compact else 3
        super().__init__(parent, bg=bg, relief=tk.RAISED, borderwidth=border_width, **kwargs)
        
        self.player_type = player_type
        self.bg = bg
        self.fg = fg
        self.info_bg = info_bg
        self.compact = compact
        
        # Adjust sizing based on compact mode - reduced heights
        title_size = 11 if compact else 14
        padding_x = 8 if compact else 15
        padding_y_top = 3 if compact else 6
        padding_y_bottom = 2 if compact else 3
        
        # Header
        header_frame = tk.Frame(self, bg=bg)
        header_frame.pack(fill=tk.X, padx=padding_x, pady=(padding_y_top, padding_y_bottom))
        
        self.title_label = tk.Label(header_frame, text=title,
                                    font=("Segoe UI", title_size, "bold"),
                                    fg=fg, bg=bg)
        self.title_label.pack(side=tk.LEFT)
        
        # Record display (only for human/AI)
        if player_type != "dealer":
            self.record = RecordDisplay(header_frame, bg=bg, compact=compact)
            self.record.pack(side=tk.RIGHT)
        
        # Separator
        sep_height = 1 if compact else 2
        sep = tk.Frame(self, height=sep_height, bg=ModernColors.WHITE if player_type != "human" else ModernColors.GRAY)
        sep.pack(fill=tk.X, padx=padding_x, pady=padding_y_bottom)
        
        # Cards container
        self.cards_frame = tk.Frame(self, bg=bg)
        card_padding = 3 if compact else 6
        self.cards_frame.pack(fill=tk.X, padx=padding_x, pady=card_padding)
        
        # Info grid
        self.info_frame = tk.Frame(self, bg=bg)
        info_padding = 3 if compact else 4
        self.info_frame.pack(fill=tk.X, padx=padding_x, pady=(0, padding_y_top))
        
        # Create info boxes
        self.sum_box = InfoBox(self.info_frame, "Sum", "-", bg=info_bg, fg=fg, compact=compact)
        self.sum_box.pack(side=tk.LEFT, padx=info_padding, fill=tk.BOTH, expand=True)
        
        # Status
        self.status_frame = tk.Frame(self.info_frame, bg=info_bg, relief=tk.FLAT)
        self.status_frame.pack(side=tk.LEFT, padx=info_padding, fill=tk.BOTH, expand=True)
        
        status_label_size = 7 if compact else 9
        tk.Label(self.status_frame, text="Status", font=("Segoe UI", status_label_size),
                fg=ModernColors.GRAY if player_type == "human" else ModernColors.LIGHT_GRAY,
                bg=info_bg).pack(pady=(3 if compact else 5, 0))
        
        self.status_badge = StatusBadge(self.status_frame, "Ready", compact=compact)
        self.status_badge.config(bg=info_bg)
        self.status_badge.pack(pady=(1 if compact else 3, 3 if compact else 5))
    
    def clear_cards(self):
        """Clear all cards from display"""
        for widget in self.cards_frame.winfo_children():
            widget.destroy()
    
    def add_card(self, rank, suit="", color="black"):
        """Add a card to the display"""
        card = ModernCard(self.cards_frame, rank, suit, color)
        card.pack(side=tk.LEFT, padx=2)
    
    def add_card_back(self):
        """Add a hidden card"""
        card = CardBack(self.cards_frame)
        card.pack(side=tk.LEFT, padx=2)
    
    def update_sum(self, value):
        """Update the sum display"""
        self.sum_box.update_value(value)
    

    
    def update_status(self, status, reward=None):
        """Update the status badge"""
        self.status_badge.set_status(status, reward)
    
    def update_record(self, wins, losses, draws):
        """Update win/loss/draw record"""
        if hasattr(self, 'record'):
            self.record.update(wins, losses, draws)


class ModernButton(tk.Button):
    """A styled modern button - compact version"""
    
    def __init__(self, parent, text, command=None, button_type="primary", **kwargs):
        # Determine colors based on button type
        if button_type == "hit":
            bg = ModernColors.BTN_HIT
            fg = ModernColors.BTN_HIT_TEXT
        elif button_type == "stick":
            bg = ModernColors.BTN_STICK
            fg = ModernColors.BTN_STICK_TEXT
        elif button_type == "new":
            bg = ModernColors.BTN_NEW
            fg = ModernColors.BTN_NEW_TEXT
        else:
            bg = ModernColors.PLAY_BLUE
            fg = ModernColors.PLAY_BLUE_DARK
        
        super().__init__(parent, text=text, command=command,
                        font=("Segoe UI", 11, "bold"),
                        bg=bg, fg=fg, padx=20, pady=8,
                        relief=tk.RAISED, bd=2, cursor="hand2",
                        activebackground=bg, activeforeground=fg,
                        **kwargs)


class WinRateChart(tk.Frame):
    """Dual chart component showing cumulative wins and win percentages"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=ModernColors.WHITE, relief=tk.RAISED, borderwidth=2, **kwargs)
        
        if not MATPLOTLIB_AVAILABLE:
            tk.Label(self, text="Matplotlib not available",
                    font=("Segoe UI", 10), bg=ModernColors.WHITE).pack(pady=10)
            return
        
        # Create matplotlib figure with two subplots side by side (no title, more vertical space)
        self.fig = Figure(figsize=(12, 4.0), dpi=100)
        
        # Left subplot - Cumulative Wins
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_xlabel('Games', fontsize=9, fontweight='bold')
        self.ax1.set_ylabel('Cumulative Wins', fontsize=9, fontweight='bold')
        self.ax1.set_title('Cumulative Wins', fontsize=10, fontweight='bold')
        self.ax1.tick_params(labelsize=8)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f9fafb')
        
        # Right subplot - Win Percentage
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel('Games', fontsize=9, fontweight='bold')
        self.ax2.set_ylabel('Win Percentage (%)', fontsize=9, fontweight='bold')
        self.ax2.set_title('Win Percentage Over Time', fontsize=10, fontweight='bold')
        self.ax2.tick_params(labelsize=8)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f9fafb')
        self.ax2.set_ylim(20, 70)  # Focus on 20%-70% range for better visibility
        
        self.fig.tight_layout(pad=1.5, rect=[0, 0.02, 1, 1])  # Add bottom margin to prevent x-label cutoff
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
    
    def update_chart(self, win_history):
        """Update both charts with win history data and win percentages"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Clear both axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Set up axes
        self.ax1.set_xlabel('Games', fontsize=9, fontweight='bold')
        self.ax1.set_ylabel('Cumulative Wins', fontsize=9, fontweight='bold')
        self.ax1.set_title('Cumulative Wins', fontsize=10, fontweight='bold')
        self.ax1.tick_params(labelsize=8)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f9fafb')
        
        self.ax2.set_xlabel('Games', fontsize=9, fontweight='bold')
        self.ax2.set_ylabel('Win Percentage (%)', fontsize=9, fontweight='bold')
        self.ax2.set_title('Win Percentage Over Time', fontsize=10, fontweight='bold')
        self.ax2.tick_params(labelsize=8)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f9fafb')
        self.ax2.set_ylim(20, 70)  # Focus on 20%-70% range for better visibility
        
        # Add theoretical win probability line at 42%
        self.ax2.axhline(y=42, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Theoretical (42%)')
        
        if "Games" not in win_history or len(win_history["Games"]) == 0:
            self.fig.tight_layout(pad=1.0)
            self.canvas.draw()
            return
        
        games = win_history["Games"]
        # Extended color palette for up to 7 entities (Human + 5 agents + Dealer)
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        plot_index = 0
        
        # Plot Human first with slight offset
        if "Human" in win_history:
            human_wins = win_history["Human"]
            human_percentages = [(w / g * 100) if g > 0 else 0 for w, g in zip(human_wins, games)]
            
            color = colors[0]
            offset_games_1 = [g + (plot_index * 0.02) for g in games]
            self.ax1.plot(offset_games_1, human_wins, marker='o', linewidth=1.8, 
                         markersize=5, label="Human", color=color, alpha=0.9)
            self.ax2.plot(offset_games_1, human_percentages, marker='o', linewidth=1.8, 
                         markersize=5, label="Human", color=color, alpha=0.9)
            plot_index += 1
        
        # Plot Agents with offsets (skip _losses entries)
        for player, wins in win_history.items():
            if player in ["Games", "Human"] or "_losses" in player:
                continue
            
            percentages = [(w / g * 100) if g > 0 else 0 for w, g in zip(wins, games)]
            
            color = colors[plot_index % len(colors)]
            offset_games_1 = [g + (plot_index * 0.02) for g in games]
            offset_games_2 = [g + (plot_index * 0.02) for g in games]
            self.ax1.plot(offset_games_1, wins, marker='s', linewidth=1.8, 
                         markersize=5, label=player, color=color, alpha=0.9)
            self.ax2.plot(offset_games_2, percentages, marker='s', linewidth=1.8, 
                         markersize=5, label=player, color=color, alpha=0.9)
            plot_index += 1
        
        # Calculate and plot dealer win percentage (only from actual losses, excluding draws)
        if len(games) > 0:
            dealer_percentages = []
            for i, game_num in enumerate(games):
                total_dealer_wins = 0  # This is the sum of all player losses
                total_players = 0
                for player, wins in win_history.items():
                    if player == "Games" or "_losses" in player:
                        continue
                    # Get the losses for this player (dealer wins against this player)
                    losses_key = f"{player}_losses"
                    if losses_key in win_history and i < len(win_history[losses_key]):
                        total_dealer_wins += win_history[losses_key][i]
                        total_players += 1
                
                if total_players > 0 and game_num > 0:
                    # Dealer win percentage = total dealer wins / (total players * games played)
                    # This now correctly excludes draws
                    dealer_win_percentage = (total_dealer_wins / (total_players * game_num)) * 100
                else:
                    dealer_win_percentage = 0
                    
                dealer_percentages.append(dealer_win_percentage)
            
            # Plot dealer win percentage with offset
            if dealer_percentages:
                color = colors[plot_index % len(colors)]
                offset_games_2 = [g + (plot_index * 0.02) for g in games]
                self.ax2.plot(offset_games_2, dealer_percentages, marker='^', linewidth=1.8, 
                             markersize=5, label="Dealer", color=color, alpha=0.9)
        
        # Add theoretical dealer win probability line at 49%
        self.ax2.axhline(y=49, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, label='Dealer Theoretical (49%)')
        
        # Add legends
        self.ax1.legend(loc='upper left', fontsize=7)
        self.ax2.legend(loc='upper left', fontsize=7)
        
        # Set x-axis limits
        if games:
            self.ax1.set_xlim(0, max(games) + 1)
            self.ax2.set_xlim(0, max(games) + 1)
        
        self.fig.tight_layout(pad=0.5, rect=[0, 0.02, 1, 1])  # Add bottom margin to prevent x-label cutoff
        self.canvas.draw()


def format_cards_display(cards):
    """Format a list of card values into a display string with suits"""
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


def get_card_color(card_index):
    """Get the color for a card based on its suit index"""
    suits_colors = ['black', 'red', 'red', 'black']
    return suits_colors[card_index % len(suits_colors)]
