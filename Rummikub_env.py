"""
Rummikub Environment for Reinforcement Learning

REWARD SYSTEM (v2 - No large constants):
=========================================
Intermediate Rewards (per turn):
  - Base play: 2.0 × (hand_before - hand_after)
  - Draw: 2.0 × (hand_before - hand_after) - 5.0
  - Ice break bonus: +30.0 (one-time)
  - Manipulation bonus: +10.0
  - NEW: 4+ tiles bonus: +5.0
  - NEW: Extension bonus: +3.0
  - NEW: Large hand penalty: -2.0 (if 20+ tiles)

Terminal Rewards (game end - NO LARGE CONSTANTS):
  - Win by empty hand: +opponent_hand_value
  - Lose by opponent empty: -agent_hand_value
  - Pool empty (lower hand wins): winner gets +(loser_hand - winner_hand)
  - Pool empty (higher hand loses): loser gets -(loser_hand - winner_hand)
  - Equal hands: 0
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import copy

class Color(Enum):
    RED = 0
    BLUE = 1
    BLACK = 2
    ORANGE = 3
    
class TileType(Enum):
    NORMAL = 0
    JOKER = 1

@dataclass
class Tile:
    """Represents a single Rummikub tile"""
    color: Optional[Color]
    number: Optional[int]
    tile_type: TileType
    tile_id: int
    
    def __hash__(self):
        return hash(self.tile_id)
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.tile_id == other.tile_id
    
    def __repr__(self):
        if self.tile_type == TileType.JOKER:
            return "JOKER"
        color_map = {Color.RED: 'R', Color.BLUE: 'b', Color.BLACK: 'B', Color.ORANGE: 'O'}
        return f"{color_map[self.color]}{self.number}"
    
    def get_value(self) -> int:
        if self.tile_type == TileType.JOKER:
            return 30
        return self.number


@dataclass
class TileSet:
    """Represents a set of tiles on the table"""
    tiles: List[Tile]
    set_type: str  # "group" or "run"
    
    def is_valid(self) -> bool:
        tile_ids = [t.tile_id for t in self.tiles]
        if len(tile_ids) != len(set(tile_ids)):
            return False
        if self.set_type == "group":
            return self._is_valid_group()
        elif self.set_type == "run":
            return self._is_valid_run()
        return False
    
    def _is_valid_group(self) -> bool:
        if len(self.tiles) < 3 or len(self.tiles) > 4:
            return False
        numbers = []
        colors = []
        for tile in self.tiles:
            if tile.tile_type != TileType.JOKER:
                numbers.append(tile.number)
                colors.append(tile.color)
        if len(numbers) > 0 and len(set(numbers)) > 1:
            return False
        if len(colors) != len(set(colors)):
            return False
        return True
    
    def _is_valid_run(self) -> bool:
        if len(self.tiles) < 3:
            return False
        colors = []
        numbers = []
        joker_count = 0
        for tile in self.tiles:
            if tile.tile_type == TileType.JOKER:
                joker_count += 1
            else:
                colors.append(tile.color)
                numbers.append(tile.number)
        if len(colors) > 0 and len(set(colors)) > 1:
            return False
        if len(numbers) != len(set(numbers)):
            return False
        if len(numbers) > 0:
            numbers.sort()
            min_num = numbers[0]
            max_num = numbers[-1]
            expected_length = max_num - min_num + 1
            internal_missing = expected_length - len(numbers)
            if internal_missing > joker_count:
                return False
        return True
    
    def get_value(self) -> int:
        """Returns the total value of tiles (jokers count as 0)"""
        return sum(t.number for t in self.tiles if t.tile_type != TileType.JOKER)
    
    def get_meld_value(self) -> int:
        """Returns value for initial meld (jokers take represented value)"""
        if self.set_type == "group":
            non_joker = [t for t in self.tiles if t.tile_type != TileType.JOKER]
            if non_joker:
                return non_joker[0].number * len(self.tiles)
            return 0
        elif self.set_type == "run":
            non_joker = sorted([t for t in self.tiles if t.tile_type != TileType.JOKER], 
                              key=lambda t: t.number)
            if non_joker:
                min_num = non_joker[0].number
                return sum(range(min_num, min_num + len(self.tiles)))
            return 0
        return 0


@dataclass 
class RummikubAction:
    """
    Represents an action in Rummikub.
    
    Enhanced with metadata for RL features:
    - set_types: List of set types being formed ('run', 'group')
    - is_extension: Whether this extends existing table sets
    - meld_value: Total value of the meld being created
    """
    action_type: str  # 'draw', 'initial_meld', 'play'
    tiles: List[Tile] = field(default_factory=list)
    sets: List[TileSet] = field(default_factory=list)
    table_config: Optional[List[TileSet]] = None
    # Enhanced metadata for RL
    set_types: List[str] = field(default_factory=list)  # ['run', 'group', ...]
    is_extension: bool = False  # True if extends existing sets
    meld_value: int = 0  # Total value of meld


class RummikubEnv:
    """
    Rummikub Environment for Reinforcement Learning
    
    REWARD PARAMETERS (v2 - balanced scale, no large constants):
    """
    # Intermediate reward parameters
    REWARD_BASE_MULTIPLIER = 2.0
    REWARD_ICE_BREAK = 30.0
    REWARD_MANIPULATION = 10.0
    REWARD_DRAW_PENALTY = -5.0
    REWARD_TILES_4_PLUS = 5.0         # NEW: Bonus for playing 4+ tiles
    REWARD_EXTENSION = 3.0            # NEW: Bonus for extending existing set
    REWARD_LARGE_HAND_PENALTY = -2.0  # NEW: Penalty per turn if 20+ tiles
    LARGE_HAND_THRESHOLD = 20         # NEW: Threshold for large hand penalty
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.tiles_deck: List[Tile] = []
        self.player_hands: List[List[Tile]] = [[], []]
        self.table: List[TileSet] = []
        self.current_player: int = 0
        self.has_melded: List[bool] = [False, False]
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_count: int = 0
        self.previous_hand_values: List[int] = [0, 0]
        self.action_generator = None
        self._initialize_deck()
    
    def _initialize_deck(self):
        """Create the full deck of 106 tiles"""
        self.tiles_deck = []
        tile_id = 0
        for copy in range(2):
            for color in Color:
                for number in range(1, 14):
                    tile = Tile(color=color, number=number, 
                                tile_type=TileType.NORMAL, tile_id=tile_id)
                    self.tiles_deck.append(tile)
                    tile_id += 1
        for _ in range(2):
            tile = Tile(color=None, number=None, 
                        tile_type=TileType.JOKER, tile_id=tile_id)
            self.tiles_deck.append(tile)
            tile_id += 1
    
    def reset(self) -> Dict:
        self.tiles_deck = []
        self._initialize_deck()
        self.rng.shuffle(self.tiles_deck)
        self.player_hands = [[], []]
        for player in range(2):
            for _ in range(14):
                tile = self.tiles_deck.pop()
                self.player_hands[player].append(tile)
        self.table = []
        self.current_player = self.rng.choice([0, 1])
        self.has_melded = [False, False]
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        self.previous_hand_values = [self._calculate_hand_value(i) for i in range(2)]
        return self._get_state()
    
    def _calculate_hand_value(self, player_id: int) -> int:
        return sum(t.get_value() for t in self.player_hands[player_id])
    
    def _count_jokers_in_hand(self, player: int) -> int:
        return sum(1 for tile in self.player_hands[player] if tile.tile_type == TileType.JOKER)
    
    def _get_state(self) -> Dict:
        """Return the current game state"""
        return {
            'my_hand': copy.deepcopy(self.player_hands[self.current_player]),
            'table': copy.deepcopy(self.table),
            'opponent_tile_count': len(self.player_hands[1 - self.current_player]),
            'pool_size': len(self.tiles_deck),
            'current_player': self.current_player,
            'has_melded': self.has_melded.copy(),
            'game_over': self.game_over,
            'winner': self.winner,
            'turn_count': self.turn_count
        }
    
    def get_legal_actions(self, player: int) -> List[RummikubAction]:
        """Get all legal actions for the specified player."""
        if self.action_generator is None:
            raise ValueError("Action generator not set")
        
        all_table_tile_ids = [t.tile_id for ts in self.table for t in ts.tiles]
        if len(all_table_tile_ids) != len(set(all_table_tile_ids)):
            if len(self.tiles_deck) > 0:
                return [RummikubAction(action_type='draw')]
            return []
        
        table_copy = copy.deepcopy(self.table)
        actions = self.action_generator.generate_all_legal_actions(
            hand_tiles=self.player_hands[player],
            table_sets=table_copy,
            has_melded=self.has_melded[player],
            pool_size=len(self.tiles_deck)
        )
        
        if len(self.tiles_deck) > 0:
            actions.append(RummikubAction(action_type='draw'))
        
        return actions
    
    def step(self, action: RummikubAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute an action and return (state, reward, done, info)
        
        Rewards are computed for BOTH players and returned in info dict.
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        acting_player = self.current_player
        opponent = 1 - acting_player
        
        hand_value_before = self._calculate_hand_value(acting_player)
        hand_size_before = len(self.player_hands[acting_player])
        
        info = {
            'action_type': action.action_type,
            'tiles_played': 0,
            'drew_tile': False,
            'ice_broken': False,
            'manipulation_occurred': False,
            'is_extension': action.is_extension,
            'invalid_action': False,
            'hand_size_before': hand_size_before,
            'hand_value_before': hand_value_before,
        }
        
        # Execute action
        if action.action_type == 'draw':
            if len(self.tiles_deck) > 0:
                drawn_tile = self.tiles_deck.pop(0)
                self.player_hands[acting_player].append(drawn_tile)
                info['drew_tile'] = True
            else:
                info['invalid_action'] = True
                
        elif action.action_type == 'initial_meld':
            if action.sets:
                all_set_tile_ids = [t.tile_id for s in action.sets for t in s.tiles]
                if len(all_set_tile_ids) != len(set(all_set_tile_ids)):
                    info['invalid_action'] = True
            
            if not info.get('invalid_action') and self._validate_initial_meld(action):
                self._apply_meld(action)
                self.has_melded[acting_player] = True
                info['ice_broken'] = True
                info['tiles_played'] = len(action.tiles)
            else:
                info['invalid_action'] = True
                
        elif action.action_type == 'play':
            if action.table_config:
                all_config_tile_ids = [t.tile_id for s in action.table_config for t in s.tiles]
                if len(all_config_tile_ids) != len(set(all_config_tile_ids)):
                    info['invalid_action'] = True
            
            if not info.get('invalid_action') and self._validate_play(action):
                info['tiles_played'] = len(action.tiles)
                # Check if manipulation occurred
                if action.table_config and len(action.table_config) != len(self.table) + len(action.sets or []):
                    info['manipulation_occurred'] = True
                self._apply_play(action)
            else:
                info['invalid_action'] = True
        
        # Calculate values after action
        hand_value_after = self._calculate_hand_value(acting_player)
        hand_size_after = len(self.player_hands[acting_player])
        info['hand_value_after'] = hand_value_after
        info['hand_size_after'] = hand_size_after
        
        # =====================================================================
        # COMPUTE REWARDS FOR BOTH PLAYERS
        # =====================================================================
        reward_acting = 0.0
        reward_opponent = 0.0
        
        if not info['invalid_action']:
            # Intermediate reward for acting player
            hand_change = hand_value_before - hand_value_after
            intermediate = self.REWARD_BASE_MULTIPLIER * hand_change
            
            if action.action_type == 'draw':
                intermediate += self.REWARD_DRAW_PENALTY
            else:
                if info.get('ice_broken'):
                    intermediate += self.REWARD_ICE_BREAK
                if info.get('manipulation_occurred'):
                    intermediate += self.REWARD_MANIPULATION
                
                # NEW: Bonus for playing 4+ tiles
                if info['tiles_played'] >= 4:
                    intermediate += self.REWARD_TILES_4_PLUS
                
                # NEW: Bonus for extension
                if action.is_extension:
                    intermediate += self.REWARD_EXTENSION
            
            # NEW: Penalty for large hand (20+ tiles)
            if hand_size_after >= self.LARGE_HAND_THRESHOLD:
                intermediate += self.REWARD_LARGE_HAND_PENALTY
            
            reward_acting = intermediate
            reward_opponent = 0.0  # Opponent gets no intermediate reward
        
        # Check termination
        done = False
        
        # Win by empty hand
        if len(self.player_hands[acting_player]) == 0:
            self.game_over = True
            self.winner = acting_player
            done = True
            
            opponent_hand_value = self._calculate_hand_value(opponent)
            
            # Terminal rewards (NO LARGE CONSTANTS!)
            # Winner gets opponent's hand value
            # Loser gets negative of opponent's hand value
            terminal_acting = opponent_hand_value
            terminal_opponent = -opponent_hand_value
            
            reward_acting += terminal_acting
            reward_opponent = terminal_opponent
            
            info['final_my_hand_value'] = 0
            info['final_opponent_hand_value'] = opponent_hand_value
            info['win_type'] = 'emptied_hand'
            info['winner'] = acting_player
            info['terminal_reward'] = terminal_acting
        
        # Pool empty - check if game should end
        elif len(self.tiles_deck) == 0:
            # Check if anyone can play
            current_can_play = any(a.action_type != 'draw' for a in self.get_legal_actions(acting_player))
            
            temp_current = self.current_player
            self.current_player = opponent
            next_can_play = any(a.action_type != 'draw' for a in self.get_legal_actions(opponent))
            self.current_player = temp_current
            
            if not current_can_play and not next_can_play:
                self.game_over = True
                done = True
                
                acting_value = self._calculate_hand_value(acting_player)
                opponent_value = self._calculate_hand_value(opponent)
                
                info['final_my_hand_value'] = acting_value
                info['final_opponent_hand_value'] = opponent_value
                
                # Determine winner by lowest hand value
                if acting_value < opponent_value:
                    # Acting player wins
                    self.winner = acting_player
                    difference = opponent_value - acting_value
                    reward_acting += difference
                    reward_opponent = -difference
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = acting_player
                    info['terminal_reward'] = difference
                elif opponent_value < acting_value:
                    # Opponent wins
                    self.winner = opponent
                    difference = acting_value - opponent_value
                    reward_acting += -difference
                    reward_opponent = difference
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = opponent
                    info['terminal_reward'] = -difference
                else:
                    # Tie - equal hands, no winner
                    self.winner = None
                    reward_acting += 0
                    reward_opponent = 0
                    info['win_type'] = 'tie'
                    info['winner'] = None
                    info['terminal_reward'] = 0
        
        # Store rewards in info for both players
        info[f'reward_for_player_{acting_player}'] = reward_acting
        info[f'reward_for_player_{opponent}'] = reward_opponent
        
        # Update state
        self.previous_hand_values[acting_player] = hand_value_after
        
        if not done:
            self.current_player = opponent
            self.turn_count += 1
        
        return self._get_state(), reward_acting, done, info
    
    def _validate_initial_meld(self, action: RummikubAction) -> bool:
        """Validate that initial meld is legal (30+ points)"""
        if not action.sets:
            return False
        total_value = sum(s.get_meld_value() for s in action.sets)
        all_valid = all(s.is_valid() for s in action.sets)
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] for t in action.tiles)
        return total_value >= 30 and all_valid and all_tiles_in_hand
    
    def _validate_play(self, action: RummikubAction) -> bool:
        """Validate that a play is legal"""
        if action.table_config is None:
            return False
        
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] for t in action.tiles)
        all_sets_valid = all(s.is_valid() for s in action.table_config)
        
        table_tiles = [t for ts in self.table for t in ts.tiles]
        new_table_tiles = [t for ts in action.table_config for t in ts.tiles]
        
        new_table_tile_ids = [t.tile_id for t in new_table_tiles]
        if len(new_table_tile_ids) != len(set(new_table_tile_ids)):
            return False
        
        expected_tile_ids = set(t.tile_id for t in table_tiles) | set(t.tile_id for t in action.tiles)
        actual_tile_ids = set(new_table_tile_ids)
        
        return all_tiles_in_hand and all_sets_valid and expected_tile_ids == actual_tile_ids
    
    def _apply_meld(self, action: RummikubAction):
        """Apply initial meld to game state"""
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        self.table.extend(copy.deepcopy(action.sets))
        self._validate_table_integrity()
    
    def _apply_play(self, action: RummikubAction):
        """Apply a play to game state"""
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        self.table = copy.deepcopy(action.table_config)
        self._validate_table_integrity()
    
    def _validate_table_integrity(self):
        """Ensure no tile appears multiple times on the table"""
        all_tile_ids = [t.tile_id for ts in self.table for t in ts.tiles]
        if len(all_tile_ids) != len(set(all_tile_ids)):
            from collections import Counter
            counts = Counter(all_tile_ids)
            duplicates = {tid: cnt for tid, cnt in counts.items() if cnt > 1}
            raise ValueError(f"Table corruption: duplicate tile_ids {duplicates}")
    
    def render(self):
        """Print the current game state"""
        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count} - Player {self.current_player}'s turn")
        print(f"{'='*60}")
        
        for i, hand in enumerate(self.player_hands):
            value = self._calculate_hand_value(i)
            print(f"\nPlayer {i} hand ({len(hand)} tiles, value={value}): ", end="")
            if i == self.current_player:
                print([str(t) for t in hand])
            else:
                print(f"[{len(hand)} hidden tiles]")
        
        print(f"\nTable ({len(self.table)} sets):")
        for i, tile_set in enumerate(self.table):
            print(f"  Set {i+1} ({tile_set.set_type}): {[str(t) for t in tile_set.tiles]}")
        
        print(f"\nPool: {len(self.tiles_deck)} tiles remaining")
        print(f"Melded: P0={self.has_melded[0]}, P1={self.has_melded[1]}")
        
        if self.game_over:
            print(f"\n{'='*60}")
            if self.winner is not None:
                print(f"GAME OVER! Winner: Player {self.winner}")
            else:
                print(f"GAME OVER! Tie!")
            print(f"{'='*60}")


if __name__ == "__main__":
    env = RummikubEnv(seed=42)
    state = env.reset()
    print("Initial state:")
    env.render()