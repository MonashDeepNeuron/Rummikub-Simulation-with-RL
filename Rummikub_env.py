import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
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
    color: Optional[Color]  # None for jokers
    number: Optional[int]   # None for jokers, 1-13 for normal tiles
    tile_type: TileType
    tile_id: int  # Unique identifier for each physical tile
    
    def __hash__(self):
        return hash(self.tile_id)
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.tile_id == other.tile_id
    
    def __repr__(self):
        if self.tile_type == TileType.JOKER:
            return "JOKER"
        # Use different cases to distinguish colors:
        # BLUE = lowercase 'b', BLACK = uppercase 'B'
        # RED = 'R', ORANGE = 'O'
        color_map = {
            Color.RED: 'R',
            Color.BLUE: 'b',      # lowercase for BLUE
            Color.BLACK: 'B',     # uppercase for BLACK
            Color.ORANGE: 'O'
        }
        return f"{color_map[self.color]}{self.number}"
    
    def get_value(self) -> int:
        """Returns the point value of the tile"""
        if self.tile_type == TileType.JOKER:
            return 30  # Joker penalty
        return self.number

@dataclass
class TileSet:
    """Represents a set of tiles on the table (either a group or a run)"""
    tiles: List[Tile]
    set_type: str  # "group" or "run"
    
    def is_valid(self) -> bool:
        """Check if this set is valid according to Rummikub rules"""
        if len(self.tiles) < 3:
            return False
        
        # Check for duplicate tiles (same tile_id)
        tile_ids = [t.tile_id for t in self.tiles]
        if len(tile_ids) != len(set(tile_ids)):
            return False  # Duplicate tiles not allowed!
            
        if self.set_type == "group":
            return self._is_valid_group()
        elif self.set_type == "run":
            return self._is_valid_run()
        return False
    
    def _is_valid_group(self) -> bool:
        """
        Check if tiles form a valid group.
        Rules: 3-4 tiles, same number, different colors, no duplicates
        """
        if len(self.tiles) < 3 or len(self.tiles) > 4:
            return False
        
        numbers = []
        colors = []
        joker_count = 0
        
        for tile in self.tiles:
            if tile.tile_type == TileType.JOKER:
                joker_count += 1
            else:
                numbers.append(tile.number)
                colors.append(tile.color)
        
        # All non-joker tiles must have the same number
        if len(numbers) > 0 and len(set(numbers)) > 1:
            return False
        
        # All non-joker tiles must have different colors (no duplicates)
        if len(colors) != len(set(colors)):
            return False
        
        # A group can have at most 4 tiles (one per color)
        if len(self.tiles) > 4:
            return False
        
        return True
    
    def _is_valid_run(self) -> bool:
        """
        Check if tiles form a valid run.
        Rules: 3+ consecutive numbers, same color, no duplicates
        """
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
        
        # All non-joker tiles must have the same color
        if len(colors) > 0 and len(set(colors)) > 1:
            return False
        
        # Check for duplicate numbers (can't have B7, B7 in same run)
        if len(numbers) != len(set(numbers)):
            return False
        
        # Check if numbers form a consecutive sequence
        if len(numbers) > 0:
            numbers.sort()
            
            # With jokers, we need to check if the gaps can be filled
            min_num = numbers[0]
            max_num = numbers[-1]
            expected_length = max_num - min_num + 1
            
            # Total tiles should equal the range
            if expected_length != len(self.tiles):
                return False
            
            # Check that we have the right number of jokers to fill gaps
            all_numbers_in_range = set(range(min_num, max_num + 1))
            missing_numbers = all_numbers_in_range - set(numbers)
            
            if len(missing_numbers) != joker_count:
                return False
        
        return True
    
    def get_value(self) -> int:
        """Returns the total value of tiles in this set (jokers count as 0)"""
        total = 0
        for tile in self.tiles:
            if tile.tile_type == TileType.JOKER:
                # For initial meld calculation, joker represents a tile value
                # But for hand penalty, it's 30
                # Here we return 0 and handle it contextually
                total += 0
            else:
                total += tile.number
        return total
    
    def get_meld_value(self) -> int:
        """
        Returns value for initial meld purposes.
        For initial meld, jokers take the value they represent.
        """
        if self.set_type == "group":
            # In a group, joker represents the same number as other tiles
            non_joker_tiles = [t for t in self.tiles if t.tile_type != TileType.JOKER]
            if non_joker_tiles:
                number = non_joker_tiles[0].number
                return number * len(self.tiles)
            return 0
        elif self.set_type == "run":
            # In a run, calculate the sum including joker values
            total = 0
            non_joker_tiles = [t for t in self.tiles if t.tile_type != TileType.JOKER]
            if non_joker_tiles:
                non_joker_tiles.sort(key=lambda t: t.number)
                min_num = non_joker_tiles[0].number
                max_num = non_joker_tiles[-1].number
                # Sum of arithmetic sequence
                expected_length = len(self.tiles)
                actual_min = min_num - sum(1 for t in self.tiles[:self.tiles.index(non_joker_tiles[0])] 
                                          if t.tile_type == TileType.JOKER)
                total = sum(range(actual_min, actual_min + expected_length))
            return total
        return 0


class RummikubAction:
    """Represents an action in Rummikub"""
    def __init__(self, action_type: str, tiles: List[Tile] = None, 
                 sets: List[TileSet] = None, table_config: List[TileSet] = None):
        """
        action_type: 'draw', 'initial_meld', 'play'
        tiles: tiles from hand being played
        sets: new sets being formed
        table_config: complete table configuration after manipulation
        """
        self.action_type = action_type
        self.tiles = tiles or []
        self.sets = sets or []
        self.table_config = table_config


class RummikubEnv:
    """Rummikub Environment for Reinforcement Learning"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.tiles_deck: List[Tile] = []
        self.player_hands: List[List[Tile]] = [[], []]  # 2 players
        self.table: List[TileSet] = []  # Sets on the table
        self.current_player: int = 0
        self.has_melded: List[bool] = [False, False]  # Track initial meld
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_count: int = 0
        
        # For reward calculation
        self.previous_hand_values: List[int] = [0, 0]
        
        # Import the hybrid action generator (to be implemented)
        self.action_generator = None  # Will be set externally
        
        self._initialize_deck()
    
    def _initialize_deck(self):
        """Create the full deck of 106 tiles"""
        self.tiles_deck = []
        tile_id = 0
        
        # Create numbered tiles (2 copies of each color-number combination)
        for copy in range(2):
            for color in Color:
                for number in range(1, 14):
                    tile = Tile(color=color, number=number, 
                               tile_type=TileType.NORMAL, tile_id=tile_id)
                    self.tiles_deck.append(tile)
                    tile_id += 1
        
        # Create 2 jokers
        for _ in range(2):
            tile = Tile(color=None, number=None, 
                       tile_type=TileType.JOKER, tile_id=tile_id)
            self.tiles_deck.append(tile)
            tile_id += 1
    
    def reset(self) -> Dict:
        """Reset the environment for a new game"""
        # Shuffle the deck
        self.rng.shuffle(self.tiles_deck)
        
        # Deal 14 tiles to each player
        self.player_hands = [[], []]
        for player in range(2):
            self.player_hands[player] = self.tiles_deck[:14]
            self.tiles_deck = self.tiles_deck[14:]
        
        # Reset game state
        self.table = []
        self.current_player = 0
        self.has_melded = [False, False]
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        
        # Initialize hand values for reward calculation
        self.previous_hand_values = [
            self._calculate_hand_value(0),
            self._calculate_hand_value(1)
        ]
        
        return self._get_state()
    
    def _calculate_hand_value(self, player: int) -> int:
        """Calculate total value of tiles in player's hand"""
        return sum(tile.get_value() for tile in self.player_hands[player])
    
    def _count_jokers_in_hand(self, player: int) -> int:
        """Count number of jokers in player's hand"""
        return sum(1 for tile in self.player_hands[player] 
                   if tile.tile_type == TileType.JOKER)
    
    def _get_state(self) -> Dict:
        """
        Return the current game state as defined by user:
        1. Board configuration and current player's hand
        2. Number of tiles opponent has
        3. Number of tiles to be drawn (pool size)
        """
        return {
            # Core state components
            'my_hand': copy.deepcopy(self.player_hands[self.current_player]),
            'table': copy.deepcopy(self.table),
            'opponent_tile_count': len(self.player_hands[1 - self.current_player]),
            'pool_size': len(self.tiles_deck),
            
            # Additional useful info
            'current_player': self.current_player,
            'has_melded': self.has_melded.copy(),
            'game_over': self.game_over,
            'winner': self.winner,
            'turn_count': self.turn_count
        }
    
    def get_legal_actions(self, player: int) -> List[RummikubAction]:
        """
        Get all legal actions for the current player.
        
        TODO: This method should call your HybridActionGenerator.
        
        Instructions:
        1. Create an instance of HybridActionGenerator (see separate file)
        2. Call: self.action_generator.generate_all_legal_actions(
                    hand_tiles=self.player_hands[player],
                    table_sets=self.table,
                    has_melded=self.has_melded[player],
                    pool_size=len(self.tiles_deck)
                 )
        3. The generator will return a list of RummikubAction objects
        
        For now, this returns basic actions for testing.
        """
        legal_actions = []
        
        # Option 1: Draw a tile (always legal if pool not empty)
        if len(self.tiles_deck) > 0:
            legal_actions.append(RummikubAction(action_type='draw'))
        
        # Option 2: Use action generator if available
        if self.action_generator is not None:
            legal_actions.extend(
                self.action_generator.generate_all_legal_actions(
                    hand_tiles=self.player_hands[player],
                    table_sets=self.table,
                    has_melded=self.has_melded[player],
                    pool_size=len(self.tiles_deck)
                )
            )
        else:
            # Fallback: basic action generation for testing
            if self.has_melded[player]:
                legal_actions.extend(self._find_valid_plays(player))
            else:
                legal_actions.extend(self._find_valid_initial_melds(player))
        
        return legal_actions
    
    def _find_valid_initial_melds(self, player: int) -> List[RummikubAction]:
        """
        TODO: This is a placeholder. Should be replaced by action generator.
        
        What you should do:
        - Remove this method entirely, OR
        - Keep it as a simple fallback for testing
        
        The HybridActionGenerator will handle this properly.
        """
        # Simple placeholder - just returns empty list
        return []
    
    def _find_valid_plays(self, player: int) -> List[RummikubAction]:
        """
        TODO: This is a placeholder. Should be replaced by action generator.
        
        What you should do:
        - Remove this method entirely, OR  
        - Keep it as a simple fallback for testing
        
        The HybridActionGenerator will handle this properly.
        """
        # Simple placeholder - just returns empty list
        return []
    
    def step(self, action: RummikubAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute an action and return (state, reward, done, info)
        
        Updated reward function:
        1. R_t = (Sum of hand at t-1) - (Sum of hand at t)
        2. Win by empty hand: R_T = 200 + sum of opponent's hand
        3. Win by lowest hand: R_T = +10
        4. Lose by lowest hand: R_T = -10
        5. Ice-breaking bonus: +20
        6. Drawing penalty: -5
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        # Store hand value before action
        hand_value_before = self._calculate_hand_value(self.current_player)
        
        # Initialize info dictionary
        info = {
            'action_type': action.action_type,
            'tiles_played': 0,
            'drew_tile': False,
            'ice_broken': False,
            'joker_retrieved': False,
            'manipulation_occurred': False,
            'draw_penalty_applied': False,
            'invalid_action': False,
            'hand_size_before': len(self.player_hands[self.current_player]),
            'hand_value_before': hand_value_before,
        }
        
        reward = 0
        
        # Execute action
        if action.action_type == 'draw':
            # Draw a tile from the pool
            if len(self.tiles_deck) > 0:
                drawn_tile = self.tiles_deck.pop(0)
                self.player_hands[self.current_player].append(drawn_tile)
                info['drew_tile'] = True
                info['draw_penalty_applied'] = True
            else:
                # Invalid action - no tiles to draw
                info['invalid_action'] = True
            
        elif action.action_type == 'initial_meld':
            # Player makes initial meld
            if self._validate_initial_meld(action):
                self._apply_meld(action)
                self.has_melded[self.current_player] = True
                info['ice_broken'] = True
                info['tiles_played'] = len(action.tiles)
            else:
                # Invalid meld
                info['invalid_action'] = True
                
        elif action.action_type == 'play':
            # Player plays tiles (after initial meld)
            if self._validate_play(action):
                info['tiles_played'] = len(action.tiles)
                # Check if manipulation occurred
                if len(action.table_config) != len(self.table) + len(action.sets):
                    info['manipulation_occurred'] = True
                self._apply_play(action)
            else:
                info['invalid_action'] = True
        
        # Calculate hand value after action
        hand_value_after = self._calculate_hand_value(self.current_player)
        info['hand_value_after'] = hand_value_after
        info['hand_size_after'] = len(self.player_hands[self.current_player])
        
        # Apply reward based on action type (as per user specification)
        if not info['invalid_action']:
            # Base reward: reduction in hand value
            base_reward = hand_value_before - hand_value_after
            
            if action.action_type == 'draw':
                # R_t = (hand_before - hand_after) - 5
                reward = base_reward - 5
            elif action.action_type == 'initial_meld':
                # R_t = (hand_before - hand_after) + 20
                reward = base_reward + 20
            elif action.action_type == 'play':
                # R_t = (hand_before - hand_after)
                reward = base_reward
            else:
                # Fallback
                reward = base_reward
        
        # Check termination conditions
        done = False
        
        # Condition 1: Current player has no tiles left (WIN by empty hand)
        if len(self.player_hands[self.current_player]) == 0:
            self.game_over = True
            self.winner = self.current_player
            done = True
            
            # Terminal reward: 200 + opponent's hand value
            opponent = 1 - self.current_player
            opponent_hand_value = self._calculate_hand_value(opponent)
            reward = 200 + opponent_hand_value
            
            info['final_opponent_hand_value'] = opponent_hand_value
            info['win_type'] = 'emptied_hand'
            info['winner'] = self.current_player
        
        # Condition 2: No more tiles in pool
        elif len(self.tiles_deck) == 0:
            # Check if anyone can make a move
            current_can_play = len(self.get_legal_actions(self.current_player)) > 1  # >1 because draw always exists
            next_player = 1 - self.current_player
            
            # Switch to next player temporarily to check their actions
            temp_current = self.current_player
            self.current_player = next_player
            next_can_play = len(self.get_legal_actions(next_player)) > 1
            self.current_player = temp_current
            
            # If neither player can play, game ends
            if not current_can_play and not next_can_play:
                self.game_over = True
                done = True
                
                # Determine winner by lowest hand value
                player_value = self._calculate_hand_value(self.current_player)
                opponent_value = self._calculate_hand_value(1 - self.current_player)
                
                info['jokers_in_hand'] = self._count_jokers_in_hand(self.current_player)
                info['final_my_hand_value'] = player_value
                info['final_opponent_hand_value'] = opponent_value
                
                if player_value < opponent_value:
                    # Current player wins
                    self.winner = self.current_player
                    reward = 10  # Winner gets +10
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = self.current_player
                elif opponent_value < player_value:
                    # Current player loses
                    self.winner = 1 - self.current_player
                    reward = -10  # Loser gets -10
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = 1 - self.current_player
                else:
                    # Tie - no winner
                    self.winner = None
                    reward = 0
                    info['win_type'] = 'tie'
                    info['winner'] = None
        
        # Update previous hand value for next turn
        self.previous_hand_values[self.current_player] = hand_value_after
        
        # Switch to next player
        if not done:
            self.current_player = 1 - self.current_player
            self.turn_count += 1
        
        state = self._get_state()
        return state, reward, done, info
    
    def _validate_initial_meld(self, action: RummikubAction) -> bool:
        """Validate that initial meld is legal (30+ points)"""
        if not action.sets:
            return False
        
        # Calculate total value (for initial meld, jokers count as their represented value)
        total_value = sum(s.get_meld_value() for s in action.sets)
        
        # Check all sets are valid
        all_valid = all(s.is_valid() for s in action.sets)
        
        # Check tiles come from player's hand
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] 
                                for t in action.tiles)
        
        # Check no tiles are used from table (initial meld can't use table)
        # action.table_config should only contain the new sets
        
        return total_value >= 30 and all_valid and all_tiles_in_hand
    
    def _validate_play(self, action: RummikubAction) -> bool:
        """Validate that a play is legal"""
        # Check that all resulting sets on table are valid
        if action.table_config is None:
            return False
        
        # Check tiles come from player's hand
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] 
                                for t in action.tiles)
        
        # Check all sets in new configuration are valid
        all_sets_valid = all(s.is_valid() for s in action.table_config)
        
        # Check that all tiles are accounted for (hand tiles + table tiles = new table tiles)
        # Count tiles: tiles from table + tiles from hand = tiles in new table
        table_tiles = []
        for tile_set in self.table:
            table_tiles.extend(tile_set.tiles)
        
        new_table_tiles = []
        for tile_set in action.table_config:
            new_table_tiles.extend(tile_set.tiles)
        
        # Tiles in new table should be: old table tiles + tiles played from hand
        expected_tile_ids = set(t.tile_id for t in table_tiles) | set(t.tile_id for t in action.tiles)
        actual_tile_ids = set(t.tile_id for t in new_table_tiles)
        
        tiles_match = expected_tile_ids == actual_tile_ids
        
        return all_tiles_in_hand and all_sets_valid and tiles_match
    
    def _apply_meld(self, action: RummikubAction):
        """Apply initial meld to game state"""
        # Remove tiles from hand
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        
        # Add sets to table
        self.table.extend(action.sets)
    
    def _apply_play(self, action: RummikubAction):
        """Apply a play to game state"""
        # Remove tiles from hand
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        
        # Update table with new configuration
        self.table = action.table_config
    
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
        print(f"Initial meld status: Player 0={self.has_melded[0]}, Player 1={self.has_melded[1]}")
        
        if self.game_over:
            print(f"\n{'='*60}")
            if self.winner is not None:
                print(f"GAME OVER! Winner: Player {self.winner}")
            else:
                print(f"GAME OVER! Tie!")
            print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Example turn: draw a tile
    legal_actions = env.get_legal_actions(env.current_player)
    if legal_actions:
        action = legal_actions[0]  # Just draw for this example
        state, reward, done, info = env.step(action)
        print(f"\nAction taken: {action.action_type}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        env.render()