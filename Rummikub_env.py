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
            min_num, max_num = numbers[0], numbers[-1]
            internal_missing = (max_num - min_num + 1) - len(numbers)
            if internal_missing > joker_count:
                return False
        return True
    
    def get_value(self) -> int:
        return sum(t.number for t in self.tiles if t.tile_type != TileType.JOKER)
    
    def get_meld_value(self) -> int:
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
                jokers_before = sum(1 for t in self.tiles[:self.tiles.index(non_joker[0])] 
                                   if t.tile_type == TileType.JOKER)
                actual_min = min_num - jokers_before
                return sum(range(actual_min, actual_min + len(self.tiles)))
            return 0
        return 0


class RummikubAction:
    def __init__(self, action_type: str, tiles: List[Tile] = None, 
                 sets: List[TileSet] = None, table_config: List[TileSet] = None):
        self.action_type = action_type
        self.tiles = tiles or []
        self.sets = sets or []
        self.table_config = table_config


class RummikubEnv:
    """
    Rummikub Environment with ALL reward logic.
    
    REWARD CONFIGURATION (class variables):
        REWARD_BASE_MULTIPLIER = 2.0
        REWARD_ICE_BREAK = 30.0
        REWARD_MANIPULATION = 10.0
        REWARD_DRAW_PENALTY = -5.0
        REWARD_WIN_EMPTY_HAND = 300.0
        REWARD_WIN_LOWEST_HAND = 50.0
        REWARD_LOSE_EMPTY_HAND = -200.0
        REWARD_LOSE_LOWEST_HAND = -75.0
    
    REWARD FORMULA:
        Acting player's turn (play/meld):
            intermediate = BASE_MULTIPLIER * (hand_before - hand_after) + bonuses
        
        Acting player's turn (draw):
            intermediate = BASE_MULTIPLIER * (hand_before - hand_after) + DRAW_PENALTY
        
        Terminal (acting player wins by empty hand):
            acting_player_reward = intermediate + WIN_EMPTY_HAND + opponent_hand
            opponent_reward = LOSE_EMPTY_HAND - opponent_hand  # ALWAYS < -200
        
        Terminal (acting player wins by lowest hand):
            acting_player_reward = intermediate + WIN_LOWEST_HAND
            opponent_reward = LOSE_LOWEST_HAND
    
    The step() function returns rewards for BOTH players:
        info['reward_for_player_0']
        info['reward_for_player_1']
    
    Use info['reward_for_player_{your_player_index}'] to get your reward.
    """
    
    # =========================================================================
    # REWARD CONFIGURATION - All reward params in ONE place
    # =========================================================================
    REWARD_BASE_MULTIPLIER = 2.0
    REWARD_ICE_BREAK = 30.0
    REWARD_MANIPULATION = 10.0
    REWARD_DRAW_PENALTY = -5.0
    REWARD_WIN_EMPTY_HAND = 300.0
    REWARD_WIN_LOWEST_HAND = 50.0
    REWARD_LOSE_EMPTY_HAND = -200.0
    REWARD_LOSE_LOWEST_HAND = -75.0
    
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
        self.tiles_deck = []
        tile_id = 0
        for _ in range(2):
            for color in Color:
                for number in range(1, 14):
                    self.tiles_deck.append(Tile(color, number, TileType.NORMAL, tile_id))
                    tile_id += 1
        for _ in range(2):
            self.tiles_deck.append(Tile(None, None, TileType.JOKER, tile_id))
            tile_id += 1
    
    def reset(self) -> Dict:
        self.tiles_deck = []
        self._initialize_deck()
        self.rng.shuffle(self.tiles_deck)
        self.player_hands = [[], []]
        for player in range(2):
            for _ in range(14):
                self.player_hands[player].append(self.tiles_deck.pop())
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
        return sum(1 for t in self.player_hands[player] if t.tile_type == TileType.JOKER)
    
    def _get_state(self) -> Dict:
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
        if self.action_generator is None:
            raise ValueError("Action generator not set")
        
        all_ids = [t.tile_id for ts in self.table for t in ts.tiles]
        if len(all_ids) != len(set(all_ids)):
            if len(self.tiles_deck) > 0:
                return [RummikubAction(action_type='draw')]
            return []
        
        actions = self.action_generator.generate_all_legal_actions(
            hand_tiles=self.player_hands[player],
            table_sets=copy.deepcopy(self.table),
            has_melded=self.has_melded[player],
            pool_size=len(self.tiles_deck)
        )
        
        if len(self.tiles_deck) > 0:
            actions.append(RummikubAction(action_type='draw'))
        
        return actions
    
    def step(self, action: RummikubAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info).
        
        The 'reward' returned is from the ACTING PLAYER's perspective.
        
        CRITICAL: Use info['reward_for_player_X'] where X is YOUR player index
        to get the correct reward for your agent.
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        acting_player = self.current_player
        opponent = 1 - acting_player
        hand_value_before = self._calculate_hand_value(acting_player)
        had_melded_before = self.has_melded[acting_player]
        
        info = {
            'action_type': action.action_type,
            'tiles_played': 0,
            'drew_tile': False,
            'ice_broken': False,
            'manipulation_occurred': False,
            'invalid_action': False,
            'hand_size_before': len(self.player_hands[acting_player]),
            'hand_value_before': hand_value_before,
            'acting_player': acting_player,
        }
        
        # Execute action
        if action.action_type == 'draw':
            if len(self.tiles_deck) > 0:
                self.player_hands[acting_player].append(self.tiles_deck.pop(0))
                info['drew_tile'] = True
            else:
                info['invalid_action'] = True
        
        elif action.action_type == 'initial_meld':
            if action.sets:
                all_ids = [t.tile_id for s in action.sets for t in s.tiles]
                if len(all_ids) != len(set(all_ids)):
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
                all_ids = [t.tile_id for s in action.table_config for t in s.tiles]
                if len(all_ids) != len(set(all_ids)):
                    info['invalid_action'] = True
            
            if not info.get('invalid_action') and self._validate_play(action):
                info['tiles_played'] = len(action.tiles)
                if len(action.table_config) != len(self.table) + len(action.sets or []):
                    info['manipulation_occurred'] = True
                self._apply_play(action)
            else:
                info['invalid_action'] = True
        
        hand_value_after = self._calculate_hand_value(acting_player)
        info['hand_value_after'] = hand_value_after
        info['hand_size_after'] = len(self.player_hands[acting_player])
        
        # =====================================================================
        # REWARD CALCULATION - ALL LOGIC HERE
        # =====================================================================
        reward_for_player = [0.0, 0.0]
        done = False
        
        # Compute intermediate reward for acting player
        if not info['invalid_action']:
            hand_change = hand_value_before - hand_value_after
            intermediate = self.REWARD_BASE_MULTIPLIER * hand_change
            
            if action.action_type == 'draw':
                intermediate += self.REWARD_DRAW_PENALTY
            else:
                if info['ice_broken'] and not had_melded_before:
                    intermediate += self.REWARD_ICE_BREAK
                if info['manipulation_occurred']:
                    intermediate += self.REWARD_MANIPULATION
            
            reward_for_player[acting_player] = intermediate
        
        # Check termination: acting player emptied hand
        if len(self.player_hands[acting_player]) == 0:
            self.game_over = True
            self.winner = acting_player
            done = True
            
            opponent_hand_value = self._calculate_hand_value(opponent)
            
            info['final_my_hand_value'] = 0
            info['final_opponent_hand_value'] = opponent_hand_value
            info['win_type'] = 'emptied_hand'
            info['winner'] = acting_player
            
            # Winner (acting player): intermediate + terminal
            winner_terminal = self.REWARD_WIN_EMPTY_HAND + opponent_hand_value
            reward_for_player[acting_player] += winner_terminal
            
            # Loser (opponent): ONLY terminal, no intermediate
            # This is ALWAYS < REWARD_LOSE_EMPTY_HAND since opponent_hand_value > 0
            loser_terminal = self.REWARD_LOSE_EMPTY_HAND - opponent_hand_value
            reward_for_player[opponent] = loser_terminal
        
        # Check termination: pool empty, no one can play
        elif len(self.tiles_deck) == 0:
            current_can = len(self.get_legal_actions(acting_player)) > 0
            
            temp = self.current_player
            self.current_player = opponent
            next_can = len(self.get_legal_actions(opponent)) > 0
            self.current_player = temp
            
            if not current_can and not next_can:
                self.game_over = True
                done = True
                
                p0_hand = self._calculate_hand_value(0)
                p1_hand = self._calculate_hand_value(1)
                
                info['final_my_hand_value'] = self._calculate_hand_value(acting_player)
                info['final_opponent_hand_value'] = self._calculate_hand_value(opponent)
                info['jokers_in_hand'] = self._count_jokers_in_hand(acting_player)
                
                if p0_hand < p1_hand:
                    self.winner = 0
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = 0
                    reward_for_player[0] += self.REWARD_WIN_LOWEST_HAND
                    reward_for_player[1] = self.REWARD_LOSE_LOWEST_HAND
                elif p1_hand < p0_hand:
                    self.winner = 1
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = 1
                    reward_for_player[1] += self.REWARD_WIN_LOWEST_HAND
                    reward_for_player[0] = self.REWARD_LOSE_LOWEST_HAND
                else:
                    self.winner = None
                    info['win_type'] = 'tie'
                    info['winner'] = None
        
        # CRITICAL: Store rewards for BOTH players
        info['reward_for_player_0'] = reward_for_player[0]
        info['reward_for_player_1'] = reward_for_player[1]
        
        # Return reward from acting player's perspective (for compatibility)
        reward = reward_for_player[acting_player]
        
        self.previous_hand_values[acting_player] = hand_value_after
        
        if not done:
            self.current_player = 1 - self.current_player
            self.turn_count += 1
        
        return self._get_state(), reward, done, info
    
    def _validate_initial_meld(self, action: RummikubAction) -> bool:
        if not action.sets:
            return False
        total = sum(s.get_meld_value() for s in action.sets)
        valid = all(s.is_valid() for s in action.sets)
        in_hand = all(t in self.player_hands[self.current_player] for t in action.tiles)
        return total >= 30 and valid and in_hand
    
    def _validate_play(self, action: RummikubAction) -> bool:
        if action.table_config is None:
            return False
        in_hand = all(t in self.player_hands[self.current_player] for t in action.tiles)
        valid = all(s.is_valid() for s in action.table_config)
        
        table_tiles = [t for ts in self.table for t in ts.tiles]
        new_tiles = [t for ts in action.table_config for t in ts.tiles]
        new_ids = [t.tile_id for t in new_tiles]
        
        if len(new_ids) != len(set(new_ids)):
            return False
        
        expected = set(t.tile_id for t in table_tiles) | set(t.tile_id for t in action.tiles)
        actual = set(new_ids)
        
        return in_hand and valid and expected == actual
    
    def _apply_meld(self, action: RummikubAction):
        for t in action.tiles:
            self.player_hands[self.current_player].remove(t)
        self.table.extend(copy.deepcopy(action.sets))
        self._validate_table_integrity()
    
    def _apply_play(self, action: RummikubAction):
        for t in action.tiles:
            self.player_hands[self.current_player].remove(t)
        self.table = copy.deepcopy(action.table_config)
        self._validate_table_integrity()
    
    def _validate_table_integrity(self):
        all_ids = [t.tile_id for ts in self.table for t in ts.tiles]
        if len(all_ids) != len(set(all_ids)):
            from collections import Counter
            dups = {k: v for k, v in Counter(all_ids).items() if v > 1}
            raise ValueError(f"Table corruption: {dups}")
    
    def render(self):
        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count} - Player {self.current_player}'s turn")
        print(f"{'='*60}")
        for i, hand in enumerate(self.player_hands):
            val = self._calculate_hand_value(i)
            if i == self.current_player:
                print(f"Player {i} ({len(hand)} tiles, val={val}): {[str(t) for t in hand]}")
            else:
                print(f"Player {i} ({len(hand)} tiles, val={val}): [hidden]")
        print(f"Table: {len(self.table)} sets")
        print(f"Pool: {len(self.tiles_deck)} | Melded: {self.has_melded}")


if __name__ == "__main__":
    # Quick test
    env = RummikubEnv(seed=42)
    print("Reward configuration:")
    print(f"  LOSE_EMPTY_HAND = {env.REWARD_LOSE_EMPTY_HAND}")
    print(f"  When opponent wins with agent having 150 in hand:")
    print(f"  Agent reward = {env.REWARD_LOSE_EMPTY_HAND} - 150 = {env.REWARD_LOSE_EMPTY_HAND - 150}")
    print(f"  This is ALWAYS < {env.REWARD_LOSE_EMPTY_HAND}")