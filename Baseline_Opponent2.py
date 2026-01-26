from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools

try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("ERROR: ILP opponent requires ortools. Install with: pip install ortools")

from Rummikub_env import Tile, TileSet, RummikubAction, Color, TileType

# Assuming these are imported from the provided Rummikub_env.py
# from Rummikub_env import Tile, TileSet, RummikubAction, Color, TileType

def get_key(tile: Tile) -> Tuple[Optional[int], Optional[int]]:
    if tile.tile_type == TileType.JOKER:
        return None, None
    return tile.color.value, tile.number

@dataclass
class PossibleMeld:
    set_type: str  # "group" or "run"
    required_tiles: List[Tuple[Optional[int], Optional[int]]]  # List of (color_value, number), None for jokers

class RummikubILPSolver:
    """
    Python implementation of the Rummikub solver using Integer Linear Programming (ILP),
    translated and adapted from the Haskell code in gregorias/rummikubsolver/src.
    
    Key components translated/adapted:
    - Data structures: Tile representations adapted to Python env's Tile class.
    - Combinatorics: Generation of all possible valid groups and runs (including joker placements).
    - Game logic: Handling of table and rack (hand) states, ensuring table solvability.
    - Solver: ILP formulation to maximize hand tiles played while maintaining valid table configuration.
    
    Parsing (Text/Megaparsec) and UI (Cli.hs, Interface/) are omitted as they are not needed for the core solver.
    Game/Core.hs, Set.hs, State.hs, TileCountArray.hs concepts are incorporated into counts and validation.
    """
    
    def __init__(self):
        self.all_melds = self._generate_all_possible_melds()
    
    def _generate_all_possible_melds(self) -> List[PossibleMeld]:
        """
        Generates all possible valid melds (groups and runs), including variations with up to 2 jokers.
        Adapted from Combinatorics.hs and Game/Set.hs.
        """
        melds: List[PossibleMeld] = []
        colors = range(4)  # 0: RED, 1: BLUE, 2: BLACK, 3: ORANGE
        
        # Generate groups (same number, different colors, 3-4 tiles)
        for n in range(1, 14):
            for size in [3, 4]:
                for num_jokers in range(3):  # Limit to 0-2 jokers
                    if num_jokers > size:
                        continue
                    num_colors = size - num_jokers
                    if num_colors < 1 or num_colors > 4:
                        continue
                    for s in itertools.combinations(colors, num_colors):
                        required = [(c, n) for c in s] + [(None, None)] * num_jokers
                        melds.append(PossibleMeld("group", required))
        
        # Generate runs (same color, consecutive numbers, 3+ tiles)
        for col in colors:
            for start in range(1, 14):
                for length in range(3, 15 - start):
                    for num_jokers in range(3):  # Limit to 0-2 jokers
                        if num_jokers >= length:  # At least one non-joker
                            continue
                        for joker_pos in itertools.combinations(range(length), num_jokers):
                            required = []
                            for pos in range(length):
                                if pos in joker_pos:
                                    required.append((None, None))
                                else:
                                    num = start + pos
                                    required.append((col, num))
                            melds.append(PossibleMeld("run", required))
        
        return melds
    
    def solve(self, hand: List[Tile], table: List[TileSet], has_melded: bool) -> Optional[RummikubAction]:
        """
        Solves for the optimal play using ILP: maximizes the number of hand tiles placed on the table
        while ensuring the entire table + played tiles can be rearranged into valid sets.
        
        Adapted from Solver.hs and Game.hs/State.hs.
        
        Returns a RummikubAction if a valid play is found, else None (implying draw).
        """
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            raise RuntimeError("Failed to create solver.")
        
        # Collect tiles
        table_tiles = [t for s in table for t in s.tiles]
        hand_tiles = hand
        
        # Get unique tile types (only those present)
        tile_types: Set[Tuple[Optional[int], Optional[int]]] = set(get_key(t) for t in table_tiles + hand_tiles)
        
        # Compute counts (adapted from TileCountArray.hs)
        table_count: Dict[Tuple[Optional[int], Optional[int]], int] = defaultdict(int)
        hand_count: Dict[Tuple[Optional[int], Optional[int]], int] = defaultdict(int)
        for t in table_tiles:
            table_count[get_key(t)] += 1
        for t in hand_tiles:
            hand_count[get_key(t)] += 1
        
        # Filter possible melds: only those where required tile types are available in sufficient quantity
        possible_melds = [
            m for m in self.all_melds
            if all(
                table_count.get(tt, 0) + hand_count.get(tt, 0) >= m.required_tiles.count(tt)
                for tt in set(m.required_tiles)
            )
        ]
        
        num_melds = len(possible_melds)
        if num_melds == 0:
            return None  # No possible melds
        
        # ILP variables: one bool var per possible meld
        x = [solver.BoolVar(f'x[{i}]') for i in range(num_melds)]
        
        # Objective: Maximize total tiles on table (equivalent to max hand tiles used, since table tiles are fixed)
        solver.Maximize(solver.Sum(x[i] * len(possible_melds[i].required_tiles) for i in range(num_melds)))
        
        # Constraints: For each tile type (including jokers), must use all table tiles, at most table + hand
        for tt in tile_types:
            used = solver.Sum(x[i] * possible_melds[i].required_tiles.count(tt) for i in range(num_melds))
            solver.Add(used >= table_count[tt])  # Guarantee table solvability
            solver.Add(used <= table_count[tt] + hand_count[tt])
        
        # Solve
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        # Get selected melds
        selected_indices = [i for i in range(num_melds) if x[i].solution_value() > 0.5]
        
        # Compute used counts
        used_count: Dict[Tuple[Optional[int], Optional[int]], int] = defaultdict(int)
        for i in selected_indices:
            for tt in possible_melds[i].required_tiles:
                used_count[tt] += 1
        
        # Compute hand tiles used
        hand_used_count: Dict[Tuple[Optional[int], Optional[int]], int] = defaultdict(int)
        for tt in tile_types:
            hand_used_count[tt] = max(0, used_count[tt] - table_count[tt])
        
        total_hand_used = sum(hand_used_count.values())
        if total_hand_used == 0:
            return None  # No tiles played
        
        # Select specific played tiles from hand
        hand_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in hand_tiles:
            hand_by_type[get_key(t)].append(t)
        
        played_tiles: List[Tile] = []
        for tt, cnt in hand_used_count.items():
            if cnt > len(hand_by_type[tt]):
                raise ValueError("Count mismatch in hand tiles.")
            played_tiles.extend(hand_by_type[tt][:cnt])
        
        # Prepare all used tiles
        all_used_tiles = table_tiles + played_tiles
        
        # Assign tiles to melds (build new table config)
        available_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in all_used_tiles:
            available_by_type[get_key(t)].append(t)
        
        new_sets: List[TileSet] = []
        for i in selected_indices:
            meld = possible_melds[i]
            set_tiles: List[Tile] = []
            for req in meld.required_tiles:
                tiles_list = available_by_type[req]
                if not tiles_list:
                    raise ValueError("Tile assignment failed: insufficient tiles.")
                set_tiles.append(tiles_list.pop())
            
            # For runs, sort tiles by number for better representation (optional, but helps)
            if meld.set_type == "run":
                def sort_key(t: Tile):
                    return t.number if t.number is not None else -1  # Place jokers at start
                set_tiles.sort(key=sort_key)
            
            new_set = TileSet(tiles=set_tiles, set_type=meld.set_type)
            if not new_set.is_valid():
                print("Invalid set detected in opponent solve:")
                print(f"Template type: {meld.set_type}")
                print(f"Template pattern: {meld.required_tiles}")
                print(f"Assigned tiles: {[str(t) for t in set_tiles]}")
                print(f"Tile IDs: {[t.tile_id for t in set_tiles]}")
                print(f"Joker count: {sum(1 for t in set_tiles if t.tile_type == TileType.JOKER)}")
                print(f"Non-joker tiles: {[str(t) for t in set_tiles if t.tile_type != TileType.JOKER]}")
                raise ValueError("Generated invalid set.")
            new_sets.append(new_set)
        
        # For initial meld, check total value >= 30
        total_value = sum(new_set.get_meld_value() for new_set in new_sets)
        if not has_melded and total_value < 30:
            return None
        
        # Create action
        if not has_melded:
            return RummikubAction(action_type='initial_meld', tiles=played_tiles, sets=new_sets)
        else:
            return RummikubAction(action_type='play', tiles=played_tiles, table_config=new_sets)