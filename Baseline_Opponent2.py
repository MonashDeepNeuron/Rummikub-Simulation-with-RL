from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools
import copy

try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("ERROR: ILP opponent requires ortools. Install with: pip install ortools")

from Rummikub_env import Tile, TileSet, RummikubAction, Color, TileType


def get_key(tile: Tile) -> Tuple[Optional[int], Optional[int]]:
    """Get tile type key: (color_value, number) or (None, None) for joker."""
    if tile.tile_type == TileType.JOKER:
        return (None, None)
    return (tile.color.value, tile.number)


@dataclass
class PossibleMeld:
    set_type: str  # "group" or "run"
    required_tiles: List[Tuple[Optional[int], Optional[int]]]


class RummikubILPSolver:
    """
    ILP-based Rummikub solver that finds optimal plays.
    """
    
    def __init__(self):
        self.all_melds = self._generate_all_possible_melds()
    
    def _generate_all_possible_melds(self) -> List[PossibleMeld]:
        """Generate all possible valid meld patterns."""
        melds: List[PossibleMeld] = []
        colors = range(4)  # 0: RED, 1: BLUE, 2: BLACK, 3: ORANGE
        
        # Generate groups (same number, different colors, 3-4 tiles)
        for n in range(1, 14):
            for size in [3, 4]:
                for num_jokers in range(min(3, size)):
                    num_colors = size - num_jokers
                    if num_colors < 1 or num_colors > 4:
                        continue
                    for color_combo in itertools.combinations(colors, num_colors):
                        required = [(c, n) for c in color_combo] + [(None, None)] * num_jokers
                        melds.append(PossibleMeld("group", required))
        
        # Generate runs (same color, consecutive numbers, 3+ tiles)
        for col in colors:
            for start in range(1, 14):
                for length in range(3, 15 - start):
                    for num_jokers in range(min(3, length)):
                        if num_jokers >= length:
                            continue
                        for joker_positions in itertools.combinations(range(length), num_jokers):
                            required = []
                            for pos in range(length):
                                if pos in joker_positions:
                                    required.append((None, None))
                                else:
                                    required.append((col, start + pos))
                            melds.append(PossibleMeld("run", required))
        
        return melds
    
    def solve(self, hand: List[Tile], table: List[TileSet], has_melded: bool) -> Optional[RummikubAction]:
        """
        Find optimal play using ILP.
        Returns RummikubAction or None (meaning draw).
        """
        if not HAS_ORTOOLS:
            return None
        
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return None
        
        # ===== STEP 1: Collect and validate input tiles =====
        table_tiles: List[Tile] = []
        for ts in table:
            table_tiles.extend(ts.tiles)
        if not has_melded:
            table_tiles = [] 
        
        # Validate inputs
        table_ids = [t.tile_id for t in table_tiles]
        hand_ids = [t.tile_id for t in hand]
        
        if len(table_ids) != len(set(table_ids)):
            return None
        if len(hand_ids) != len(set(hand_ids)):
            return None
        if set(table_ids) & set(hand_ids):
            return None
        
        # ===== STEP 2: Build tile inventory by TYPE =====
        # CRITICAL: Store (tile_id -> Tile) mapping to ensure unique tiles
        all_tiles_by_id: Dict[int, Tile] = {}
        for t in table_tiles:
            all_tiles_by_id[t.tile_id] = t
        for t in hand:
            all_tiles_by_id[t.tile_id] = t
        
        # Count tiles by type
        table_count: Dict[Tuple, int] = defaultdict(int)
        hand_count: Dict[Tuple, int] = defaultdict(int)
        
        # Also track which tile_ids belong to each type
        table_ids_by_type: Dict[Tuple, List[int]] = defaultdict(list)
        hand_ids_by_type: Dict[Tuple, List[int]] = defaultdict(list)
        
        for t in table_tiles:
            tt = get_key(t)
            table_count[tt] += 1
            table_ids_by_type[tt].append(t.tile_id)
        
        for t in hand:
            tt = get_key(t)
            hand_count[tt] += 1
            hand_ids_by_type[tt].append(t.tile_id)
        
        all_types = set(table_count.keys()) | set(hand_count.keys())
        
        # ===== STEP 3: Filter feasible melds =====
        feasible_melds = []
        for m in self.all_melds:
            can_form = True
            for tt in set(m.required_tiles):
                needed = m.required_tiles.count(tt)
                available = table_count.get(tt, 0) + hand_count.get(tt, 0)
                if needed > available:
                    can_form = False
                    break
            if can_form:
                feasible_melds.append(m)
        
        if not feasible_melds:
            return None
        
        # ===== STEP 4: Set up and solve ILP =====
        num_melds = len(feasible_melds)
        x = [solver.BoolVar(f'x_{i}') for i in range(num_melds)]
        
        for tt in all_types:
            usage = solver.Sum(
                x[i] * feasible_melds[i].required_tiles.count(tt) 
                for i in range(num_melds)
            )
            solver.Add(usage >= table_count.get(tt, 0))
            solver.Add(usage <= table_count.get(tt, 0) + hand_count.get(tt, 0))
        
        solver.Maximize(solver.Sum(
            x[i] * len(feasible_melds[i].required_tiles) 
            for i in range(num_melds)
        ))
        
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        selected = [i for i in range(num_melds) if x[i].solution_value() > 0.5]
        if not selected:
            return None
        
        # ===== STEP 5: Calculate exact tile requirements =====
        type_usage: Dict[Tuple, int] = defaultdict(int)
        for i in selected:
            for tt in feasible_melds[i].required_tiles:
                type_usage[tt] += 1
        
        # Verify feasibility
        for tt in all_types:
            needed = type_usage.get(tt, 0)
            available = table_count.get(tt, 0) + hand_count.get(tt, 0)
            if needed > available:
                return None
        
        # How many from hand?
        hand_usage: Dict[Tuple, int] = {}
        for tt in all_types:
            hand_usage[tt] = max(0, type_usage.get(tt, 0) - table_count.get(tt, 0))
        
        total_hand_used = sum(hand_usage.values())
        if total_hand_used == 0:
            return None
        
        # ===== STEP 6: Build pool of available tile_ids =====
        # CRITICAL: Work with tile_ids, not tile objects, to prevent reference issues
        available_ids_by_type: Dict[Tuple, List[int]] = defaultdict(list)
        
        # Add all table tile_ids
        for tt, ids in table_ids_by_type.items():
            available_ids_by_type[tt].extend(ids)
        
        # Add required hand tile_ids
        played_tile_ids: List[int] = []
        for tt, count in hand_usage.items():
            if count > 0:
                hand_ids_for_type = hand_ids_by_type.get(tt, [])
                if count > len(hand_ids_for_type):
                    return None
                ids_to_use = hand_ids_for_type[:count]
                played_tile_ids.extend(ids_to_use)
                available_ids_by_type[tt].extend(ids_to_use)
        
        # ===== STEP 7: Assign tiles to melds using tile_ids =====
        new_sets: List[TileSet] = []
        globally_used_ids: Set[int] = set()  # Track ALL assigned tile_ids
        
        for meld_idx in selected:
            meld = feasible_melds[meld_idx]
            assigned_ids: List[int] = []
            
            for tile_type in meld.required_tiles:
                available_ids = available_ids_by_type.get(tile_type, [])
                
                # Find an unused tile_id
                found_id = None
                for tid in available_ids:
                    if tid not in globally_used_ids:
                        found_id = tid
                        break
                
                if found_id is None:
                    # No available tile - ILP solution is infeasible
                    return None
                
                globally_used_ids.add(found_id)
                assigned_ids.append(found_id)
            
            # Convert tile_ids to Tile objects
            set_tiles = [all_tiles_by_id[tid] for tid in assigned_ids]
            
            # Sort runs by number
            if meld.set_type == "run":
                set_tiles.sort(key=lambda t: t.number if t.number is not None else 0)
            
            # Create and validate TileSet
            new_set = TileSet(tiles=set_tiles, set_type=meld.set_type)
            if not new_set.is_valid():
                return None
            
            # Extra validation: no duplicate tile_ids in this set
            set_ids = [t.tile_id for t in set_tiles]
            if len(set_ids) != len(set(set_ids)):
                return None
            
            new_sets.append(new_set)
        
        # ===== STEP 8: Final validation =====
        all_result_ids = [t.tile_id for s in new_sets for t in s.tiles]
        if len(all_result_ids) != len(set(all_result_ids)):
            from collections import Counter
            counts = Counter(all_result_ids)
            dups = {k: v for k, v in counts.items() if v > 1}
            print(f"BUG IN SOLVER: Duplicate tile_ids in result: {dups}")
            return None
        
        # Check initial meld value
        if not has_melded:
            total_value = sum(s.get_meld_value() for s in new_sets)
            if total_value < 30:
                return None
        
        # ===== STEP 9: Build action =====
        # Get original hand tiles for played_tiles (for removal from hand)
        played_tiles = [all_tiles_by_id[tid] for tid in played_tile_ids]
        
        # CRITICAL: Create completely fresh TileSet objects with fresh Tile copies
        # to prevent any shared references
        final_sets: List[TileSet] = []
        for ns in new_sets:
            fresh_tiles = []
            for t in ns.tiles:
                # Create a brand new Tile object
                fresh_tile = Tile(
                    color=t.color,
                    number=t.number,
                    tile_type=t.tile_type,
                    tile_id=t.tile_id
                )
                fresh_tiles.append(fresh_tile)
            final_sets.append(TileSet(tiles=fresh_tiles, set_type=ns.set_type))
        
        # One more validation on final_sets
        final_ids = [t.tile_id for s in final_sets for t in s.tiles]
        if len(final_ids) != len(set(final_ids)):
            print("BUG: final_sets has duplicates!")
            return None
        
        if not has_melded:
            return RummikubAction(
                action_type='initial_meld',
                tiles=played_tiles,
                sets=final_sets
            )
        else:
            return RummikubAction(
                action_type='play',
                tiles=played_tiles,
                table_config=final_sets
            )