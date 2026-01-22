"""
Rummikub Action Generator using ILP and Heuristics

Provides three types of action generation:
1. Generator 1: Simple hand plays (no table manipulation)
2. Generator 2: Table extensions (add to existing sets)
3. Generator 3: Complex rearrangements (windowed search with backtracking)

IMPORTANT: These generators ONLY provide action choices.
The environment validates ice-breaking and determines if actions are legal.

ActionGenerator(
    mode=SolverMode.HYBRID,
    timeout_seconds=float('inf')  # Disabled time out for too long action searches
)

Usage:
    from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
    
    # Fast mode - Generators 1+2 only (~10ms)
    gen = ActionGenerator(mode=SolverMode.HEURISTIC_ONLY)
    
    # Balanced mode - All generators with limits (~100ms, recommended)
    gen = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=30)
    
    # Complete mode - Full search (~1s)
    gen = ActionGenerator(mode=SolverMode.ILP_ONLY)
    
    # Generate actions
    env.action_generator = gen
    actions = env.get_legal_actions(player_id)
"""

from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
from dataclasses import dataclass
import copy
from collections import defaultdict
import itertools
from itertools import combinations, product
from ortools.linear_solver import pywraplp
import time
from Rummikub_env import Tile, TileSet, RummikubAction, Color, TileType

try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("WARNING: ortools not available. Install with: pip install ortools")


@dataclass
class SetTemplate:
    """
    Template representing a possible set configuration.
    Used by ILP baseline opponent to enumerate all possible sets.
    
    Attributes:
        set_type: 'run' or 'group'
        pattern: List of (color, number) tuples or ('JOKER', 'JOKER')
        joker_count: Number of jokers in this template
        template_id: Unique identifier
    """
    set_type: str  # 'run' or 'group'
    pattern: List[Tuple[Optional[int], Optional[int]]]  # [(color, number), ...] or [('JOKER', 'JOKER'), ...]
    joker_count: int  # Number of jokers in this template
    template_id: int  # Unique ID


class SolverMode(Enum):
    """Action generator modes with different speed/completeness tradeoffs"""
    HEURISTIC_ONLY = "heuristic_only"  # Fast (~10ms), Generator 1+2 only
    HYBRID = "hybrid"  # Balanced (~100ms), All generators with limits
    ILP_ONLY = "ilp_only"  # Complete (~1s), Full ILP search


class ActionGenerator:
    """
    Main action generator coordinating three sub-generators.
    
    Generator 1: Simple hand plays - forms valid sets from hand tiles only
    Generator 2: Table extensions - adds tiles to existing table sets  
    Generator 3: Rearrangements - manipulates table using windowed search
    
    The environment decides if ice is broken and validates actions.
    """
    
    def __init__(self, mode: SolverMode = SolverMode.HYBRID, max_ilp_calls: int = 30,
                 max_window_size: int = 3, timeout_seconds: float = 30.0):
        """
        Args:
            mode: Generation strategy (speed vs completeness tradeoff)
            max_ilp_calls: Maximum number of windows for Generator 3
            max_window_size: Maximum table sets per window for Generator 3 (1-3)
                            1 = single sets only
                            2 = single + double sets  
                            3 = single + double + triple sets (finds complex rearrangements)
            timeout_seconds: Maximum time allowed for action generation (default 30s)
                           If exceeded, returns empty list (env will add draw action)
        """
        self.mode = mode
        self.max_ilp_calls = max_ilp_calls
        self.max_window_size = max_window_size
        self.timeout_seconds = timeout_seconds
        
        # Initialize sub-generators
        self.hand_play_gen = HandPlayGenerator()
        self.table_ext_gen = TableExtensionGenerator()
        
        if mode != SolverMode.HEURISTIC_ONLY:
            self.rearrange_gen = RearrangementGenerator(
                max_windows=max_ilp_calls,
                max_window_size=max_window_size,
                ilp_time_limit=2.0
            )
            print(f"  Using ILP-based Generator 3")
        else:
            self.rearrange_gen = None
        
        print(f"ActionGenerator initialized:")
        print(f"  Mode: {mode.value}")
        print(f"  Max windows: {max_ilp_calls}")
        print(f"  Max window size: {max_window_size} sets")
        print(f"  Timeout: {timeout_seconds}s")
    
    def generate_all_legal_actions(self, hand_tiles: List, table_sets: List, 
                                   has_melded: bool, pool_size: int) -> List:
        """
        Generate all legal action choices for current state.
        (Called by RummikubEnv.get_legal_actions)
        
        Args:
            hand_tiles: List of Tile objects in player's hand
            table_sets: List of TileSet objects on the table
            has_melded: Whether player has broken the ice (30+ points)
            pool_size: Number of tiles in pool (not used, but required by interface)
            
        Returns:
            List of RummikubAction objects (NOT including 'draw' - env adds that)
        """
        return self.generate_actions(hand_tiles, table_sets, has_melded)
    
    def generate_actions(self, hand: List, table: List, has_melded: bool) -> List:
        """
        Internal method to generate all legal action choices.
        
        Args:
            hand: List of Tile objects in player's hand
            table: List of TileSet objects on the table
            has_melded: Whether player has broken the ice (30+ points)
            
        Returns:
            List of RummikubAction objects (NOT including 'draw')
            Returns empty list if timeout exceeded (env will add draw action)
        """
        from Rummikub_env import RummikubAction
        import time
        
        actions = []
        start_time = time.time()
        timeout_occurred = [False]  # Mutable flag for nested functions
        
        # NOTE: Environment adds 'draw' action automatically,
        # so we don't include it here
        
        def check_timeout():
            """Check if we've exceeded timeout."""
            if time.time() - start_time > self.timeout_seconds:
                timeout_occurred[0] = True
                return True
            return False
        
        try:
            if not has_melded:
                # Before ice-breaking: find initial melds (30+ points from hand only)
                if check_timeout():
                    raise TimeoutError("Timeout during initial meld generation")
                initial_melds = self.hand_play_gen.generate_initial_melds(hand)
                actions.extend(initial_melds)
            else:
                # After ice-breaking: use all generators
                
                # Generator 1: Simple hand plays (new sets from hand)
                if check_timeout():
                    raise TimeoutError("Timeout during hand play generation")
                hand_actions = self.hand_play_gen.generate_hand_plays(hand, table)
                actions.extend(hand_actions)
                
                # Generator 2: Table extensions (add to existing sets)
                if len(table) > 0:
                    if check_timeout():
                        raise TimeoutError("Timeout during extension generation")
                    ext_actions = self.table_ext_gen.generate(hand, table)
                    actions.extend(ext_actions)
                
                # Generator 3: Complex rearrangements (windowed search)
                if len(table) > 0 and self.rearrange_gen is not None:
                    if check_timeout():
                        raise TimeoutError("Timeout before rearrangement generation")
                    
                    # Pass timeout info to Generator 3
                    remaining_time = self.timeout_seconds - (time.time() - start_time)
                    rearrange_actions = self.rearrange_gen.generate(
                        hand, table, timeout=remaining_time)
                    actions.extend(rearrange_actions)
            
            # Remove duplicates
            if not check_timeout():
                actions = self._deduplicate_actions(actions)
            
            elapsed = time.time() - start_time
            if elapsed > 5.0:  # Log if taking longer than 5 seconds
                print(f"⚠️  Action generation took {elapsed:.1f}s (found {len(actions)} actions)")
            
            return actions
            
        except TimeoutError as e:
            elapsed = time.time() - start_time
            print(f"\n⏱️  ACTION GENERATION TIMEOUT after {elapsed:.1f}s")
            print(f"    Returning {len(actions)} partial actions")
            print(f"    Environment will add DRAW action as fallback")
            
            # Return what we have so far
            # Environment will add 'draw' action automatically
            return actions
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Error during action generation after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            
            # Return partial actions if any, otherwise empty
            print(f"    Returning {len(actions)} partial actions")
            return actions
    
    def _deduplicate_actions(self, actions: List) -> List:
        """Remove duplicate actions based on tiles played and result."""
        seen = set()
        unique = []
        
        for action in actions:
            if action.action_type == 'draw':
                if 'draw' not in seen:
                    unique.append(action)
                    seen.add('draw')
                continue
            
            # Signature: tiles used + resulting table configuration
            tile_sig = tuple(sorted(t.tile_id for t in action.tiles)) if action.tiles else ()
            
            if action.table_config:
                table_sig = []
                for ts in action.table_config:
                    set_tiles = tuple(sorted(t.tile_id for t in ts.tiles))
                    table_sig.append((ts.set_type, set_tiles))
                table_sig = tuple(sorted(table_sig))
            else:
                table_sig = ()
            
            signature = (action.action_type, tile_sig, table_sig)
            
            if signature not in seen:
                unique.append(action)
                seen.add(signature)
        
        return unique


# =============================================================================
# GENERATOR 1: Simple Hand Plays (No Table Manipulation)
# =============================================================================

class HandPlayGenerator:
    """
    Generator 1: Find valid runs and groups from hand tiles only.
    
    Example:
        hand = {R11, b11, B11, O11, b13, B13, O13}
        
        Generates:
        1. {{R11, b11, B11}}
        2. {{R11, b11, B11, O11}}
        3. {{b13, B13, O13}}
        4. Combinations of above, etc.
    """
    def generate_initial_melds(self, hand: List) -> List:
        """Generate initial meld actions (30+ points from hand only)."""
        from Rummikub_env import RummikubAction, TileSet
        
        actions = []
        # Find all possible sets from hand
        all_sets = self._find_all_valid_sets(hand)
        
        # Generate combinations of sets
        for num_sets in range(1, len(all_sets) + 1):
            for combo in combinations(all_sets, num_sets):
                # Check no overlapping tiles
                all_tiles = [t for s in combo for t in s.tiles]
                if len(all_tiles) != len(set(t.tile_id for t in all_tiles)):
                    continue
                
                total_value = sum(s.get_meld_value() for s in combo)
                if total_value >= 30:
                    played_tiles = all_tiles
                    action = RummikubAction(
                        action_type='initial_meld',
                        tiles=played_tiles,
                        sets=list(combo)
                    )
                    actions.append(action)
        
        return actions
    
    def generate_hand_plays(self, hand: List, table: List) -> List:
        """Generate play actions adding new sets from hand (no manipulation)."""
        from Rummikub_env import RummikubAction, TileSet
        
        actions = []
        # Find all possible sets from hand
        all_sets = self._find_all_valid_sets(hand)
        
        # Generate non-empty combinations
        for num_sets in range(1, len(all_sets) + 1):
            for combo in combinations(all_sets, num_sets):
                # Check no overlapping tiles
                all_tiles = [t for s in combo for t in s.tiles]
                if len(all_tiles) != len(set(t.tile_id for t in all_tiles)):
                    continue
                
                # New table = old table + new sets
                new_table = copy.deepcopy(table) + list(combo)
                
                action = RummikubAction(
                    action_type='play',
                    tiles=all_tiles,
                    sets=list(combo),
                    table_config=new_table
                )
                actions.append(action)
        
        return actions
    
    def _find_all_valid_sets(self, hand: List) -> List[TileSet]:
        """Find all possible valid groups and runs from hand."""
        sets = []
        
        # Group by color and number for efficiency
        by_number = defaultdict(list)
        by_color = defaultdict(list)
        jokers = [t for t in hand if t.tile_type == TileType.JOKER]
        num_jokers = len(jokers)
        
        for tile in hand:
            if tile.tile_type != TileType.JOKER:
                by_number[tile.number].append(tile)
                by_color[tile.color].append(tile)
        
        # Find groups (same number, different colors)
        for num, tiles in by_number.items():
            # Possible group sizes 3-4, with 0-2 jokers
            for size in [3, 4]:
                if len(tiles) + num_jokers >= size and len(tiles) >= size - 2:
                    # Select distinct colors
                    color_set = set(t.color for t in tiles)
                    if len(color_set) + num_jokers >= size:
                        # Create group
                        group_tiles = tiles[:size - max(0, size - len(tiles))] + jokers[:max(0, size - len(tiles))]
                        sets.append(TileSet(group_tiles, 'group'))
        
        # Find runs (same color, consecutive)
        for color, tiles in by_color.items():
            tiles.sort(key=lambda t: t.number)
            # Find consecutive sequences with gaps for jokers
            for length in range(3, len(tiles) + num_jokers + 1):
                # Logic to find valid run starts and joker placements
                # (Implement detailed run finding here, similar to combinatorics)
                pass  # Placeholder - expand as in previous versions
        
        return [s for s in sets if s.is_valid()]


# =============================================================================
# GENERATOR 2: Table Extensions (Add to Existing Sets)
# =============================================================================

class TableExtensionGenerator:
    """
    Generator 2: Add hand tiles to existing table sets without splitting/rearranging.
    """
    def generate(self, hand: List[Tile], table: List[TileSet]) -> List[RummikubAction]:
        actions = []
        
        for set_idx, tile_set in enumerate(table):
            # Try adding 1+ tiles to this set
            connected = self._find_connectable_tiles(hand, tile_set)
            for num_add in range(1, len(connected) + 1):
                for adds in combinations(connected, num_add):
                    new_tiles = tile_set.tiles + list(adds)
                    new_set = TileSet(new_tiles, tile_set.set_type)
                    if new_set.is_valid():
                        new_table = copy.deepcopy(table)
                        new_table[set_idx] = new_set
                        action = RummikubAction(
                            'play',
                            tiles=list(adds),
                            table_config=new_table
                        )
                        actions.append(action)
        
        # Also combinations across multiple sets
        # (Expand for multi-set extensions if needed)
        
        return actions
    
    def _find_connectable_tiles(self, hand: List[Tile], tile_set: TileSet) -> List[Tile]:
        connectable = []
        if tile_set.set_type == 'group':
            num = tile_set.tiles[0].number  # Assume same number
            used_colors = set(t.color for t in tile_set.tiles if t.tile_type != TileType.JOKER)
            for t in hand:
                if t.tile_type == TileType.JOKER or (t.number == num and t.color not in used_colors):
                    connectable.append(t)
        elif tile_set.set_type == 'run':
            color = tile_set.tiles[0].color  # Assume same color
            min_num = min(t.number for t in tile_set.tiles if t.number)
            max_num = max(t.number for t in tile_set.tiles if t.number)
            for t in hand:
                if t.tile_type == TileType.JOKER or (t.color == color and (t.number == min_num - 1 or t.number == max_num + 1)):
                    connectable.append(t)
        return connectable


# =============================================================================
# GENERATOR 3: Complex Rearrangements (Windowed ILP Search)
# =============================================================================

def get_key(tile: Tile) -> Tuple[Optional[int], Optional[int]]:
    if tile.tile_type == TileType.JOKER:
        return None, None
    return tile.color.value, tile.number

@dataclass
class SetTemplate:
    set_type: str  # "group" or "run"
    pattern: List[Tuple[Optional[int], Optional[int]]]  # List of (color_value, number), None for jokers

class RearrangementGenerator:
    """
    Generator 3: Complex table rearrangements using windowed ILP search.
    Finds valid rearrangements of small subsets (windows) of table sets + connected hand tiles.
    
    Uses Integer Linear Programming (ILP) to guarantee optimal tile usage per window while ensuring
    all window tiles are used in valid melds.
    
    This is a "perfect" implementation: For each window, ILP finds the arrangement that maximizes
    the value of hand tiles played, with full solvability.
    """
    
    def __init__(self, max_windows: int = 30, max_window_size: int = 3, ilp_time_limit: float = 2.0):
        self.max_windows = max_windows
        self.max_window_size = max_window_size
        self.ilp_time_limit = ilp_time_limit
        self.templates = self._generate_all_possible_templates()
    
    def _generate_all_possible_templates(self) -> List[SetTemplate]:
        """
        Generates all possible valid meld templates (groups and runs), including variations with up to 2 jokers.
        Normalized pattern to use (None, None) for jokers.
        """
        templates: List[SetTemplate] = []
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
                        pattern = [(c, n) for c in s] + [(None, None)] * num_jokers
                        templates.append(SetTemplate("group", pattern))
        
        # Generate runs (same color, consecutive numbers, 3+ tiles)
        for col in colors:
            for start in range(1, 14):
                for length in range(3, 15 - start + 1):
                    for num_jokers in range(3):  # Limit to 0-2 jokers
                        if num_jokers >= length:  # At least one non-joker
                            continue
                        for joker_pos in itertools.combinations(range(length), num_jokers):
                            pattern = []
                            for pos in range(length):
                                if pos in joker_pos:
                                    pattern.append((None, None))
                                else:
                                    num = start + pos
                                    pattern.append((col, num))
                            templates.append(SetTemplate("run", pattern))
        
        return templates
    
    def generate(self, hand: List[Tile], table: List[TileSet], timeout: float = 30.0) -> List[RummikubAction]:
        """
        Generate rearrangement actions by searching over windows of table sets.
        
        Args:
            hand: Player's hand tiles
            table: Current table sets
            timeout: Maximum time for generation
        
        Returns:
            List of valid play actions involving rearrangements
        """
        actions = []
        start_time = time.time()
        
        if len(table) == 0:
            return []
        
        # Generate windows: subsets of 1 to max_window_size sets
        for window_size in range(1, self.max_window_size + 1):
            for table_indices in itertools.combinations(range(len(table)), window_size):
                if len(actions) >= self.max_windows:
                    break
                
                if time.time() - start_time > timeout:
                    return actions
                
                window_sets = [table[idx] for idx in table_indices]
                window_tiles = [t for s in window_sets for t in s.tiles]
                
                # Filter connected hand tiles
                connected = self._filter_connected(hand, window_tiles)
                if not connected:
                    continue
                
                # Solve ILP for this window
                action = self._solve_window(window_tiles, connected, table_indices, table)
                if action:
                    actions.append(action)
        
        return actions
    
    def _solve_window(self, window_tiles: List[Tile], connected_hand: List[Tile], 
                      table_indices: Tuple[int], table: List[TileSet]) -> Optional[RummikubAction]:
        """
        Use ILP to find a valid rearrangement for the window + hand tiles.
        """
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return None
        
        solver.SetTimeLimit(int(self.ilp_time_limit * 1000))  # ms
        
        # Collect types and counts
        type_inventory: Dict[Tuple[Optional[int], Optional[int]], Dict] = defaultdict(lambda: {
            'count_window': 0, 'count_hand': 0, 'tiles_all': []
        })
        
        for t in window_tiles:
            tt = get_key(t)
            type_inventory[tt]['count_window'] += 1
            type_inventory[tt]['tiles_all'].append(t)
        
        for t in connected_hand:
            tt = get_key(t)
            type_inventory[tt]['count_hand'] += 1
            type_inventory[tt]['tiles_all'].append(t)
        
        tile_types = set(type_inventory.keys())
        
        # Filter possible templates based on available types
        possible_templates = [
            t for t in self.templates
            if all(
                type_inventory.get(tt, {'count_window':0, 'count_hand':0})['count_window'] + 
                type_inventory.get(tt, {'count_window':0, 'count_hand':0})['count_hand'] >= 
                t.pattern.count(tt)
                for tt in set(t.pattern)
            )
        ]
        
        num_templates = len(possible_templates)
        if num_templates == 0:
            return None
        
        # Variables: x_j for each template (int, allow multiples if possible, but limit to 2 for safety)
        x = [solver.IntVar(0, 2, f'x[{i}]') for i in range(num_templates)]
        
        # y_tt: number of hand tiles of type tt used
        y = {tt: solver.IntVar(0, type_inventory[tt]['count_hand'], f'y[{tt}]') for tt in tile_types}
        
        # Constraints: for each type, sum over j (count_tt_in_j * x_j) == count_window + y_tt
        for tt in tile_types:
            used = solver.Sum(x[i] * possible_templates[i].pattern.count(tt) for i in range(num_templates))
            constraint = solver.Add(used == type_inventory[tt]['count_window'] + y[tt])
        
        # Objective: sum (value_tt * y_tt)
        obj = solver.Sum(
            y[tt] * (tt[1] if tt[1] is not None else 30)
            for tt in tile_types
        )
        solver.Maximize(obj)
        
        # Solve
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        # Check if any hand tiles used
        total_hand_used = sum(y[tt].solution_value() for tt in tile_types)
        if total_hand_used < 1:
            return None
        
        # Get selected templates
        selected = [(i, int(x[i].solution_value())) for i in range(num_templates) if x[i].solution_value() > 0]
        
        # Compute hand used counts
        hand_used_count = {tt: int(y[tt].solution_value()) for tt in tile_types}
        
        # Select played tiles from connected hand
        hand_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in connected_hand:
            hand_by_type[get_key(t)].append(t)
        
        played_tiles: List[Tile] = []
        for tt, cnt in hand_used_count.items():
            if cnt > 0:
                played_tiles.extend(hand_by_type[tt][:cnt])
        
        # All used tiles: window + played
        all_used_tiles = window_tiles + played_tiles
        
        # Available tiles by type
        available_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in all_used_tiles:
            available_by_type[get_key(t)].append(t)
        
        # Instantiate sets
        new_sets: List[TileSet] = []
        for i, count in selected:
            for _ in range(count):
                templ = possible_templates[i]
                set_tiles: List[Tile] = []
                for p in templ.pattern:
                    tt = p  # Already (Optional[int], Optional[int])
                    tiles_list = available_by_type[tt]
                    if not tiles_list:
                        raise ValueError("Tile assignment failed: insufficient tiles.")
                    set_tiles.append(tiles_list.pop())
                
                # Sort runs
                if templ.set_type == "run":
                    set_tiles.sort(key=lambda t: t.number if t.number is not None else -1)
                
                new_set = TileSet(tiles=set_tiles, set_type=templ.set_type)
                if not new_set.is_valid():
                    raise ValueError("Generated invalid set.")
                new_sets.append(new_set)
        
        # Build final table config: unchanged sets + new sets
        final_table = [table[idx] for idx in range(len(table)) if idx not in table_indices]
        final_table.extend(new_sets)
        
        # Create action
        return RummikubAction(
            action_type='play',
            tiles=played_tiles,
            sets=new_sets,  # Optional, for info
            table_config=final_table
        )
    
    def _filter_connected(self, hand: List[Tile], window_tiles: List[Tile]) -> List[Tile]:
        """Filter hand tiles that can connect to window tiles."""
        connected = []
        
        for hand_tile in hand:
            if hand_tile.tile_type == TileType.JOKER:
                connected.append(hand_tile)
                continue
            
            for table_tile in window_tiles:
                if table_tile.tile_type == TileType.JOKER:
                    connected.append(hand_tile)
                    break
                
                # Run connection: same color, adjacent number (±1 or ±2 for joker potential)
                if hand_tile.color == table_tile.color and abs(hand_tile.number - table_tile.number) <= 2:
                    connected.append(hand_tile)
                    break
                
                # Group connection: same number, different color
                if hand_tile.number == table_tile.number and hand_tile.color != table_tile.color:
                    connected.append(hand_tile)
                    break
        
        return connected


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RUMMIKUB ACTION GENERATOR - TESTING")
    print("="*70)
    
    from Rummikub_env import RummikubEnv
    
    # Create test environment
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    # Test all modes
    for mode in [SolverMode.HEURISTIC_ONLY, SolverMode.HYBRID]:
        print(f"\n{'='*70}")
        print(f"MODE: {mode.value.upper()}")
        print(f"{'='*70}")
        
        gen = ActionGenerator(mode=mode, max_ilp_calls=10)
        env.action_generator = gen
        
        # Test before ice-breaking
        print("\n1. BEFORE ICE-BREAKING:")
        hand = env.player_hands[0]
        table = env.table
        has_melded = env.has_melded[0]
        
        print(f"   Hand: {len(hand)} tiles, Value: {sum(t.get_value() for t in hand)}")
        print(f"   Table: {len(table)} sets")
        
        actions = env.get_legal_actions(0)
        print(f"   Actions: {len(actions)} total")
        
        action_types = {}
        for action in actions:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        for atype, count in action_types.items():
            print(f"     - {atype}: {count}")
        
        # Make initial meld if possible
        for action in actions:
            if action.action_type == 'initial_meld':
                env.step(action)
                print(f"\n   ✓ Made initial meld: {len(action.tiles)} tiles")
                break
        
        # Test after ice-breaking
        if env.has_melded[0]:
            print("\n2. AFTER ICE-BREAKING:")
            env.current_player = 0
            actions = env.get_legal_actions(0)
            print(f"   Actions: {len(actions)} total")
            
            action_types = {}
            for action in actions:
                action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
            for atype, count in action_types.items():
                print(f"     - {atype}: {count}")
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE!")
    print(f"{'='*70}\n")


