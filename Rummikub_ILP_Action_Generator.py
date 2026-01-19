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

import numpy as np
from typing import List, Set, Tuple, Optional, Dict
from itertools import combinations, product
from enum import Enum
from dataclasses import dataclass
import copy

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
    pattern: List[Tuple]  # [(color, number), ...] or [('JOKER', 'JOKER'), ...]
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
                max_melds_per_window=10,
                max_window_size=max_window_size
            )
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
        4. {{R11, b11, B11}, {b13, B13, O13}}
        5. {{R11, b11, B11, O11}, {b13, B13, O13}}
        ... etc
    """
    
    def generate_initial_melds(self, hand: List) -> List:
        """
        Generate initial melds (30+ points, from hand only).
        Used before ice-breaking.
        """
        from Rummikub_env import RummikubAction
        
        actions = []
        
        # Try all subsets of hand (largest first for efficiency)
        for size in range(len(hand), 2, -1):
            for tile_combo in combinations(hand, size):
                tiles = list(tile_combo)
                
                # Find all valid partitions
                partitions = self._find_valid_partitions(tiles)
                
                for partition in partitions:
                    # Check if meld value >= 30
                    total_value = sum(s.get_meld_value() for s in partition)
                    
                    if total_value >= 30:
                        tiles_used = []
                        for ts in partition:
                            tiles_used.extend(ts.tiles)
                        
                        action = RummikubAction(
                            action_type='initial_meld',
                            tiles=tiles_used,
                            sets=partition,
                            table_config=partition  # New sets become table
                        )
                        actions.append(action)
        
        return actions
    
    def generate_hand_plays(self, hand: List, table: List) -> List:
        """
        Generate play actions from hand only (after ice-breaking).
        """
        from Rummikub_env import RummikubAction
        
        if len(hand) < 3:
            return []
        
        actions = []
        
        # Try all subsets of hand (size 3+)
        for size in range(3, len(hand) + 1):
            for tile_combo in combinations(hand, size):
                tiles = list(tile_combo)
                
                # Find all valid partitions
                partitions = self._find_valid_partitions(tiles)
                
                for partition in partitions:
                    tiles_used = []
                    for ts in partition:
                        tiles_used.extend(ts.tiles)
                    
                    # New table = old table + new sets
                    new_table = copy.deepcopy(table)
                    new_table.extend(partition)
                    
                    action = RummikubAction(
                        action_type='play',
                        tiles=tiles_used,
                        sets=partition,
                        table_config=new_table
                    )
                    actions.append(action)
        
        return actions
    
    def _find_valid_partitions(self, tiles: List) -> List[List]:
        """
        Find all valid ways to partition tiles into runs and groups.
        Uses backtracking search.
        """
        from Rummikub_env import TileSet
        
        if len(tiles) < 3:
            return []
        
        partitions = []
        
        def backtrack(remaining: List, current_partition: List):
            if len(remaining) == 0:
                # Found valid complete partition
                if len(current_partition) > 0:
                    partitions.append(copy.deepcopy(current_partition))
                return
            
            if len(remaining) < 3:
                # Can't form more sets
                return
            
            # Try all possible sets from remaining tiles
            for size in range(3, min(len(remaining) + 1, 14)):  # Max 13 for runs
                for combo in combinations(remaining, size):
                    tile_list = list(combo)
                    
                    # Try as run
                    test_run = TileSet(tiles=tile_list, set_type='run')
                    if test_run.is_valid():
                        new_remaining = [t for t in remaining if t not in combo]
                        current_partition.append(test_run)
                        backtrack(new_remaining, current_partition)
                        current_partition.pop()
                    
                    # Try as group (max 4 tiles)
                    if size <= 4:
                        test_group = TileSet(tiles=tile_list, set_type='group')
                        if test_group.is_valid():
                            new_remaining = [t for t in remaining if t not in combo]
                            current_partition.append(test_group)
                            backtrack(new_remaining, current_partition)
                            current_partition.pop()
        
        backtrack(tiles, [])
        return partitions


# =============================================================================
# GENERATOR 2: Table Extensions (Add to Existing Sets)
# =============================================================================

class TableExtensionGenerator:
    """
    Generator 2: Add tiles from hand to existing table sets.
    
    Example:
        hand = {R1, R6, R8, R8, R11, R12, b1, b3, b9, b10, b11, b13, ...}
        table = {{R1, B1, O1}, {R9, R10, R11}}
        
        Generates:
        1. Play b1 to {R1, B1, O1} => {b1, R1, B1, O1}
        2. Play R8 to {R9, R10, R11} => {R8, R9, R10, R11}
        3. Play R12 to {R9, R10, R11} => {R9, R10, R11, R12}
        4. Combo: b1 to set1 AND R8 to set2
        5. Combo: b1 to set1 AND R12 to set2
        6. Combo: b1 to set1 AND R8, R12 to set2 => {R8, R9, R10, R11, R12}
        ... etc
    """
    
    def generate(self, hand: List, table: List) -> List:
        """Generate all table extension actions."""
        from Rummikub_env import RummikubAction
        
        if len(table) == 0 or len(hand) == 0:
            return []
        
        actions = []
        
        # Find all possible extensions for each table set
        extensions_per_set = []
        
        for set_idx, table_set in enumerate(table):
            extensions = self._find_extensions(table_set, hand)
            
            if len(extensions) > 0:
                extensions_per_set.append((set_idx, extensions))
        
        if len(extensions_per_set) == 0:
            return []
        
        # Generate all combinations of extensions
        combinations_list = self._generate_combos(extensions_per_set)
        
        for combo in combinations_list:
            # combo is list of (set_idx, tiles_to_add, new_set)
            tiles_used = []
            new_table = copy.deepcopy(table)
            
            for set_idx, tiles_to_add, new_set in combo:
                tiles_used.extend(tiles_to_add)
                new_table[set_idx] = new_set
            
            action = RummikubAction(
                action_type='play',
                tiles=tiles_used,
                sets=None,  # Extensions modify existing sets
                table_config=new_table
            )
            actions.append(action)
        
        return actions
    
    def _find_extensions(self, table_set, hand: List) -> List[Tuple]:
        """Find all ways to extend a single table set."""
        if table_set.set_type == 'run':
            return self._extend_run(table_set, hand)
        elif table_set.set_type == 'group':
            return self._extend_group(table_set, hand)
        return []
    
    def _extend_run(self, run, hand: List) -> List[Tuple]:
        """Find ways to extend a run by adding consecutive tiles."""
        from Rummikub_env import TileSet, TileType
        
        extensions = []
        
        # Get run color and number range
        non_jokers = [t for t in run.tiles if t.tile_type != TileType.JOKER]
        if len(non_jokers) == 0:
            return []
        
        run_color = non_jokers[0].color
        numbers = sorted([t.number for t in non_jokers])
        min_num = numbers[0]
        max_num = numbers[-1]
        
        # Find matching tiles in hand
        matching = [t for t in hand 
                   if t.tile_type != TileType.JOKER and t.color == run_color]
        
        # Single tile extensions
        for tile in matching:
            # Add to beginning
            if tile.number == min_num - 1 and tile.number >= 1:
                new_tiles = [tile] + run.tiles
                new_set = TileSet(tiles=new_tiles, set_type='run')
                if new_set.is_valid():
                    extensions.append(([tile], new_set))
            
            # Add to end
            if tile.number == max_num + 1 and tile.number <= 13:
                new_tiles = run.tiles + [tile]
                new_set = TileSet(tiles=new_tiles, set_type='run')
                if new_set.is_valid():
                    extensions.append(([tile], new_set))
        
        # Two tile extensions (both ends)
        for tile1 in matching:
            for tile2 in matching:
                if tile1.tile_id != tile2.tile_id:
                    if (tile1.number == min_num - 1 and 
                        tile2.number == max_num + 1 and
                        tile1.number >= 1 and tile2.number <= 13):
                        new_tiles = [tile1] + run.tiles + [tile2]
                        new_set = TileSet(tiles=new_tiles, set_type='run')
                        if new_set.is_valid():
                            extensions.append(([tile1, tile2], new_set))
        
        # Multiple consecutive tiles at one end
        for num_tiles in range(2, 4):  # Try 2-3 tiles
            for combo in combinations(matching, num_tiles):
                tiles = list(combo)
                tile_numbers = [t.number for t in tiles]
                
                # Check if consecutive
                tile_numbers.sort()
                is_consecutive = all(
                    tile_numbers[i+1] - tile_numbers[i] == 1 
                    for i in range(len(tile_numbers) - 1)
                )
                
                if not is_consecutive:
                    continue
                
                # Try adding to beginning
                if tile_numbers[-1] == min_num - 1:
                    tiles_sorted = sorted(tiles, key=lambda t: t.number)
                    new_tiles = tiles_sorted + run.tiles
                    new_set = TileSet(tiles=new_tiles, set_type='run')
                    if new_set.is_valid():
                        extensions.append((tiles, new_set))
                
                # Try adding to end
                if tile_numbers[0] == max_num + 1:
                    tiles_sorted = sorted(tiles, key=lambda t: t.number)
                    new_tiles = run.tiles + tiles_sorted
                    new_set = TileSet(tiles=new_tiles, set_type='run')
                    if new_set.is_valid():
                        extensions.append((tiles, new_set))
        
        return extensions
    
    def _extend_group(self, group, hand: List) -> List[Tuple]:
        """Find ways to extend a group by adding same number, different color."""
        from Rummikub_env import TileSet, TileType
        
        extensions = []
        
        # Get group number and used colors
        non_jokers = [t for t in group.tiles if t.tile_type != TileType.JOKER]
        if len(non_jokers) == 0:
            return []
        
        group_number = non_jokers[0].number
        used_colors = set(t.color for t in non_jokers)
        
        # Group already max size (4 tiles)
        if len(group.tiles) >= 4:
            return []
        
        # Find matching tiles in hand (same number, different color)
        matching = [t for t in hand 
                   if t.tile_type != TileType.JOKER 
                   and t.number == group_number 
                   and t.color not in used_colors]
        
        for tile in matching:
            new_tiles = group.tiles + [tile]
            new_set = TileSet(tiles=new_tiles, set_type='group')
            if new_set.is_valid():
                extensions.append(([tile], new_set))
        
        return extensions
    
    def _generate_combos(self, extensions_per_set: List) -> List:
        """Generate all valid combinations of extensions."""
        all_combos = []
        
        # Single extensions
        for set_idx, extensions in extensions_per_set:
            for tiles, new_set in extensions:
                all_combos.append([(set_idx, tiles, new_set)])
        
        # Multiple extensions (ensure no tile overlap)
        if len(extensions_per_set) >= 2:
            for size in range(2, len(extensions_per_set) + 1):
                for combo_indices in combinations(range(len(extensions_per_set)), size):
                    # Get extension options for selected sets
                    options = []
                    for idx in combo_indices:
                        set_idx, extensions = extensions_per_set[idx]
                        options.append([(set_idx, tiles, new_set) 
                                       for tiles, new_set in extensions])
                    
                    # Generate all combinations of extensions
                    for ext_combo in product(*options):
                        # Check for tile overlaps
                        all_tile_ids = []
                        for _, tiles, _ in ext_combo:
                            all_tile_ids.extend([t.tile_id for t in tiles])
                        
                        # Valid if no duplicates
                        if len(all_tile_ids) == len(set(all_tile_ids)):
                            all_combos.append(list(ext_combo))
        
        return all_combos


# =============================================================================
# GENERATOR 3: Complex Rearrangements (Windowed Search)
# =============================================================================

class RearrangementGenerator:
    """
    Generator 3: Approximate full table rearrangement using windowed search.
    
    Strategy:
    1. Select windows: Pick 1-2 table melds + limited hand subset
    2. Filter tiles: Keep only tiles that "connect" to selected table sets
    3. Backtracking search: Re-partition window tiles into valid melds
    4. Limit: Top 10 melds per window for performance
    
    This approximates the true rearrangement problem while maintaining
    reasonable performance.
    """
    
    def __init__(self, max_windows: int = 30, max_melds_per_window: int = 10,
                 max_window_size: int = 3):
        """
        Args:
            max_windows: Maximum number of windows to explore
            max_melds_per_window: Maximum melds to keep per window
            max_window_size: Maximum number of table sets per window (1-3)
                            1 = single sets only
                            2 = single + pair sets (default in v1)
                            3 = single + pair + triple sets (for complex rearrangements)
        """
        self.max_windows = max_windows
        self.max_melds_per_window = max_melds_per_window
        self.max_window_size = max_window_size
    
    def generate(self, hand: List, table: List, timeout: float = None) -> List:
        """
        Generate rearrangement actions using windowed search (1-3 set windows).
        
        Args:
            hand: Hand tiles
            table: Table sets
            timeout: Maximum time remaining (seconds). If exceeded, stops early.
        """
        from Rummikub_env import RummikubAction
        import time
        
        if len(table) == 0:
            return []
        
        start_time = time.time()
        actions = []
        windows_explored = 0
        
        def check_timeout():
            if timeout is not None and (time.time() - start_time) > timeout:
                return True
            return False
        
        # Try single-set windows
        if self.max_window_size >= 1:
            for set_idx in range(len(table)):
                if windows_explored >= self.max_windows or check_timeout():
                    break
                
                window_actions = self._explore_window([set_idx], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        # Try two-set windows
        if self.max_window_size >= 2 and windows_explored < self.max_windows and len(table) >= 2:
            for idx1, idx2 in combinations(range(len(table)), 2):
                if windows_explored >= self.max_windows or check_timeout():
                    break
                
                window_actions = self._explore_window([idx1, idx2], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        # Try three-set windows for complex rearrangements
        if self.max_window_size >= 3 and windows_explored < self.max_windows and len(table) >= 3:
            for idx1, idx2, idx3 in combinations(range(len(table)), 3):
                if windows_explored >= self.max_windows or check_timeout():
                    if check_timeout():
                        elapsed = time.time() - start_time
                        print(f"    Generator 3 stopped after {elapsed:.1f}s ({windows_explored} windows)")
                    break
                
                window_actions = self._explore_window([idx1, idx2, idx3], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        return actions
    
    def _explore_window(self, table_indices: List[int], hand: List, 
                       table: List) -> List:
        """Explore a single window and generate actions."""
        from Rummikub_env import RummikubAction, TileType
        
        actions = []
        
        # Get tiles from selected table sets
        table_tiles = []
        for idx in table_indices:
            table_tiles.extend(table[idx].tiles)
        
        # Filter hand tiles that "connect" to table tiles
        connected = self._filter_connected(hand, table_tiles)
        
        # Limit hand subset to keep search tractable
        if len(connected) > 6:
            # Prioritize: non-jokers, high value
            connected.sort(key=lambda t: (
                t.tile_type == TileType.JOKER,
                -t.get_value()
            ))
            connected = connected[:6]
        
        if len(connected) == 0:
            return []
        
        # Pool: table tiles + connected hand tiles
        pool = table_tiles + connected
        
        # Find all valid re-partitions using backtracking
        partitions = self._backtrack_search(pool)
        
        # Keep top N partitions by value played from hand
        if len(partitions) > self.max_melds_per_window:
            def score_partition(partition):
                return sum(
                    sum(t.get_value() for t in ts.tiles if t in connected)
                    for ts in partition
                )
            partitions.sort(key=score_partition, reverse=True)
            partitions = partitions[:self.max_melds_per_window]
        
        # Convert partitions to actions
        for partition in partitions:
            # Must use at least one hand tile
            tiles_from_hand = [
                t for ts in partition for t in ts.tiles 
                if t in connected
            ]
            
            if len(tiles_from_hand) == 0:
                continue
            
            # Build new table: unchanged sets + new partition
            new_table = []
            for idx, ts in enumerate(table):
                if idx not in table_indices:
                    new_table.append(ts)
            new_table.extend(partition)
            
            action = RummikubAction(
                action_type='play',
                tiles=tiles_from_hand,
                sets=partition,
                table_config=new_table
            )
            actions.append(action)
        
        return actions
    
    def _filter_connected(self, hand: List, table_tiles: List) -> List:
        """
        Filter hand tiles that 'connect' to table tiles.
        
        A tile connects if:
        - Same number (group potential)
        - Same color + adjacent number (run potential)
        - Is a joker (universal)
        """
        from Rummikub_env import TileType
        
        connected = []
        
        # Extract table properties
        table_numbers = set()
        table_colors = set()
        
        for tile in table_tiles:
            if tile.tile_type != TileType.JOKER:
                table_numbers.add(tile.number)
                table_colors.add(tile.color)
        
        # Check each hand tile
        for tile in hand:
            # Jokers connect to everything
            if tile.tile_type == TileType.JOKER:
                connected.append(tile)
                continue
            
            # Same number => group potential
            if tile.number in table_numbers:
                connected.append(tile)
                continue
            
            # Same color + nearby number => run potential
            if tile.color in table_colors:
                for num in table_numbers:
                    if abs(tile.number - num) <= 2:
                        connected.append(tile)
                        break
        
        return connected
    
    def _backtrack_search(self, pool: List) -> List[List]:
        """Find valid partitions of pool using backtracking."""
        from Rummikub_env import TileSet
        
        if len(pool) < 3:
            return []
        
        partitions = []
        max_partitions = self.max_melds_per_window * 2  # Find more, then filter
        
        def backtrack(remaining: List, current: List):
            if len(partitions) >= max_partitions:
                return
            
            if len(remaining) == 0:
                if len(current) > 0:
                    partitions.append(copy.deepcopy(current))
                return
            
            if len(remaining) < 3:
                return
            
            # Try forming sets greedily
            for size in range(3, min(len(remaining) + 1, 14)):
                for combo in combinations(remaining, size):
                    tiles = list(combo)
                    
                    # Try as run
                    test_run = TileSet(tiles=tiles, set_type='run')
                    if test_run.is_valid():
                        new_remaining = [t for t in remaining if t not in combo]
                        current.append(test_run)
                        backtrack(new_remaining, current)
                        current.pop()
                        
                        if len(partitions) >= max_partitions:
                            return
                    
                    # Try as group (max 4)
                    if size <= 4:
                        test_group = TileSet(tiles=tiles, set_type='group')
                        if test_group.is_valid():
                            new_remaining = [t for t in remaining if t not in combo]
                            current.append(test_group)
                            backtrack(new_remaining, current)
                            current.pop()
                            
                            if len(partitions) >= max_partitions:
                                return
        
        backtrack(pool, [])
        return partitions
    # =============================================================================
    # NEW METHODS TO ADD TO RearrangementGenerator CLASS
    # =============================================================================

    def _has_joker_on_table(self, table_tiles: List) -> bool:
        """Check if any table tiles in the window contain a joker."""
        from Rummikub_env import TileType
        
        return any(t.tile_type == TileType.JOKER for t in table_tiles)


    def _has_joker_retrieval(self, table_tiles: List, partition: List) -> Tuple[bool, Optional[object]]:
        """
        Check if a partition retrieves a joker from the table.
        
        Returns:
            (retrieval_occurred, retrieved_joker_object or None)
        """
        from Rummikub_env import TileType
        
        # Find jokers on table
        table_jokers = [t for t in table_tiles if t.tile_type == TileType.JOKER]
        
        if not table_jokers:
            return False, None
        
        # Check if joker appears in partition
        partition_tiles = [t for ts in partition for t in ts.tiles]
        partition_jokers = [t for t in partition_tiles if t.tile_type == TileType.JOKER]
        
        # If table had joker but partition doesn't, joker was retrieved
        if len(table_jokers) > len(partition_jokers):
            # Find which joker was retrieved (might be multiple)
            retrieved = [tj for tj in table_jokers if tj not in partition_jokers]
            return True, retrieved[0] if retrieved else None
        
        return False, None


    def _find_joker_compatible_tiles(self, hand: List, already_connected: List) -> List:
        """
        Find hand tiles that could form sets WITH a joker.
        
        These are tiles that AREN'T connected to the table tiles,
        but COULD work with a retrieved joker.
        
        Strategy: Include any 2+ tiles that can form a set with a joker:
        - Runs: Two consecutive tiles (joker fills gap)
        - Groups: Two same-number tiles (joker adds third color)
        """
        from Rummikub_env import TileType
        
        additional = []
        already_ids = set(t.tile_id for t in already_connected)
        
        # Get remaining hand tiles
        remaining = [t for t in hand if t.tile_id not in already_ids 
                    and t.tile_type != TileType.JOKER]
        
        # Check all pairs of remaining tiles
        from itertools import combinations
        for t1, t2 in combinations(remaining, 2):
            # Run potential: Same color, numbers differ by 1 or 2
            if (t1.color == t2.color and 
                1 <= abs(t1.number - t2.number) <= 2):
                if t1 not in additional:
                    additional.append(t1)
                if t2 not in additional:
                    additional.append(t2)
            
            # Group potential: Same number, different colors
            if (t1.number == t2.number and 
                t1.color != t2.color):
                if t1 not in additional:
                    additional.append(t1)
                if t2 not in additional:
                    additional.append(t2)
        
        return additional


    def _validate_joker_usage(self, table_tiles: List, partition: List, 
                            hand_tiles: List) -> bool:
        """
        Validate that if a joker is retrieved, it's used in the same turn.
        
        Args:
            table_tiles: Original tiles in the window
            partition: Proposed new partition
            hand_tiles: Tiles from hand that are used
        
        Returns:
            True if valid (no joker retrieval, or joker is used)
        """
        has_retrieval, retrieved_joker = self._has_joker_retrieval(table_tiles, partition)
        
        if not has_retrieval:
            return True  # No joker retrieval, always valid
        
        # Joker was retrieved - must be used in partition
        partition_tiles = [t for ts in partition for t in ts.tiles]
        
        # Check if the retrieved joker is in the partition
        if retrieved_joker in partition_tiles:
            return True  # Joker is used ✓
        
        # Joker not in partition - INVALID
        return False


    def _explore_window_with_joker_awareness(self, table_indices: List[int], 
                                            hand: List, table: List) -> List:
        """
        Enhanced window exploration with joker retrieval awareness.
        
        This replaces the original _explore_window when joker handling is needed.
        """
        from Rummikub_env import RummikubAction, TileType
        
        actions = []
        
        # Get tiles from selected table sets
        table_tiles = []
        for idx in table_indices:
            table_tiles.extend(table[idx].tiles)
        
        # Filter hand tiles that "connect" to table tiles (original logic)
        connected = self._filter_connected(hand, table_tiles)
        
        # NEW: If window has joker, expand search space
        if self._has_joker_on_table(table_tiles):
            # Add tiles that could work WITH the joker if retrieved
            joker_compatible = self._find_joker_compatible_tiles(hand, connected)
            connected.extend(joker_compatible)
            
            # Also ensure any jokers in hand are included
            hand_jokers = [t for t in hand if t.tile_type == TileType.JOKER 
                        and t not in connected]
            connected.extend(hand_jokers)
        
        # Limit hand subset to keep search tractable
        if len(connected) > 8:  # Increased from 6 due to joker scenarios
            connected.sort(key=lambda t: (
                t.tile_type == TileType.JOKER,
                -t.get_value()
            ))
            connected = connected[:8]
        
        if len(connected) == 0:
            return []
        
        # Pool: table tiles + connected hand tiles
        pool = table_tiles + connected
        
        # Find all valid re-partitions using backtracking
        partitions = self._backtrack_search(pool)
        
        # NEW: Filter out partitions with invalid joker usage
        valid_partitions = []
        for partition in partitions:
            if self._validate_joker_usage(table_tiles, partition, connected):
                valid_partitions.append(partition)
        
        partitions = valid_partitions
        
        # Keep top N partitions by value played from hand
        if len(partitions) > self.max_melds_per_window:
            def score_partition(partition):
                return sum(
                    sum(t.get_value() for t in ts.tiles if t in connected)
                    for ts in partition
                )
            partitions.sort(key=score_partition, reverse=True)
            partitions = partitions[:self.max_melds_per_window]
        
        # Convert partitions to actions
        for partition in partitions:
            # Must use at least one hand tile
            tiles_from_hand = [
                t for ts in partition for t in ts.tiles 
                if t in connected
            ]
            
            if len(tiles_from_hand) == 0:
                continue
            
            # Build new table: unchanged sets + new partition
            new_table = []
            for idx, ts in enumerate(table):
                if idx not in table_indices:
                    new_table.append(ts)
            new_table.extend(partition)
            
            action = RummikubAction(
                action_type='play',
                tiles=tiles_from_hand,
                sets=partition,
                table_config=new_table
            )
            actions.append(action)
        
        return actions


    # =============================================================================
    # MODIFIED generate() METHOD
    # =============================================================================

    def generate_with_joker_support(self, hand: List, table: List) -> List:
        """
        Generate rearrangement actions with joker retrieval support.
        
        This replaces the original generate() method.
        """
        from Rummikub_env import RummikubAction
        
        if len(table) == 0:
            return []
        
        actions = []
        windows_explored = 0
        
        # Try single-set windows
        if self.max_window_size >= 1:
            for set_idx in range(len(table)):
                if windows_explored >= self.max_windows:
                    break
                
                # Use joker-aware exploration
                window_actions = self._explore_window_with_joker_awareness(
                    [set_idx], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        # Try two-set windows
        if self.max_window_size >= 2 and windows_explored < self.max_windows and len(table) >= 2:
            for idx1, idx2 in combinations(range(len(table)), 2):
                if windows_explored >= self.max_windows:
                    break
                
                window_actions = self._explore_window_with_joker_awareness(
                    [idx1, idx2], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        # Try three-set windows
        if self.max_window_size >= 3 and windows_explored < self.max_windows and len(table) >= 3:
            for idx1, idx2, idx3 in combinations(range(len(table)), 3):
                if windows_explored >= self.max_windows:
                    break
                
                window_actions = self._explore_window_with_joker_awareness(
                    [idx1, idx2, idx3], hand, table)
                actions.extend(window_actions)
                windows_explored += 1
        
        return actions


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