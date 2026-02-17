"""
Rummikub Action Generator using ILP and Heuristics

Provides three types of action generation:
1. Generator 1: Simple hand plays (no table manipulation)
2. Generator 2: Table extensions (add to existing sets)
3. Generator 3: Complex rearrangements (windowed search with backtracking)

ENHANCED: Actions now include metadata for RL:
- set_types: List of set types ('run', 'group')
- is_extension: Whether action extends existing table sets
- meld_value: Total value of the meld

Usage:
    from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
    gen = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=30)
    env.action_generator = gen
    actions = env.get_legal_actions(player_id)
"""

from typing import List, Optional, Tuple, Dict, Set
from enum import Enum
from dataclasses import dataclass
import copy
from collections import defaultdict
import itertools
from itertools import combinations
import time

try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("WARNING: ortools not available. Install with: pip install ortools")

from Rummikub_env import Tile, TileSet, RummikubAction, Color, TileType


@dataclass
class SetTemplate:
    """Template representing a possible set configuration."""
    set_type: str
    pattern: List[Tuple[Optional[int], Optional[int]]]
    joker_count: int = 0
    template_id: int = 0


class SolverMode(Enum):
    """Action generator modes"""
    HEURISTIC_ONLY = "heuristic_only"
    HYBRID = "hybrid"
    ILP_ONLY = "ilp_only"


def get_key(tile: Tile) -> Tuple[Optional[int], Optional[int]]:
    """Get tile type key."""
    if tile.tile_type == TileType.JOKER:
        return (None, None)
    return (tile.color.value, tile.number)


class ActionGenerator:
    """Main action generator coordinating three sub-generators."""
    
    def __init__(self, mode: SolverMode = SolverMode.HYBRID, max_ilp_calls: int = 30,
                 max_window_size: int = 3, timeout_seconds: float = 30.0):
        self.mode = mode
        self.max_ilp_calls = max_ilp_calls
        self.max_window_size = max_window_size
        self.timeout_seconds = timeout_seconds
        
        self.hand_play_gen = HandPlayGenerator()
        self.table_ext_gen = TableExtensionGenerator()
        
        if mode != SolverMode.HEURISTIC_ONLY:
            self.rearrange_gen = RearrangementGenerator(
                max_windows=max_ilp_calls,
                max_window_size=max_window_size,
                ilp_time_limit=2.0
            )
        else:
            self.rearrange_gen = None
    
    def generate_all_legal_actions(self, hand_tiles: List, table_sets: List, 
                                   has_melded: bool, pool_size: int) -> List:
        return self.generate_actions(hand_tiles, table_sets, has_melded)
    
    def generate_actions(self, hand: List, table: List, has_melded: bool) -> List:
        actions = []
        start_time = time.time()
        
        def check_timeout():
            return time.time() - start_time > self.timeout_seconds
        
        try:
            if not has_melded:
                if check_timeout():
                    raise TimeoutError()
                initial_melds = self.hand_play_gen.generate_initial_melds(hand)
                actions.extend(initial_melds)
            else:
                if check_timeout():
                    raise TimeoutError()
                hand_actions = self.hand_play_gen.generate_hand_plays(hand, table)
                actions.extend(hand_actions)
                
                if len(table) > 0:
                    if check_timeout():
                        raise TimeoutError()
                    ext_actions = self.table_ext_gen.generate(hand, table)
                    actions.extend(ext_actions)
                
                if len(table) > 0 and self.rearrange_gen is not None:
                    if check_timeout():
                        raise TimeoutError()
                    remaining_time = self.timeout_seconds - (time.time() - start_time)
                    rearrange_actions = self.rearrange_gen.generate(hand, table, timeout=remaining_time)
                    actions.extend(rearrange_actions)
            
            if not check_timeout():
                actions = self._deduplicate_actions(actions)
            
            return actions
            
        except TimeoutError:
            return actions
        except Exception as e:
            print(f"Error during action generation: {e}")
            return actions
    
    def _deduplicate_actions(self, actions: List) -> List:
        seen = set()
        unique = []
        
        for action in actions:
            if action.action_type == 'draw':
                if 'draw' not in seen:
                    unique.append(action)
                    seen.add('draw')
                continue
            
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


class HandPlayGenerator:
    """Generator 1: Find valid runs and groups from hand tiles only."""
    
    def generate_initial_melds(self, hand: List) -> List:
        actions = []
        all_sets = self._find_all_valid_sets(hand)
        
        for num_sets in range(1, len(all_sets) + 1):
            for combo in combinations(all_sets, num_sets):
                all_tiles = [t for s in combo for t in s.tiles]
                if len(all_tiles) != len(set(t.tile_id for t in all_tiles)):
                    continue
                
                total_value = sum(s.get_meld_value() for s in combo)
                if total_value >= 30:
                    sets_copy = [copy.deepcopy(s) for s in combo]
                    set_types = [s.set_type for s in sets_copy]
                    
                    action = RummikubAction(
                        action_type='initial_meld',
                        tiles=all_tiles,
                        sets=sets_copy,
                        set_types=set_types,
                        is_extension=False,
                        meld_value=total_value
                    )
                    actions.append(action)
        
        return actions
    
    def generate_hand_plays(self, hand: List, table: List) -> List:
        actions = []
        all_sets = self._find_all_valid_sets(hand)
        
        for num_sets in range(1, len(all_sets) + 1):
            for combo in combinations(all_sets, num_sets):
                all_tiles = [t for s in combo for t in s.tiles]
                if len(all_tiles) != len(set(t.tile_id for t in all_tiles)):
                    continue
                
                sets_copy = [copy.deepcopy(s) for s in combo]
                new_table = copy.deepcopy(table) + sets_copy
                
                all_table_tile_ids = [t.tile_id for ts in new_table for t in ts.tiles]
                if len(all_table_tile_ids) != len(set(all_table_tile_ids)):
                    continue
                
                set_types = [s.set_type for s in sets_copy]
                meld_value = sum(s.get_meld_value() for s in sets_copy)
                
                action = RummikubAction(
                    action_type='play',
                    tiles=all_tiles,
                    sets=list(combo),
                    table_config=new_table,
                    set_types=set_types,
                    is_extension=False,
                    meld_value=meld_value
                )
                actions.append(action)
        
        return actions
    
    def _find_all_valid_sets(self, hand: List[Tile]) -> List[TileSet]:
        sets = []
        jokers = [t for t in hand if t.tile_type == TileType.JOKER]
        non_jokers = [t for t in hand if t.tile_type != TileType.JOKER]
        num_jokers = len(jokers)
        
        # Find groups
        by_number = defaultdict(list)
        for t in non_jokers:
            by_number[t.number].append(t)
        
        for num, tiles in by_number.items():
            by_color = defaultdict(list)
            for t in tiles:
                by_color[t.color].append(t)
            
            available_colors = list(by_color.keys())
            num_available = len(available_colors)
            
            for group_size in range(3, 5):
                for num_jok in range(max(0, group_size - num_available), min(num_jokers, group_size - 1) + 1):
                    num_needed_colors = group_size - num_jok
                    if num_needed_colors > num_available:
                        continue
                    
                    for selected_colors in itertools.combinations(available_colors, num_needed_colors):
                        selected_tiles = [by_color[col][0] for col in selected_colors]
                        selected_jokers = jokers[:num_jok]
                        group_tiles = selected_tiles + selected_jokers
                        new_group = TileSet(group_tiles, "group")
                        if new_group.is_valid():
                            sets.append(new_group)
        
        # Find runs
        by_color = defaultdict(list)
        for t in non_jokers:
            by_color[t.color].append(t)
        
        for col in Color:
            col_tiles = by_color[col]
            if len(col_tiles) == 0:
                continue
            
            by_num = defaultdict(list)
            for t in col_tiles:
                by_num[t.number].append(t)
            
            available_nums = sorted(by_num.keys())
            n = len(available_nums)
            
            for mask in range(1, 1 << n):
                subset_nums = [available_nums[i] for i in range(n) if (mask & (1 << i))]
                if len(subset_nums) < 1:
                    continue
                
                subset_nums.sort()
                min_num = subset_nums[0]
                max_num = subset_nums[-1]
                expected = max_num - min_num + 1
                missing = expected - len(subset_nums)
                
                if missing < 0 or missing > num_jokers:
                    continue
                
                if expected < 3:
                    continue
                
                selected_tiles = [by_num[num][0] for num in subset_nums]
                selected_jokers = jokers[:missing]
                run_tiles = selected_tiles + selected_jokers
                new_run = TileSet(run_tiles, "run")
                if new_run.is_valid():
                    sets.append(new_run)
        
        return sets


class TableExtensionGenerator:
    """Generator 2: Add hand tiles to existing table sets."""
    
    def generate(self, hand: List[Tile], table: List[TileSet]) -> List[RummikubAction]:
        actions = []
        
        for set_idx, tile_set in enumerate(table):
            connected = self._find_connectable_tiles(hand, tile_set)
            for num_add in range(1, len(connected) + 1):
                for adds in combinations(connected, num_add):
                    new_tiles = copy.deepcopy(tile_set.tiles) + list(adds)
                    new_set = TileSet(new_tiles, tile_set.set_type)
                    if new_set.is_valid():
                        new_table = copy.deepcopy(table)
                        new_table[set_idx] = new_set
                        
                        all_tile_ids = [t.tile_id for ts in new_table for t in ts.tiles]
                        if len(all_tile_ids) != len(set(all_tile_ids)):
                            continue
                        
                        meld_value = new_set.get_meld_value() - tile_set.get_meld_value()
                        
                        action = RummikubAction(
                            action_type='play',
                            tiles=list(adds),
                            table_config=new_table,
                            set_types=[tile_set.set_type],
                            is_extension=True,
                            meld_value=meld_value
                        )
                        actions.append(action)
        
        return actions
    
    def _find_connectable_tiles(self, hand: List[Tile], tile_set: TileSet) -> List[Tile]:
        connectable = []
        if tile_set.set_type == 'group':
            non_joker = [t for t in tile_set.tiles if t.tile_type != TileType.JOKER]
            if not non_joker:
                return connectable
            num = non_joker[0].number
            used_colors = set(t.color for t in tile_set.tiles if t.tile_type != TileType.JOKER)
            for t in hand:
                if t.tile_type == TileType.JOKER:
                    connectable.append(t)
                elif t.number == num and t.color not in used_colors:
                    connectable.append(t)
        elif tile_set.set_type == 'run':
            non_joker = [t for t in tile_set.tiles if t.tile_type != TileType.JOKER]
            if not non_joker:
                return connectable
            color = non_joker[0].color
            numbers = sorted(t.number for t in tile_set.tiles if t.number)
            min_num = numbers[0] if numbers else 1
            max_num = numbers[-1] if numbers else 13
            for t in hand:
                if t.tile_type == TileType.JOKER:
                    connectable.append(t)
                elif t.color == color and (t.number == min_num - 1 or t.number == max_num + 1):
                    connectable.append(t)
        return connectable


class RearrangementGenerator:
    """Generator 3: Complex table rearrangements using windowed ILP."""
    
    def __init__(self, max_windows: int = 30, max_window_size: int = 3, ilp_time_limit: float = 2.0):
        self.max_windows = max_windows
        self.max_window_size = max_window_size
        self.ilp_time_limit = ilp_time_limit
        self.templates = self._generate_all_possible_templates()
    
    def _generate_all_possible_templates(self) -> List[SetTemplate]:
        templates: List[SetTemplate] = []
        colors = range(4)
        
        for n in range(1, 14):
            for size in [3, 4]:
                for num_jokers in range(3):
                    if num_jokers > size:
                        continue
                    num_colors = size - num_jokers
                    if num_colors < 1 or num_colors > 4:
                        continue
                    for s in itertools.combinations(colors, num_colors):
                        pattern = [(c, n) for c in s] + [(None, None)] * num_jokers
                        templates.append(SetTemplate("group", pattern, num_jokers))
        
        for col in colors:
            for start in range(1, 14):
                for length in range(3, 15 - start + 1):
                    for num_jokers in range(3):
                        if num_jokers >= length:
                            continue
                        for joker_pos in itertools.combinations(range(length), num_jokers):
                            pattern = []
                            for pos in range(length):
                                if pos in joker_pos:
                                    pattern.append((None, None))
                                else:
                                    num = start + pos
                                    pattern.append((col, num))
                            templates.append(SetTemplate("run", pattern, num_jokers))
        
        return templates
    
    def generate(self, hand: List[Tile], table: List[TileSet], timeout: float = 30.0) -> List[RummikubAction]:
        actions = []
        start_time = time.time()
        
        if len(table) == 0:
            return []
        
        for window_size in range(1, self.max_window_size + 1):
            for table_indices in itertools.combinations(range(len(table)), window_size):
                if len(actions) >= self.max_windows:
                    break
                
                if time.time() - start_time > timeout:
                    return actions
                
                window_sets = [table[idx] for idx in table_indices]
                window_tiles = [t for s in window_sets for t in s.tiles]
                
                connected = self._filter_connected(hand, window_tiles)
                if not connected:
                    continue
                
                action = self._solve_window(window_tiles, connected, table_indices, table)
                if action:
                    actions.append(action)
        
        return actions
    
    def _solve_window(self, window_tiles: List[Tile], connected_hand: List[Tile], 
                      table_indices: Tuple[int], table: List[TileSet]) -> Optional[RummikubAction]:
        if not HAS_ORTOOLS:
            return None
            
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return None
        
        solver.SetTimeLimit(int(self.ilp_time_limit * 1000))
        
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
        
        x = [solver.BoolVar(f'x[{i}]') for i in range(num_templates)]
        y = {tt: solver.IntVar(0, type_inventory[tt]['count_hand'], f'y[{tt}]') for tt in tile_types}
        
        for tt in tile_types:
            used = solver.Sum(x[i] * possible_templates[i].pattern.count(tt) for i in range(num_templates))
            solver.Add(used == type_inventory[tt]['count_window'] + y[tt])
        
        obj = solver.Sum(
            y[tt] * (tt[1] if tt[1] is not None else 30)
            for tt in tile_types
        )
        solver.Maximize(obj)
        
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        total_hand_used = sum(y[tt].solution_value() for tt in tile_types)
        if total_hand_used < 1:
            return None
        
        selected = [(i, int(x[i].solution_value())) for i in range(num_templates) if x[i].solution_value() > 0]
        hand_used_count = {tt: int(y[tt].solution_value()) for tt in tile_types}
        
        hand_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in connected_hand:
            hand_by_type[get_key(t)].append(t)
        
        played_tiles: List[Tile] = []
        for tt, cnt in hand_used_count.items():
            if cnt > 0:
                played_tiles.extend(hand_by_type[tt][:cnt])
        
        all_used_tiles = window_tiles + played_tiles
        
        all_tile_ids = [t.tile_id for t in all_used_tiles]
        if len(all_tile_ids) != len(set(all_tile_ids)):
            return None
        
        available_by_type: Dict[Tuple[Optional[int], Optional[int]], List[Tile]] = defaultdict(list)
        for t in all_used_tiles:
            available_by_type[get_key(t)].append(t)
        
        new_sets: List[TileSet] = []
        set_types_created: List[str] = []
        
        for i, count in selected:
            for _ in range(count):
                templ = possible_templates[i]
                set_tiles: List[Tile] = []
                for p in templ.pattern:
                    tt = p
                    tiles_list = available_by_type[tt]
                    if not tiles_list:
                        return None
                    set_tiles.append(tiles_list.pop())
                
                if templ.set_type == "run":
                    set_tiles.sort(key=lambda t: t.number if t.number is not None else -1)
                
                new_set = TileSet(tiles=set_tiles, set_type=templ.set_type)
                if not new_set.is_valid():
                    return None
                
                set_ids = [t.tile_id for t in set_tiles]
                if len(set_ids) != len(set(set_ids)):
                    return None
                
                new_sets.append(new_set)
                set_types_created.append(templ.set_type)
        
        final_table = [copy.deepcopy(table[idx]) for idx in range(len(table)) if idx not in table_indices]
        final_table.extend(new_sets)
        
        all_final_tile_ids = [t.tile_id for ts in final_table for t in ts.tiles]
        if len(all_final_tile_ids) != len(set(all_final_tile_ids)):
            return None
        
        meld_value = sum(s.get_meld_value() for s in new_sets)
        
        return RummikubAction(
            action_type='play',
            tiles=played_tiles,
            sets=new_sets,
            table_config=final_table,
            set_types=set_types_created,
            is_extension=False,
            meld_value=meld_value
        )
    
    def _filter_connected(self, hand: List[Tile], window_tiles: List[Tile]) -> List[Tile]:
        connected = []
        
        for hand_tile in hand:
            if hand_tile.tile_type == TileType.JOKER:
                connected.append(hand_tile)
                continue
            
            for table_tile in window_tiles:
                if table_tile.tile_type == TileType.JOKER:
                    connected.append(hand_tile)
                    break
                
                if hand_tile.color == table_tile.color and abs(hand_tile.number - table_tile.number) <= 2:
                    connected.append(hand_tile)
                    break
                
                if hand_tile.number == table_tile.number and hand_tile.color != table_tile.color:
                    connected.append(hand_tile)
                    break
        
        return connected


if __name__ == "__main__":
    print("="*70)
    print("RUMMIKUB ACTION GENERATOR - TESTING")
    print("="*70)
    
    from Rummikub_env import RummikubEnv
    
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    gen = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=10)
    env.action_generator = gen
    
    print("\nTesting action metadata:")
    actions = env.get_legal_actions(0)
    print(f"Generated {len(actions)} actions")
    
    for i, action in enumerate(actions[:5]):
        if action.action_type != 'draw':
            print(f"  Action {i}: type={action.action_type}, "
                  f"set_types={action.set_types}, "
                  f"is_extension={action.is_extension}, "
                  f"meld_value={action.meld_value}")
    
    print("\nTesting complete!")