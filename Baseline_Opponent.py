"""
ILP Baseline Opponent - Mathematical Solver from Paper

Implements BOTH models from the paper:
1. Model 1: Maximize tiles/value (simple)
2. Model 2: Maximize tiles/value + minimize table changes (secondary objective)

Usage:
    from ilp_baseline_opponent import ILPOpponent
    
    # Model 1: Simple maximization
    opponent = ILPOpponent(objective='maximize_value')
    
    # Model 2: Maximize value + minimize changes
    opponent = ILPOpponent(objective='maximize_value_minimize_changes')
    
    action = opponent.select_action(hand, table, has_melded, pool_size)
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("ERROR: ILP opponent requires ortools. Install with: pip install ortools")


class ILPOpponent:
    """
    Baseline opponent using ILP solver from the paper.
    
    Supports two models:
    1. maximize_tiles / maximize_value: Simple greedy optimization
    2. maximize_tiles_minimize_changes / maximize_value_minimize_changes: 
       Optimization with secondary objective to minimize table disruption
    """
    
    def __init__(self, objective: str = 'maximize_value', 
                 time_limit: float = 5.0,
                 M: int = 40,
                 timeout_seconds: float = 60.0):
        """
        Args:
            objective: One of:
                - 'maximize_tiles': Maximize number of tiles played (Model 1)
                - 'maximize_value': Maximize value of tiles played (Model 1)
                - 'maximize_tiles_minimize_changes': Max tiles + min changes (Model 2)
                - 'maximize_value_minimize_changes': Max value + min changes (Model 2)
            time_limit: Maximum time for each ILP solve (seconds)
            M: Constant for secondary objective (default 40, as in paper)
            timeout_seconds: Overall timeout for action selection (default 60s)
                           Set to None or float('inf') to disable timeout
        """
        if not HAS_ORTOOLS:
            raise ImportError("ILP opponent requires ortools. Install with: pip install ortools")
        
        valid_objectives = [
            'maximize_tiles', 
            'maximize_value',
            'maximize_tiles_minimize_changes',
            'maximize_value_minimize_changes'
        ]
        
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        
        self.objective = objective
        self.time_limit = time_limit
        self.M = M
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else float('inf')
        
        # Determine if using Model 2 (with minimize changes)
        self.minimize_changes = 'minimize_changes' in objective
        
        # Pre-compute all possible sets
        self.all_possible_sets = self._generate_all_set_templates()
        
        print(f"ILP Opponent initialized:")
        print(f"  Objective: {objective}")
        print(f"  Model: {'2 (minimize changes)' if self.minimize_changes else '1 (simple)'}")
        print(f"  Set templates: {len(self.all_possible_sets)}")
        if timeout_seconds is None or timeout_seconds == float('inf'):
            print(f"  Timeout: DISABLED")
        else:
            print(f"  Timeout: {timeout_seconds}s")
    
    def select_action(self, hand_tiles: List, table_sets: List, 
                     has_melded: bool, pool_size: int):
        """
        Select the optimal action using ILP solver.
        
        Returns:
            RummikubAction - the best action to take
            
        If timeout exceeded, returns draw action with warning message.
        """
        from Rummikub_env import RummikubAction
        import time
        
        start_time = time.time()
        
        try:
            # If can't meld yet, try to find initial meld
            if not has_melded:
                # Check timeout before searching
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError("Timeout before initial meld search")
                
                best_action = self._find_best_initial_meld(hand_tiles)
                if best_action:
                    return best_action
                # Can't meld, must draw
                return RummikubAction(action_type='draw')
            
            # After melding, find optimal play
            # Check timeout before ILP solve
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                raise TimeoutError("Timeout before ILP solve")
            
            if self.minimize_changes:
                best_action = self._solve_optimal_play_model2(hand_tiles, table_sets)
            else:
                best_action = self._solve_optimal_play_model1(hand_tiles, table_sets)
            
            # Check if we found a solution within timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds and best_action is None:
                raise TimeoutError("Timeout during ILP solve")
            
            if best_action:
                return best_action
            
            # No valid play found, draw
            return RummikubAction(action_type='draw')
            
        except TimeoutError as e:
            elapsed = time.time() - start_time
            print(f"\n⏱️  BASELINE OPPONENT TIMEOUT after {elapsed:.1f}s")
            print(f"    Falling back to DRAW action")
            return RummikubAction(action_type='draw')
        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Error in baseline opponent after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            print(f"    Falling back to DRAW action")
            return RummikubAction(action_type='draw')
    
    def _find_best_initial_meld(self, hand_tiles: List):
        """Find best initial meld (>= 30 points)."""
        from Rummikub_env import RummikubAction
        from itertools import combinations
        
        best_meld = None
        best_value = 0
        
        # Try all subsets
        for size in range(len(hand_tiles), 2, -1):
            for tile_combo in combinations(hand_tiles, size):
                tile_list = list(tile_combo)
                
                # Try to partition into valid sets
                partitions = self._find_valid_partitions(tile_list)
                
                for partition in partitions:
                    total_value = sum(s.get_meld_value() for s in partition)
                    
                    if total_value >= 30:
                        # Choose best based on objective
                        if 'tiles' in self.objective:
                            score = len(tile_list)
                        else:  # value
                            score = total_value
                        
                        if score > best_value:
                            best_value = score
                            tiles_used = []
                            for ts in partition:
                                tiles_used.extend(ts.tiles)
                            
                            best_meld = RummikubAction(
                                action_type='initial_meld',
                                tiles=tiles_used,
                                sets=partition,
                                table_config=partition
                            )
        
        return best_meld
    
    # =========================================================================
    # MODEL 1: Simple Maximization (from paper's first model)
    # =========================================================================
    
    def _solve_optimal_play_model1(self, hand_tiles: List, table_sets: List):
        """
        MODEL 1: Simple optimization
        
        Variables:
            x_j: set j can be placed 0, 1, or 2 times
            y_i: tile i can be placed 0, 1, or 2 from rack
        
        Constraints:
            sum(s_ij * x_j) = t_i + y_i  for all i
            y_i <= r_i  for all i
        
        Objective:
            Maximize sum(y_i) or sum(v_i * y_i)
        """
        from Rummikub_env import RummikubAction, TileType
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return None
        
        solver.SetTimeLimit(int(self.time_limit * 1000))
        
        # Build tile inventory
        tile_inventory = {}
        
        for tile_set in table_sets:
            for tile in tile_set.tiles:
                if tile.tile_id not in tile_inventory:
                    tile_inventory[tile.tile_id] = {'on_table': 0, 'in_hand': 0, 'tile': tile}
                tile_inventory[tile.tile_id]['on_table'] += 1
        
        for tile in hand_tiles:
            if tile.tile_id not in tile_inventory:
                tile_inventory[tile.tile_id] = {'on_table': 0, 'in_hand': 0, 'tile': tile}
            tile_inventory[tile.tile_id]['in_hand'] += 1
        
        # Create variables
        x_vars = {}  # x_j: set j appears this many times
        for j in range(len(self.all_possible_sets)):
            x_vars[j] = solver.IntVar(0, 2, f'x_{j}')
        
        y_vars = {}  # y_i: tile i played from hand
        for tile_id, info in tile_inventory.items():
            max_play = info['in_hand']
            y_vars[tile_id] = solver.IntVar(0, max_play, f'y_{tile_id}')
        
        # Build constraint matrix
        s_matrix = self._build_constraint_matrix(tile_inventory)
        
        # Add constraints: sum(s_ij * x_j) = t_i + y_i
        for tile_id, info in tile_inventory.items():
            t_i = info['on_table']
            
            constraint = solver.Constraint(t_i, t_i, f'tile_{tile_id}')
            
            for j in range(len(self.all_possible_sets)):
                if tile_id in s_matrix and j in s_matrix[tile_id]:
                    constraint.SetCoefficient(x_vars[j], s_matrix[tile_id][j])
            
            constraint.SetCoefficient(y_vars[tile_id], -1)
        
        # Set objective (MODEL 1: simple maximization)
        objective = solver.Objective()
        
        if 'tiles' in self.objective:
            # Maximize number of tiles played
            for tile_id, var in y_vars.items():
                objective.SetCoefficient(var, 1)
        else:  # value
            # Maximize value of tiles played
            for tile_id, var in y_vars.items():
                tile = tile_inventory[tile_id]['tile']
                value = tile.number if tile.tile_type != TileType.JOKER else 30
                objective.SetCoefficient(var, value)
        
        objective.SetMaximization()
        
        # Solve
        status = solver.Solve()
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        # Extract solution
        return self._extract_solution(solver, x_vars, y_vars, tile_inventory, 
                                      hand_tiles, table_sets)
    
    # =========================================================================
    # MODEL 2: Maximize + Minimize Changes (from paper's second model)
    # =========================================================================
    
    def _solve_optimal_play_model2(self, hand_tiles: List, table_sets: List):
        """
        MODEL 2: Maximize tiles/value + Minimize table changes
        
        Variables:
            x_j: set j can be placed 0, 1, or 2 times (new table)
            y_i: tile i can be placed 0, 1, or 2 from rack
            z_j: set j occurs in BOTH old and new solutions (unchanged)
        
        Parameters:
            w_j: set j is 0, 1, or 2 times on current table
        
        Constraints:
            sum(s_ij * x_j) = t_i + y_i  for all i
            y_i <= r_i  for all i
            z_j <= x_j  for all j
            z_j <= w_j  for all j
        
        Objective:
            Maximize sum(v_i * y_i) + (1/M) * sum(z_j)
            
        The (1/M) term ensures primary objective (tiles/value) dominates,
        while secondary objective (minimize changes) breaks ties.
        """
        from Rummikub_env import RummikubAction, TileType
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            return None
        
        solver.SetTimeLimit(int(self.time_limit * 1000))
        
        # Build tile inventory
        tile_inventory = {}
        
        for tile_set in table_sets:
            for tile in tile_set.tiles:
                if tile.tile_id not in tile_inventory:
                    tile_inventory[tile.tile_id] = {'on_table': 0, 'in_hand': 0, 'tile': tile}
                tile_inventory[tile.tile_id]['on_table'] += 1
        
        for tile in hand_tiles:
            if tile.tile_id not in tile_inventory:
                tile_inventory[tile.tile_id] = {'on_table': 0, 'in_hand': 0, 'tile': tile}
            tile_inventory[tile.tile_id]['in_hand'] += 1
        
        # Build w_j: count how many times each set appears on current table
        w_j = self._count_sets_on_table(table_sets)
        
        # Create variables
        x_vars = {}  # x_j: set j in new solution
        z_vars = {}  # z_j: set j unchanged (in both old and new)
        
        for j in range(len(self.all_possible_sets)):
            x_vars[j] = solver.IntVar(0, 2, f'x_{j}')
            z_vars[j] = solver.IntVar(0, 2, f'z_{j}')
        
        y_vars = {}  # y_i: tile i played from hand
        for tile_id, info in tile_inventory.items():
            max_play = info['in_hand']
            y_vars[tile_id] = solver.IntVar(0, max_play, f'y_{tile_id}')
        
        # Build constraint matrix
        s_matrix = self._build_constraint_matrix(tile_inventory)
        
        # Constraint 1: sum(s_ij * x_j) = t_i + y_i
        for tile_id, info in tile_inventory.items():
            t_i = info['on_table']
            
            constraint = solver.Constraint(t_i, t_i, f'tile_{tile_id}')
            
            for j in range(len(self.all_possible_sets)):
                if tile_id in s_matrix and j in s_matrix[tile_id]:
                    constraint.SetCoefficient(x_vars[j], s_matrix[tile_id][j])
            
            constraint.SetCoefficient(y_vars[tile_id], -1)
        
        # Constraint 2 & 3: z_j <= x_j and z_j <= w_j
        # This ensures z_j = min(x_j, w_j) = sets that remain unchanged
        for j in range(len(self.all_possible_sets)):
            # z_j <= x_j
            constraint1 = solver.Constraint(-solver.infinity(), 0, f'z_le_x_{j}')
            constraint1.SetCoefficient(z_vars[j], 1)
            constraint1.SetCoefficient(x_vars[j], -1)
            
            # z_j <= w_j
            w_j_value = w_j.get(j, 0)
            constraint2 = solver.Constraint(-solver.infinity(), w_j_value, f'z_le_w_{j}')
            constraint2.SetCoefficient(z_vars[j], 1)
        
        # Set objective (MODEL 2: primary + secondary)
        objective = solver.Objective()
        
        # Primary objective: Maximize tiles or value played
        if 'tiles' in self.objective:
            # Maximize number of tiles
            for tile_id, var in y_vars.items():
                objective.SetCoefficient(var, 1.0)
        else:  # value
            # Maximize value of tiles
            for tile_id, var in y_vars.items():
                tile = tile_inventory[tile_id]['tile']
                value = tile.number if tile.tile_type != TileType.JOKER else 30
                objective.SetCoefficient(var, float(value))
        
        # Secondary objective: Maximize unchanged sets (minimize changes)
        # Coefficient = 1/M to ensure it's a tiebreaker
        for j, var in z_vars.items():
            objective.SetCoefficient(var, 1.0 / self.M)
        
        objective.SetMaximization()
        
        # Solve
        status = solver.Solve()
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None
        
        # Extract solution
        return self._extract_solution(solver, x_vars, y_vars, tile_inventory, 
                                      hand_tiles, table_sets)
    
    def _count_sets_on_table(self, table_sets: List) -> dict:
        """
        Count how many times each set template appears on current table.
        Returns: dict mapping template_id -> count (w_j in the model)
        """
        w_j = {}
        
        for j, template in enumerate(self.all_possible_sets):
            count = 0
            
            for table_set in table_sets:
                if self._matches_template(table_set, template):
                    count += 1
            
            if count > 0:
                w_j[j] = count
        
        return w_j
    
    def _matches_template(self, tile_set, template) -> bool:
        """Check if a TileSet matches a template."""
        from Rummikub_env import TileType
        
        if tile_set.set_type != template.set_type:
            return False
        
        if len(tile_set.tiles) != len(template.pattern):
            return False
        
        # Count jokers
        set_jokers = sum(1 for t in tile_set.tiles if t.tile_type == TileType.JOKER)
        if set_jokers != template.joker_count:
            return False
        
        # For groups: same number, different colors
        if template.set_type == 'group':
            non_joker_tiles = [t for t in tile_set.tiles if t.tile_type != TileType.JOKER]
            if non_joker_tiles:
                numbers = [t.number for t in non_joker_tiles]
                if len(set(numbers)) != 1:
                    return False
        
        # For runs: consecutive numbers, same color
        elif template.set_type == 'run':
            non_joker_tiles = [t for t in tile_set.tiles if t.tile_type != TileType.JOKER]
            if non_joker_tiles:
                colors = [t.color for t in non_joker_tiles]
                if len(set(colors)) != 1:
                    return False
        
        return True
    
    # =========================================================================
    # SOLUTION EXTRACTION (shared by both models)
    # =========================================================================
    
    def _extract_solution(self, solver, x_vars, y_vars, tile_inventory, 
                         hand_tiles, table_sets):
        """Extract solution from ILP solver."""
        from Rummikub_env import RummikubAction
        
        # Check if any tiles played
        tiles_played = sum(y_vars[tid].solution_value() for tid in tile_inventory.keys())
        
        if tiles_played == 0:
            return None
        
        # Build new table configuration
        new_table = []
        used_tile_ids = []  # Use list to track tile IDs (allows duplicates for counting)
        
        for j, var in x_vars.items():
            count = int(var.solution_value())
            if count > 0:
                set_template = self.all_possible_sets[j]
                
                for _ in range(count):
                    tile_set = self._instantiate_set(set_template, tile_inventory, set(used_tile_ids))
                    if tile_set:
                        new_table.append(tile_set)
                        for tile in tile_set.tiles:
                            used_tile_ids.append(tile.tile_id)  # Append to list (allows duplicates)
        
        # Count occurrences of each tile_id
        from collections import Counter
        used_tile_counts = Counter(used_tile_ids)
        
        # Determine which hand tiles were played
        hand_tiles_played = []
        for tile in hand_tiles:
            # Check if this tile appears in new table MORE than on old table
            times_in_new_table = used_tile_counts.get(tile.tile_id, 0)
            times_on_old_table = tile_inventory[tile.tile_id]['on_table']
            
            # If appears MORE times in new table, it came from hand
            if times_in_new_table > times_on_old_table:
                hand_tiles_played.append(tile)
        
        if not new_table or not hand_tiles_played:
            return None
        
        action = RummikubAction(
            action_type='play',
            tiles=hand_tiles_played,
            sets=new_table,
            table_config=new_table
        )
        
        return action
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _generate_all_set_templates(self):
        """
        Generate all possible set templates INCLUDING JOKERS.
        
        Based on academic paper approach:
        - Runs of length 3-5 with 0, 1, or 2 jokers
        - Groups of size 3-4 with 0, 1, or 2 jokers
        
        This allows ILP to find joker manipulation strategies.
        """
        from Rummikub_ILP_Action_Generator import SetTemplate
        from itertools import combinations
        
        templates = []
        template_id = 0
        
        # =====================================================================
        # RUNS (with and without jokers)
        # =====================================================================
        
        for color in range(4):
            # Length 3 runs
            for start in range(1, 12):  # 1-11 start positions
                # Without jokers
                templates.append(SetTemplate(
                    'run', 
                    [(color, start+i) for i in range(3)], 
                    0, template_id))
                template_id += 1
                
                # With 1 joker (3 positions)
                for joker_pos in range(3):
                    pattern = []
                    for i in range(3):
                        if i == joker_pos:
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 1, template_id))
                    template_id += 1
                
                # With 2 jokers (3 combinations)
                for joker_pos1, joker_pos2 in combinations(range(3), 2):
                    pattern = []
                    for i in range(3):
                        if i in (joker_pos1, joker_pos2):
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 2, template_id))
                    template_id += 1
            
            # Length 4 runs
            for start in range(1, 11):  # 1-10 start positions
                # Without jokers
                templates.append(SetTemplate(
                    'run',
                    [(color, start+i) for i in range(4)],
                    0, template_id))
                template_id += 1
                
                # With 1 joker (4 positions)
                for joker_pos in range(4):
                    pattern = []
                    for i in range(4):
                        if i == joker_pos:
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 1, template_id))
                    template_id += 1
                
                # With 2 jokers (6 combinations)
                for joker_pos1, joker_pos2 in combinations(range(4), 2):
                    pattern = []
                    for i in range(4):
                        if i in (joker_pos1, joker_pos2):
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 2, template_id))
                    template_id += 1
            
            # Length 5 runs
            for start in range(1, 10):  # 1-9 start positions
                # Without jokers
                templates.append(SetTemplate(
                    'run',
                    [(color, start+i) for i in range(5)],
                    0, template_id))
                template_id += 1
                
                # With 1 joker (5 positions)
                for joker_pos in range(5):
                    pattern = []
                    for i in range(5):
                        if i == joker_pos:
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 1, template_id))
                    template_id += 1
                
                # With 2 jokers (10 combinations)
                for joker_pos1, joker_pos2 in combinations(range(5), 2):
                    pattern = []
                    for i in range(5):
                        if i in (joker_pos1, joker_pos2):
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((color, start+i))
                    templates.append(SetTemplate('run', pattern, 2, template_id))
                    template_id += 1
        
        # =====================================================================
        # GROUPS (with and without jokers)
        # =====================================================================
        
        for number in range(1, 14):  # Numbers 1-13
            # Size 3 groups
            for color_combo in combinations(range(4), 3):
                # Without jokers
                templates.append(SetTemplate(
                    'group',
                    [(c, number) for c in color_combo],
                    0, template_id))
                template_id += 1
                
                # With 1 joker (replace each color with joker)
                for skip_idx in range(3):
                    pattern = []
                    for i, c in enumerate(color_combo):
                        if i == skip_idx:
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((c, number))
                    templates.append(SetTemplate('group', pattern, 1, template_id))
                    template_id += 1
                
                # With 2 jokers (replace two colors with jokers)
                for skip_idx1, skip_idx2 in combinations(range(3), 2):
                    pattern = []
                    for i, c in enumerate(color_combo):
                        if i in (skip_idx1, skip_idx2):
                            pattern.append(('JOKER', 'JOKER'))
                        else:
                            pattern.append((c, number))
                    templates.append(SetTemplate('group', pattern, 2, template_id))
                    template_id += 1
            
            # Size 4 groups (all colors)
            # Without jokers
            templates.append(SetTemplate(
                'group',
                [(c, number) for c in range(4)],
                0, template_id))
            template_id += 1
            
            # With 1 joker (replace each color with joker)
            for skip_idx in range(4):
                pattern = []
                for i in range(4):
                    if i == skip_idx:
                        pattern.append(('JOKER', 'JOKER'))
                    else:
                        pattern.append((i, number))
                templates.append(SetTemplate('group', pattern, 1, template_id))
                template_id += 1
            
            # With 2 jokers (replace two colors with jokers)
            for skip_idx1, skip_idx2 in combinations(range(4), 2):
                pattern = []
                for i in range(4):
                    if i in (skip_idx1, skip_idx2):
                        pattern.append(('JOKER', 'JOKER'))
                    else:
                        pattern.append((i, number))
                templates.append(SetTemplate('group', pattern, 2, template_id))
                template_id += 1
        
        print(f"Generated {template_id} set templates (including joker scenarios)")
        return templates
    
    def _build_constraint_matrix(self, tile_inventory):
        """Build s_ij matrix."""
        from Rummikub_env import TileType
        
        s_matrix = {}
        
        for tile_id, info in tile_inventory.items():
            tile = info['tile']
            s_matrix[tile_id] = {}
            
            for j, template in enumerate(self.all_possible_sets):
                count = 0
                
                for pattern_pos in template.pattern:
                    if pattern_pos == ('JOKER', 'JOKER'):
                        if tile.tile_type == TileType.JOKER:
                            count += 1
                    else:
                        p_color, p_number = pattern_pos
                        if (tile.tile_type != TileType.JOKER and
                            tile.color.value == p_color and
                            tile.number == p_number):
                            count += 1
                
                if count > 0:
                    s_matrix[tile_id][j] = count
        
        return s_matrix
    
    def _instantiate_set(self, set_template, tile_inventory, used_tiles):
        """Instantiate a set from template."""
        from Rummikub_env import TileSet, TileType
        
        available_tiles = []
        for tile_id, info in tile_inventory.items():
            if tile_id not in used_tiles:
                available_tiles.append(info['tile'])
        
        if len(available_tiles) < len(set_template.pattern):
            return None
        
        selected_tiles = []
        temp_used = set()
        
        for pattern_pos in set_template.pattern:
            found = False
            
            for tile in available_tiles:
                if tile.tile_id in temp_used:
                    continue
                
                if pattern_pos == ('JOKER', 'JOKER'):
                    if tile.tile_type == TileType.JOKER:
                        selected_tiles.append(tile)
                        temp_used.add(tile.tile_id)
                        found = True
                        break
                else:
                    p_color, p_number = pattern_pos
                    if (tile.tile_type != TileType.JOKER and
                        tile.color.value == p_color and
                        tile.number == p_number):
                        selected_tiles.append(tile)
                        temp_used.add(tile.tile_id)
                        found = True
                        break
            
            if not found:
                return None
        
        tile_set = TileSet(tiles=selected_tiles, set_type=set_template.set_type)
        return tile_set if tile_set.is_valid() else None
    
    def _find_valid_partitions(self, tiles):
        """Find all valid partitions of tiles."""
        from Rummikub_env import TileSet
        from itertools import combinations
        
        if len(tiles) < 3:
            return []
        
        partitions = []
        
        for size in range(3, len(tiles) + 1):
            for combo in combinations(tiles, size):
                tile_list = list(combo)
                
                for set_type in ['run', 'group']:
                    test_set = TileSet(tiles=tile_list, set_type=set_type)
                    if test_set.is_valid():
                        remaining = [t for t in tiles if t not in combo]
                        
                        if len(remaining) == 0:
                            partitions.append([test_set])
                        elif len(remaining) >= 3:
                            sub = self._find_valid_partitions(remaining)
                            for s in sub:
                                partitions.append([test_set] + s)
        
        return partitions


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Example 1: Model 1 vs Model 2
------------------------------
from Rummikub_env import RummikubEnv
from ilp_baseline_opponent import ILPOpponent

env = RummikubEnv()

# Model 1: Simple maximization (greedy)
opponent_model1 = ILPOpponent(objective='maximize_value')

# Model 2: Maximize + minimize table changes
opponent_model2 = ILPOpponent(objective='maximize_value_minimize_changes')

state = env.reset()

# Both opponents see same state
hand = env.player_hands[0]
table = env.table
has_melded = env.has_melded[0]
pool_size = len(env.tiles_deck)

# Compare decisions
action1 = opponent_model1.select_action(hand, table, has_melded, pool_size)
action2 = opponent_model2.select_action(hand, table, has_melded, pool_size)

print(f"Model 1: {len(action1.tiles)} tiles played")
print(f"Model 2: {len(action2.tiles)} tiles played (with minimal disruption)")


Example 2: All Four Objectives
-------------------------------
objectives = [
    'maximize_tiles',
    'maximize_value',
    'maximize_tiles_minimize_changes',
    'maximize_value_minimize_changes'
]

for obj in objectives:
    opponent = ILPOpponent(objective=obj)
    action = opponent.select_action(hand, table, True, pool_size)
    print(f"{obj}: {len(action.tiles)} tiles, changes={...}")


Example 3: Train Against Model 2 (Smarter Opponent)
----------------------------------------------------
# Model 2 is harder to beat because it plays more "human-like"
# by not unnecessarily disrupting the table

env = RummikubEnv()
opponent = ILPOpponent(objective='maximize_value_minimize_changes', M=40)
agent = YourRLAgent()

for episode in range(10000):
    state = env.reset()
    done = False
    
    while not done:
        if env.current_player == 0:
            # Your agent
            legal_actions = env.get_legal_actions(0)
            action = agent.select_action(state, legal_actions)
        else:
            # Smarter ILP opponent (Model 2)
            action = opponent.select_action(
                env.player_hands[1],
                env.table,
                env.has_melded[1],
                len(env.tiles_deck)
            )
        
        state, reward, done, info = env.step(action)


Example 4: Tune M Parameter
----------------------------
# M controls balance between primary and secondary objectives
# Default M=40 (from paper)

# Low M: Secondary objective has more influence
opponent_low_m = ILPOpponent(
    objective='maximize_value_minimize_changes',
    M=20  # More emphasis on minimizing changes
)

# High M: Primary objective dominates more
opponent_high_m = ILPOpponent(
    objective='maximize_value_minimize_changes',
    M=100  # Less emphasis on minimizing changes
)


Example 5: Compare Model Behavior
----------------------------------
def compare_models(hand, table, has_melded, pool_size):
    model1 = ILPOpponent(objective='maximize_value')
    model2 = ILPOpponent(objective='maximize_value_minimize_changes')
    
    action1 = model1.select_action(hand, table, has_melded, pool_size)
    action2 = model2.select_action(hand, table, has_melded, pool_size)
    
    print("Model 1 (Simple):")
    print(f"  Tiles played: {len(action1.tiles)}")
    print(f"  Value: {sum(t.get_value() for t in action1.tiles)}")
    print(f"  New table sets: {len(action1.table_config)}")
    
    print("\nModel 2 (Minimize Changes):")
    print(f"  Tiles played: {len(action2.tiles)}")
    print(f"  Value: {sum(t.get_value() for t in action2.tiles)}")
    print(f"  New table sets: {len(action2.table_config)}")
    
    # Count unchanged sets
    unchanged = sum(1 for s1 in table for s2 in action2.table_config 
                   if s1 == s2)
    print(f"  Unchanged sets: {unchanged}/{len(table)}")

# Model 2 should preserve more existing sets when possible
"""