"""
Human vs ILP Solver Opponent - Interactive Testing

This allows you to play Rummikub against the pure ILP solver from Baseline_Opponent2.py
to test for logical bugs and verify behavior as per the Rummikub Solver paper.

The solver maximizes tiles played from hand while ensuring full table solvability via ILP.

Usage:
    python test2.py
"""

from Rummikub_env import RummikubEnv, RummikubAction, TileSet
from Baseline_Opponent2 import RummikubILPSolver
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode


def render_with_sorted_hands(env: RummikubEnv):
    """
    Custom render function that displays hands in sorted order.
    
    Sorts tiles by:
    1. Tile type (regular tiles before jokers)
    2. Color (Red, Blue, Black, Orange)
    3. Number (1-13)
    """
    from Rummikub_env import TileType
    
    def sort_hand(hand):
        """Sort hand tiles for display."""
        return sorted(hand, key=lambda t: (
            t.tile_type.value,                    # Regular tiles before jokers
            t.color.value if t.color else 99,     # Sort by color
            t.number if t.number else 99          # Sort by number
        ))
    
    print("\n" + "="*70)
    
    # Display each player's hand
    for player in range(2):
        hand = env.player_hands[player]
        sorted_hand = sort_hand(hand)
        hand_value = sum(t.get_value() for t in hand)
        hand_str = [str(t) for t in sorted_hand]
        
        if player == env.current_player:
            marker = " (current turn)"
        else:
            marker = ""
        
        print(f"Player {player} hand ({len(hand)} tiles, value={hand_value}){marker}: {hand_str}")
    
    # Display table
    print(f"\nTable ({len(env.table)} sets):")
    if env.table:
        for i, tile_set in enumerate(env.table):
            # Sort tiles for display
            display_tiles = tile_set.tiles.copy()
            if tile_set.set_type == 'run':
                display_tiles.sort(key=lambda t: t.number if t.number else 0)
            tiles_str = [str(t) for t in display_tiles]
            value = sum(t.get_value() for t in tile_set.tiles if t.tile_type != TileType.JOKER)
            print(f"  Set {i+1} ({tile_set.set_type}, value={value}): {tiles_str}")
    else:
        print("  (empty)")
    
    # Display pool
    print(f"\nPool: {len(env.tiles_deck)} tiles remaining")
    
    # Display meld status
    print(f"Initial meld status: Player 0={env.has_melded[0]}, Player 1={env.has_melded[1]}")
    
    print("="*70)


class HumanPlayer:
    """Interactive human player for testing."""
    
    def select_action(self, env: RummikubEnv) -> RummikubAction:
        """
        Let human select an action interactively.
        """
        current_player = env.current_player
        hand = env.player_hands[current_player]
        table = env.table
        has_melded = env.has_melded[current_player]
        pool_size = len(env.tiles_deck)
        
        print("\n" + "="*70)
        print("YOUR TURN")
        print("="*70)
        
        # Show game state
        self._display_state(env)
        
        # Get legal actions
        print("\nFinding legal actions...")
        
        import time
        start_time = time.time()
        legal_actions = env.get_legal_actions(current_player)
        elapsed = time.time() - start_time
        
        print(f"  (Found {len(legal_actions)} actions in {elapsed:.2f}s)")
        
        if len(legal_actions) == 0:
            print("ERROR: No legal actions found!")
            return None
        
        # Check if only draw action available
        play_actions = [a for a in legal_actions if a.action_type != 'draw']
        if len(play_actions) == 0:
            if not has_melded:
                print("\n  ℹ️  No valid initial meld found.")
                print("      You need sets totaling 30+ points to break the ice.")
                print(f"      Your hand value: {sum(t.get_value() for t in hand)}")
                print("      Try: Groups (3-4 same numbers) or Runs (3+ consecutive, same color)")
            else:
                print("\n  ℹ️  No valid plays from your hand.")
                print("      You can only draw this turn.")
        
        # Show legal actions
        print(f"\nYou have {len(legal_actions)} legal actions:")
        print()
        
        for i, action in enumerate(legal_actions):
            self._display_action(i, action, env)
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect action (0-{len(legal_actions)-1}): ").strip()
                idx = int(choice)
                
                if 0 <= idx < len(legal_actions):
                    return legal_actions[idx]
                else:
                    print(f"Invalid choice. Please enter 0-{len(legal_actions)-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user.")
                return None
    
    def _display_state(self, env: RummikubEnv):
        """Display current game state."""
        current_player = env.current_player
        hand = env.player_hands[current_player]
        table = env.table
        opponent_hand_size = len(env.player_hands[1 - current_player])
        
        print(f"\nYour hand ({len(hand)} tiles, value={sum(t.get_value() for t in hand)}):")
        sorted_hand = sorted(hand, key=lambda t: (t.tile_type.value, 
                                                   t.color.value if t.color else -1, 
                                                   t.number if t.number else -1))
        for i, tile in enumerate(sorted_hand):
            print(f"  [{i}] {tile}", end="")
            if (i + 1) % 8 == 0:
                print()
        print()
        
        print(f"\nTable ({len(table)} sets):")
        if table:
            for i, tile_set in enumerate(table):
                # Sort tiles for display (especially for runs)
                display_tiles = tile_set.tiles.copy()
                if tile_set.set_type == 'run':
                    # Sort by number for runs
                    display_tiles.sort(key=lambda t: t.number if t.number else 0)
                tiles_str = ", ".join(str(t) for t in display_tiles)
                value = sum(t.get_value() for t in tile_set.tiles if t.tile_type.name != 'JOKER')
                print(f"  Set {i+1} ({tile_set.set_type}, value={value}): [{tiles_str}]")
        else:
            print("  (empty)")
        
        print(f"\nOpponent: {opponent_hand_size} tiles")
        print(f"Pool: {len(env.tiles_deck)} tiles remaining")
        print(f"Has melded: {env.has_melded[current_player]}")
    
    def _display_action(self, idx: int, action: RummikubAction, env: RummikubEnv):
        """Display a single action option with clear indication of what's from hand."""
        if action.action_type == 'draw':
            print(f"  [{idx}] DRAW a tile from pool")
        
        elif action.action_type == 'initial_meld':
            tiles_str = ", ".join(str(t) for t in action.tiles)
            total_value = sum(s.get_meld_value() for s in action.sets)
            print(f"  [{idx}] INITIAL MELD (value={total_value}):")
            print(f"      ► Playing from hand: {tiles_str}")
            print(f"      ► Creating {len(action.sets)} set(s):")
            for i, tile_set in enumerate(action.sets):
                # Sort for display
                display_tiles = tile_set.tiles.copy()
                if tile_set.set_type == 'run':
                    display_tiles.sort(key=lambda t: t.number if t.number else 0)
                set_tiles = ", ".join(str(t) for t in display_tiles)
                set_value = tile_set.get_meld_value()
                print(f"         • [{set_tiles}] ({tile_set.set_type}, value={set_value})")
        
        elif action.action_type == 'play':
            tiles_str = ", ".join(str(t) for t in action.tiles)
            tiles_value = sum(t.get_value() for t in action.tiles)
            print(f"  [{idx}] PLAY {len(action.tiles)} tiles (value={tiles_value}):")
            print(f"      ► Playing from hand: {tiles_str}")
            
            # Determine which sets are new/modified vs unchanged
            if action.table_config:
                current_table = env.table
                hand_tile_ids = set(t.tile_id for t in action.tiles)
                
                # Categorize sets
                new_or_modified = []
                unchanged = []
                
                for tile_set in action.table_config:
                    set_tile_ids = set(t.tile_id for t in tile_set.tiles)
                    
                    # Check if contains hand tiles
                    has_hand_tile = bool(hand_tile_ids & set_tile_ids)
                    
                    if has_hand_tile:
                        new_or_modified.append(tile_set)
                    else:
                        # Check if exactly matches an existing table set
                        is_unchanged = any(
                            set_tile_ids == set(t.tile_id for t in table_set.tiles)
                            for table_set in current_table
                        )
                        if is_unchanged:
                            unchanged.append(tile_set)
                        else:
                            # Modified (rearrangement without hand tiles in this set)
                            new_or_modified.append(tile_set)
                
                # Display new/modified sets
                if new_or_modified:
                    print(f"      ► NEW/MODIFIED sets:")
                    for tile_set in new_or_modified:
                        display_tiles = tile_set.tiles.copy()
                        if tile_set.set_type == 'run':
                            display_tiles.sort(key=lambda t: t.number if t.number else 0)
                        set_tiles = ", ".join(str(t) for t in display_tiles)
                        
                        # Count tiles from hand vs table
                        from_hand = sum(1 for t in tile_set.tiles if t.tile_id in hand_tile_ids)
                        from_table = len(tile_set.tiles) - from_hand
                        
                        if from_table > 0:
                            detail = f", includes {from_table} from table"
                        else:
                            detail = ", all new"
                        
                        print(f"         • [{set_tiles}] ({tile_set.set_type}{detail})")
                
                # Show unchanged sets count
                if unchanged:
                    print(f"      • {len(unchanged)} table set(s) remain unchanged")
                
                print(f"      → Final table: {len(action.table_config)} total sets")


def play_game():
    """Main game loop for human vs ILP solver opponent."""
    
    print("\n" + "="*70)
    print("RUMMIKUB: Human vs Pure ILP Solver Opponent")
    print("="*70)
    print("\nTesting the RummikubILPSolver by playing against it.")
    print("This helps verify logical correctness and paper-described behavior:")
    print("  - Maximizes hand tiles played per turn")
    print("  - Guarantees full table solvability after rearrangement")
    print("  - Handles jokers and all valid melds via ILP")
    print("\nNote: Solver may take time on complex boards (observe for bugs/timeouts).")
    
    # Setup
    env = RummikubEnv(seed=None)  # Random seed for variety
    
    # Choose action generator mode for human legal actions
    print("\n" + "="*70)
    print("ACTION GENERATOR MODES (for Human Legal Actions)")
    print("="*70)
    print("""
The Action Generator has 3 modes that balance speed vs completeness for listing your legal moves:

MODE 1: HEURISTIC_ONLY (Fastest ~10ms)
  • Uses: Generator 1 (Hand Plays) + Generator 2 (Table Extensions)
  • Generator 1: Finds all valid sets from YOUR HAND ONLY
  • Generator 2: Extends EXISTING table sets (add tiles to runs/groups)
  • Missing: Generator 3 (complex table rearrangements)
  • Best for: Quick games, simple strategies

MODE 2: HYBRID (Balanced ~100ms) ⭐ RECOMMENDED
  • Uses: ALL THREE Generators
  • Generator 1: Hand plays (new sets from hand)
  • Generator 2: Table extensions (add to existing sets)
  • Generator 3: Rearrangements (manipulate 1-2 table sets + hand tiles)
  • Finds most moves without being too slow
  • Best for: Training RL agents, balanced gameplay

MODE 3: ILP_ONLY (Complete ~1s)
  • Uses: All generators with HIGHER search limits
  • Generator 3 explores MORE windows (more thorough)
  • Finds virtually all possible moves
  • Best for: Benchmarking, finding optimal plays
    """)
    
    print("Choose action generator mode:")
    print("  [1] HEURISTIC_ONLY: Very fast (~10ms), might miss complex rearrangements")
    print("  [2] HYBRID: Balanced (~100ms), finds most moves (RECOMMENDED)")
    print("  [3] ILP_ONLY: Complete (~1s), finds all moves")
    
    while True:
        choice = input("Select mode (1/2/3, default=2): ").strip()
        if choice == '1':
            mode = SolverMode.HEURISTIC_ONLY
            print("  Using HEURISTIC_ONLY mode (Generators 1+2)")
            break
        elif choice == '2' or choice == '':
            mode = SolverMode.HYBRID
            print("  Using HYBRID mode (All generators)")
            break
        elif choice == '3':
            mode = SolverMode.ILP_ONLY
            print("  Using ILP_ONLY mode (Complete search)")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Attach action generator to environment (for human legal actions)
    env.action_generator = ActionGenerator(mode=mode, max_ilp_calls=50)
    
    # Initialize pure ILP solver opponent
    opponent = RummikubILPSolver()
    
    human = HumanPlayer()
    
    # Choose player order
    print("\nChoose your position:")
    print("  [1] You go first")
    print("  [2] Opponent goes first")
    
    while True:
        choice = input("Select (1 or 2): ").strip()
        if choice == '1':
            human_player = 0
            break
        elif choice == '2':
            human_player = 1
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Start game
    print("\n" + "="*70)
    print("GAME START!")
    print("="*70)
    
    state = env.reset()
    render_with_sorted_hands(env)
    
    done = False
    turn_count = 0
    
    while not done:
        turn_count += 1
        print(f"\n{'='*70}")
        print(f"TURN {turn_count} - {'HUMAN' if env.current_player == human_player else 'ILP SOLVER OPPONENT'}")
        print(f"{'='*70}")
        
        if env.current_player == human_player:
            # Human's turn
            action = human.select_action(env)
            
            if action is None:
                print("Game interrupted.")
                return
            
            print(f"\nYou chose: {action.action_type}")
            
        else:
            # ILP Solver opponent's turn
            print("\nILP Solver is computing optimal move...")
            
            import time
            start_time = time.time()
            action = opponent.solve(
                env.player_hands[env.current_player],
                env.table,
                env.has_melded[env.current_player]
            )
            elapsed = time.time() - start_time
            
            if action is None:
                action = RummikubAction(action_type='draw')
                print(f"  Solver chose to DRAW (no valid play found in {elapsed:.2f}s)")
            else:
                print(f"  Solver chose: {action.action_type} (computed in {elapsed:.2f}s)")
                print(f"  Played {len(action.tiles)} tiles")
                tiles_str = ", ".join(str(t) for t in action.tiles)
                print(f"  Tiles: {tiles_str}")
                
                # Validate that all tiles are actually in opponent's hand
                opponent_hand = env.player_hands[env.current_player]
                opponent_tile_ids = [t.tile_id for t in opponent_hand]
                action_tile_ids = [t.tile_id for t in action.tiles]
                
                for action_tile_id in action_tile_ids:
                    if action_tile_id not in opponent_tile_ids:
                        print(f"  ⚠️ WARNING: Action includes tile {action_tile_id} not in opponent's hand!")
        
        # Execute action
        state, reward, done, info = env.step(action)
        
        # Show result
        print(f"\nReward: {reward}")
        if info.get('ice_broken'):
            print("  🎉 Ice broken! (30+ points played)")
        if info.get('manipulation_occurred'):
            print("  🔄 Table manipulation occurred")
        
        # Show updated state
        render_with_sorted_hands(env)
        
        # Check if game over
        if done:
            print("\n" + "="*70)
            print("GAME OVER!")
            print("="*70)
            
            if env.winner == human_player:
                print("\n🎉 YOU WIN! 🎉")
            elif env.winner == 1 - human_player:
                print("\n😞 ILP SOLVER WINS 😞")
            else:
                print("\n🤝 TIE 🤝")
            
            print(f"\nFinal scores:")
            print(f"  Your hand value: {sum(t.get_value() for t in env.player_hands[human_player])}")
            print(f"  ILP Solver hand value: {sum(t.get_value() for t in env.player_hands[1-human_player])}")
            print(f"\nTotal turns: {turn_count}")
            
            break
        
        # Pause between turns
        if env.current_player != human_player:
            input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    while True:
        try:
            play_game()
            
            # Play again?
            print("\n" + "="*70)
            choice = input("\nPlay again? (y/n): ").strip().lower()
            if choice != 'y':
                print("\nThanks for testing! Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nGame interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()