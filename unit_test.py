"""
Unit Tests for Rummikub Environment

Quick tests to verify environment is working correctly.

Usage:
    python test_environment.py
"""

from Rummikub_env import RummikubEnv, Tile, TileSet, Color, TileType
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode
from Baseline_Opponent import ILPOpponent
import numpy as np


def test_environment_creation():
    """Test that environment can be created."""
    print("Testing environment creation...")
    env = RummikubEnv(seed=42)
    assert env is not None
    assert len(env.tiles_deck) == 106
    print("  ✓ Environment created successfully")


def test_reset():
    """Test environment reset."""
    print("\nTesting environment reset...")
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    assert 'my_hand' in state
    assert 'table' in state
    assert 'opponent_tile_count' in state
    assert 'pool_size' in state
    
    assert len(state['my_hand']) == 14
    assert state['opponent_tile_count'] == 14
    assert len(state['table']) == 0
    print("  ✓ Reset works correctly")


def test_tile_validation():
    """Test tile and set validation."""
    print("\nTesting tile validation...")
    
    # Create test tiles
    tile1 = Tile(Color.RED, 5, TileType.NORMAL, 0)
    tile2 = Tile(Color.BLUE, 5, TileType.NORMAL, 1)
    tile3 = Tile(Color.BLACK, 5, TileType.NORMAL, 2)
    
    # Valid group
    group = TileSet([tile1, tile2, tile3], 'group')
    assert group.is_valid()
    print("  ✓ Valid group detected")
    
    # Invalid group (duplicate tile)
    invalid_group = TileSet([tile1, tile1, tile2], 'group')
    assert not invalid_group.is_valid()
    print("  ✓ Invalid group (duplicates) rejected")
    
    # Valid run
    tile4 = Tile(Color.RED, 6, TileType.NORMAL, 3)
    tile5 = Tile(Color.RED, 7, TileType.NORMAL, 4)
    run = TileSet([tile1, tile4, tile5], 'run')
    assert run.is_valid()
    print("  ✓ Valid run detected")
    
    # Invalid run (wrong colors)
    invalid_run = TileSet([tile1, tile2, tile3], 'run')
    assert not invalid_run.is_valid()
    print("  ✓ Invalid run (different colors) rejected")


def test_action_generator():
    """Test action generator modes."""
    print("\nTesting action generator...")
    
    for mode in [SolverMode.HEURISTIC_ONLY, SolverMode.HYBRID]:
        print(f"  Testing {mode.value} mode...")
        
        env = RummikubEnv(seed=42)
        env.action_generator = ActionGenerator(mode=mode, max_ilp_calls=5)
        
        state = env.reset()
        
        # Should always have at least draw action
        legal_actions = env.get_legal_actions(0)
        assert len(legal_actions) >= 1
        assert legal_actions[0].action_type == 'draw'
        
        print(f"    ✓ {mode.value} generated {len(legal_actions)} actions")


# def test_game_loop():
#     """Test a complete game can be played."""
#     print("\nTesting complete game loop...")
    
#     env = RummikubEnv(seed=42)
#     env.action_generator = ActionGenerator(mode=SolverMode.HEURISTIC_ONLY)
    
#     state = env.reset()
#     done = False
#     turns = 0
#     max_turns = 200
    
#     while not done and turns < max_turns:
#         legal_actions = env.get_legal_actions(env.current_player)
        
#         if not legal_actions:
#             print(f"    ERROR: No legal actions at turn {turns}")
#             break
        
#         # Take random action
#         action = np.random.choice(legal_actions)
#         state, reward, done, info = env.step(action)
#         turns += 1
    
#     if done:
#         print(f"  ✓ Game completed in {turns} turns")
#         print(f"    Winner: Player {env.winner}")
#     else:
#         print(f"  ✓ Game ran for {turns} turns (max reached)")


def test_reward_function():
    """Test reward function works correctly."""
    print("\nTesting reward function...")
    
    env = RummikubEnv(seed=42)
    env.action_generator = ActionGenerator(mode=SolverMode.HEURISTIC_ONLY)
    
    state = env.reset()
    
    # Test draw penalty
    draw_action = None
    for action in env.get_legal_actions(0):
        if action.action_type == 'draw':
            draw_action = action
            break
    
    if draw_action:
        hand_value_before = sum(t.get_value() for t in env.player_hands[0])
        state, reward, done, info = env.step(draw_action)
        hand_value_after = sum(t.get_value() for t in env.player_hands[0])
        
        expected_reward = hand_value_before - hand_value_after - 5
        assert reward == expected_reward, f"Expected {expected_reward}, got {reward}"
        print(f"  ✓ Draw penalty correct: {reward}")


def test_ilp_opponent():
    """Test ILP opponent can make moves."""
    print("\nTesting ILP opponent...")
    
    opponent = ILPOpponent(objective='maximize_value')
    
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    # Play until opponent can make a move
    for _ in range(50):
        if env.current_player == 1:
            action = opponent.select_action(
                env.player_hands[1],
                env.table,
                env.has_melded[1],
                len(env.tiles_deck)
            )
            
            assert action is not None
            print(f"  ✓ ILP opponent selected action: {action.action_type}")
            break
        
        # Advance game
        legal_actions = env.get_legal_actions(env.current_player)
        state, reward, done, info = env.step(legal_actions[0])
        
        if done:
            break


def test_two_opponents():
    """Test two ILP opponents playing against each other."""
    print("\nTesting ILP vs ILP...")
    
    env = RummikubEnv(seed=42)
    opponent1 = ILPOpponent(objective='maximize_value')
    opponent2 = ILPOpponent(objective='maximize_value_minimize_changes')
    
    state = env.reset()
    done = False
    turns = 0
    max_turns = 100
    
    while not done and turns < max_turns:
        if env.current_player == 0:
            opponent = opponent1
        else:
            opponent = opponent2
        
        action = opponent.select_action(
            env.player_hands[env.current_player],
            env.table,
            env.has_melded[env.current_player],
            len(env.tiles_deck)
        )
        
        state, reward, done, info = env.step(action)
        turns += 1
    
    if done:
        print(f"  ✓ ILP vs ILP completed in {turns} turns")
        print(f"    Winner: Player {env.winner}")
    else:
        print(f"  ✓ ILP vs ILP ran {turns} turns")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUMMIKUB ENVIRONMENT TESTS")
    print("="*70)
    
    tests = [
        test_environment_creation,
        test_reset,
        test_tile_validation,
        test_action_generator,
        # test_game_loop,
        test_reward_function,
        test_ilp_opponent,
        test_two_opponents,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! Environment is ready for use.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please review errors above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()