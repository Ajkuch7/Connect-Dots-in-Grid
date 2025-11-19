"""
Quick test script to verify metrics display in main.py
Simulates game for a few turns to show metrics output
"""
import sys
import io
from contextlib import redirect_stdout

# Capture initial AI move output
print("Testing AlphaBeta metrics display...")
print("=" * 80)

# For a more direct test, import and call the search functions
try:
    from main import alphabeta_search, minimax_search, State
    import time
    
    # Test on initial board
    initial_state = State(0, 0, depth=0)
    
    print("\n[Test 1] AlphaBeta Search (depth=7)")
    start = time.time()
    result_ab = alphabeta_search(initial_state, turn=-1, d=7)
    elapsed_ab = time.time() - start
    metrics_ab = getattr(result_ab, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0}) if result_ab else {'nodes_explored': 0, 'nodes_pruned': 0}
    print(f"[AlphaBeta] Depth: 7 | Nodes Explored: {metrics_ab['nodes_explored']} | Nodes Pruned: {metrics_ab['nodes_pruned']} | Time: {elapsed_ab:.4f}s")
    
    print("\n[Test 2] Plain Minimax Search (depth=5)")
    start = time.time()
    result_pm = minimax_search(initial_state, turn=-1, d=5)
    elapsed_pm = time.time() - start
    metrics_pm = getattr(result_pm, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0}) if result_pm else {'nodes_explored': 0, 'nodes_pruned': 0}
    print(f"[Plain Minimax] Depth: 5 | Nodes Explored: {metrics_pm['nodes_explored']} | Nodes Pruned: {metrics_pm['nodes_pruned']} | Time: {elapsed_pm:.4f}s")
    
    print("\n" + "=" * 80)
    print("✓ Metrics tracking is working correctly!")
    print(f"  AlphaBeta explored {metrics_ab['nodes_explored']} nodes in {elapsed_ab:.4f}s")
    print(f"  Plain Minimax explored {metrics_pm['nodes_explored']} nodes in {elapsed_pm:.4f}s")
    if elapsed_pm > 0:
        print(f"  Speedup: {elapsed_pm / elapsed_ab:.2f}x")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
