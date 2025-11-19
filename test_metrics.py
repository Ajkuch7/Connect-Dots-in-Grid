"""
Quick test to show AI metrics for both AlphaBeta and Plain Minimax
"""
import time
from core import State, alphabeta_search, minimax_search

# Test AlphaBeta and Plain Minimax metrics on the same initial board state
initial_state = State(0, 0, depth=0)

print("=" * 80)
print("AI METRICS COMPARISON: AlphaBeta vs Plain Minimax")
print("=" * 80)

# Test AlphaBeta at depth 7
print("\n[AlphaBeta Minimax]")
print("Depth: 7")
start = time.time()
result_ab = alphabeta_search(initial_state, turn=-1, d=7)
time_ab = time.time() - start
metrics_ab = getattr(result_ab, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0})
print(f"Nodes Explored: {metrics_ab['nodes_explored']}")
print(f"Nodes Pruned: {metrics_ab['nodes_pruned']}")
print(f"Total Nodes: {metrics_ab['nodes_explored'] + metrics_ab['nodes_pruned']}")
print(f"Pruning Ratio: {metrics_ab['nodes_pruned'] / (metrics_ab['nodes_explored'] + metrics_ab['nodes_pruned']) * 100:.1f}%" if (metrics_ab['nodes_explored'] + metrics_ab['nodes_pruned']) > 0 else "N/A")
print(f"Time: {time_ab:.4f}s")

# Test Plain Minimax at depth 5 and depth 7 for comparison
print("\n[Plain Minimax (Depth 5)]")
print("Depth: 5")
start = time.time()
result_pm5 = minimax_search(initial_state, turn=-1, d=5)
time_pm5 = time.time() - start
metrics_pm5 = getattr(result_pm5, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0})
print(f"Nodes Explored: {metrics_pm5['nodes_explored']}")
print(f"Nodes Pruned: {metrics_pm5['nodes_pruned']}")
print(f"Total Nodes: {metrics_pm5['nodes_explored'] + metrics_pm5['nodes_pruned']}")
print(f"Time: {time_pm5:.4f}s")

print("\n[Plain Minimax (Depth 6)]")
print("Depth: 6")
start = time.time()
result_pm6 = minimax_search(initial_state, turn=-1, d=6)
time_pm6 = time.time() - start
metrics_pm6 = getattr(result_pm6, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0})
print(f"Nodes Explored: {metrics_pm6['nodes_explored']}")
print(f"Nodes Pruned: {metrics_pm6['nodes_pruned']}")
print(f"Total Nodes: {metrics_pm6['nodes_explored'] + metrics_pm6['nodes_pruned']}")
print(f"Time: {time_pm6:.4f}s")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"AlphaBeta (depth 7):        {metrics_ab['nodes_explored']} nodes explored, {time_ab:.4f}s")
print(f"Plain Minimax (depth 5):    {metrics_pm5['nodes_explored']} nodes explored, {time_pm5:.4f}s")
print(f"Plain Minimax (depth 6):    {metrics_pm6['nodes_explored']} nodes explored, {time_pm6:.4f}s")
print(f"\nSpeedup (AlphaBeta vs Plain d6): {time_pm6 / time_ab:.2f}x faster")
print(f"Node reduction (AlphaBeta vs Plain d7 equivalent): {(1 - metrics_ab['nodes_explored'] / (metrics_pm6['nodes_explored'] * 1.5)) * 100:.1f}% fewer (rough estimate)")

