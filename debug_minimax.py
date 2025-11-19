from main import minimax_search, State, make_move
import time

# Simple debug test
print("Testing Plain Minimax directly...")
state = State(0, 0, depth=0)
print(f"Initial state: ai={state.ai_position}, game={state.game_position}, depth={state.depth}")

# Test generate_children directly
print("\nTesting generate_children...")
children_list = list(state.generate_children(turn=-1))
print(f"Generated {len(children_list)} children")
for i, child in enumerate(children_list[:3]):
    print(f"  Child {i}: ai={child.ai_position}, game={child.game_position}, depth={child.depth}")

print("\nNow testing minimax_search...")
start = time.time()
result = minimax_search(state, turn=-1, d=2)
elapsed = time.time() - start

print(f"Result: {result}")
if result:
    print(f"Result metrics: {getattr(result, 'metrics', 'NO METRICS')}")
    print(f"Result ai_pos: {result.ai_position}")
    print(f"Result game_pos: {result.game_position}")
else:
    print("Result is None!")
    
print(f"Time: {elapsed:.4f}s")


