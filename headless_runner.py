import time
from core import State, make_move, make_move_opponent, alphabeta_search, minimax_search, column_from_move


def is_ai_turn(state, who_went_first):
    # Returns True if it's the 'ai' side's turn in this state, given who went first
    return (who_went_first == -1 and state.depth % 2 == 0) or (who_went_first == 0 and state.depth % 2 == 1)


def choose_move_column(state, search_fn, who_went_first, d):
    """Choose column for the side to move using search_fn.
    search_fn takes (State, turn, d) and returns a child State where .ai_position is the side that search_fn treats as 'ai'.

    If it's the actual ai's turn, call directly. If it's the opponent's turn, map positions so search_fn "ai" corresponds to the opponent, then map back.
    Returns (column, new_state)
    """
    def first_child(s, wf):
        try:
            return next(s.generate_children(wf))
        except StopIteration:
            return None

    if is_ai_turn(state, who_went_first):
        # 'ai' side to move: call search directly
        child = search_fn(state, who_went_first, d)
        if child is None:
            # fallback to first legal move if search returned None
            child = first_child(state, who_went_first)
            if child is None:
                return None, None
        col = column_from_move(state.ai_position, child.ai_position)
        # Apply move result to produce next state (child already is the next state), so return child
        return col, child
    else:
        # opponent to move — map opponent pieces to "ai" for the search
        search_state = State(state.player_position, state.game_position, depth=state.depth)
        flipped_turn = 0 if who_went_first == -1 else -1
        child = search_fn(search_state, flipped_turn, d)
        if child is None:
            # fallback to first legal opponent move
            child = first_child(search_state, flipped_turn)
            if child is None:
                return None, None
        # column relative to opponent pieces
        col = column_from_move(search_state.ai_position, child.ai_position)
        # apply this column to the original state as opponent move
        new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, col)
        new_state = State(new_ai_pos, new_mask, state.depth + 1)
        return col, new_state


def play_one_game(search_ai_fn, search_opponent_fn, depth_ai=5, depth_op=5, who_went_first=-1, max_moves=1000):
    """Play a single headless game where `search_ai_fn` is used for the 'ai' side and `search_opponent_fn` is used for the opponent side.

    who_went_first: -1 means the ai side (search_ai_fn) moved first; 0 means the opponent moved first.
    Returns winner: -1 (ai wins), 1 (opponent wins), 0 draw, and total moves and elapsed time per move stats
    """
    state = State(0, 0, depth=0)
    moves = 0
    total_time = 0.0
    while True:
        if state.terminal_node_test():
            return state.status, moves, total_time
        # determine which search to use for side to move
        if is_ai_turn(state, who_went_first):
            start = time.time()
            col, child = choose_move_column(state, search_ai_fn, who_went_first, depth_ai)
            elapsed = time.time() - start
            total_time += elapsed
            if child is None:
                # no legal move found -> treat as draw
                return 0, moves, total_time
            state = child
        else:
            start = time.time()
            # opponent sees its pieces as 'ai' for search
            # use search_opponent_fn but we must flip the role mapping inside choose_move_column
            # reuse choose_move_column logic by passing search_opponent_fn as search_fn
            col, state = choose_move_column(state, search_opponent_fn, who_went_first, depth_op)
            elapsed = time.time() - start
            total_time += elapsed
        moves += 1
        if moves > max_moves:
            # treat as draw
            return 0, moves, total_time


if __name__ == '__main__':
    # Quick comparison: alphabeta_search (with pruning) vs minimax_search (without pruning)
    # We'll run a small number of games to compare win counts and average times.
    NUM_GAMES = 10
    DEPTH = 5  # keep depth small for plain minimax to finish quickly

    results = {"alpha_wins": 0, "plain_wins": 0, "draws": 0}
    total_moves = 0
    total_time = 0.0

    print(f"Running {NUM_GAMES} headless games: AlphaBeta (depth={DEPTH}) vs Plain Minimax (depth={DEPTH})")

    for i in range(NUM_GAMES):
        # Alternate who goes first for fairness
        who_first = -1 if i % 2 == 0 else 0
        # When who_first == -1: ai (alphabeta) is the 'ai' side and moves first when depth%2==0
        # We'll treat 'ai' side as alphabeta, opponent as plain minimax
        winner, moves, time_spent = play_one_game(alphabeta_search, minimax_search, depth_ai=DEPTH, depth_op=DEPTH, who_went_first=who_first)
        total_moves += moves
        total_time += time_spent
        if winner == -1:
            # 'ai' side (alphabeta) won
            results['alpha_wins'] += 1
            print(f"Game {i+1}: AlphaBeta won in {moves} moves (who_first={who_first})")
        elif winner == 1:
            results['plain_wins'] += 1
            print(f"Game {i+1}: Plain Minimax won in {moves} moves (who_first={who_first})")
        else:
            results['draws'] += 1
            print(f"Game {i+1}: Draw in {moves} moves (who_first={who_first})")

    print('\nSummary:')
    print(f"AlphaBeta wins: {results['alpha_wins']}")
    print(f"Plain Minimax wins: {results['plain_wins']}")
    print(f"Draws: {results['draws']}")
    avg_moves = total_moves / NUM_GAMES
    avg_time = total_time / total_moves if total_moves > 0 else 0
    print(f"Avg moves per game: {avg_moves:.2f}")
    print(f"Avg time per move: {avg_time:.4f}s")
