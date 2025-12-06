"""Evaluate Minimax, Alpha-Beta, and Q-Learning agents.

Runs headless matches and reports win/draw counts, average moves, average time
per move, and (for search agents) aggregated nodes explored/pruned.
"""
import time
import argparse
from collections import defaultdict

from core import State, make_move, make_move_opponent, column_from_move
from core import minimax_search, alphabeta_search
from q_learning import QLearningAgent


def is_ai_turn(state, who_went_first):
    return (who_went_first == -1 and state.depth % 2 == 0) or (who_went_first == 0 and state.depth % 2 == 1)


def q_search_factory(agent):
    """Return a search-like function compatible with headless_runner.play_one_game.

    The function signature is (state, turn, d) and returns a child State.
    """

    def q_search(state, turn, d=0):
        # Agent expects pieces in state.ai_position
        col = agent.choose_action(state, turn, epsilon=0.0)
        if col is None:
            return None
        new_ai_pos, new_mask = make_move(state.ai_position, state.game_position, col)
        child = State(new_ai_pos, new_mask, state.depth + 1)
        # attach empty metrics
        child.metrics = {'nodes_explored': 0, 'nodes_pruned': 0}
        return child

    return q_search


def choose_move_and_metrics(state, search_fn, who_went_first, d):
    """Choose a child state and return (col, child, metrics).

    Handles the mapping when it's the opponent's turn by transforming the state
    so the search treats the moving side as 'ai'.
    """
    # If it's the 'ai' side's turn relative to who_went_first, call directly
    if is_ai_turn(state, who_went_first):
        child = search_fn(state, who_went_first, d)
        if child is None:
            # fallback to first legal child
            try:
                child = next(state.generate_children(who_went_first))
            except StopIteration:
                return None, None, {'nodes_explored': 0, 'nodes_pruned': 0}
        col = column_from_move(state.ai_position, child.ai_position)
        metrics = getattr(child, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0})
        return col, child, metrics
    else:
        # Opponent to move - transform
        transformed_state = State(state.player_position, state.game_position, state.depth)
        flipped_turn = 0 if who_went_first == -1 else -1
        child_trans = search_fn(transformed_state, flipped_turn, d)
        if child_trans is None:
            # fallback to first legal transformed child
            try:
                child_trans = next(transformed_state.generate_children(flipped_turn))
            except StopIteration:
                return None, None, {'nodes_explored': 0, 'nodes_pruned': 0}
        # Determine column from transformed result
        col = column_from_move(transformed_state.ai_position, child_trans.ai_position)
        # Apply as opponent move to original state
        new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, col)
        child = State(new_ai_pos, new_mask, state.depth + 1)
        metrics = getattr(child_trans, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0})
        return col, child, metrics


def play_and_collect(search_a, search_b, episodes=100, depth_a=5, depth_b=5, record_episodes=False):
    stats = defaultdict(int)
    times = []
    moves_total = 0
    nodes_explored = 0
    nodes_pruned = 0
    episodes_list = []

    for ep in range(episodes):
        who_first = -1 if ep % 2 == 0 else 0
        state = State(0, 0, depth=0)
        moves = 0
        total_time = 0.0
        while True:
            if state.terminal_node_test():
                stats['wins_ai'] += 1 if state.status == -1 else 0
                stats['wins_opponent'] += 1 if state.status == 1 else 0
                stats['draws'] += 1 if state.status == 0 else 0
                # record episode
                if record_episodes:
                    episodes_list.append({
                        'winner': ('ai' if state.status == -1 else ('opponent' if state.status == 1 else 'draw')),
                        'moves': moves,
                        'time': total_time,
                        'nodes_explored': 0,
                        'nodes_pruned': 0,
                    })
                break

            if is_ai_turn(state, who_first):
                start = time.time()
                col, child, metrics = choose_move_and_metrics(state, search_a, who_first, depth_a)
                elapsed = time.time() - start
                total_time += elapsed
                if child is None:
                    # no legal move -> draw
                    stats['draws'] += 1
                    if record_episodes:
                        episodes_list.append({
                            'winner': 'draw',
                            'moves': moves,
                            'time': total_time,
                            'nodes_explored': 0,
                            'nodes_pruned': 0,
                        })
                    break
                state = child
                nodes_explored += metrics.get('nodes_explored', 0)
                nodes_pruned += metrics.get('nodes_pruned', 0)
            else:
                start = time.time()
                col, child, metrics = choose_move_and_metrics(state, search_b, who_first, depth_b)
                elapsed = time.time() - start
                total_time += elapsed
                if child is None:
                    stats['draws'] += 1
                    if record_episodes:
                        episodes_list.append({
                            'winner': 'draw',
                            'moves': moves,
                            'time': total_time,
                            'nodes_explored': 0,
                            'nodes_pruned': 0,
                        })
                    break
                state = child
                nodes_explored += metrics.get('nodes_explored', 0)
                nodes_pruned += metrics.get('nodes_pruned', 0)

            moves += 1
            if moves > 1000:
                stats['draws'] += 1
                break

        # record per-episode metrics (fill nodes/pruned for this episode)
        if record_episodes:
            # episodes_list last element corresponds to terminal state or earlier breaks; update its node counts if present
            if episodes_list and isinstance(episodes_list[-1], dict):
                episodes_list[-1]['nodes_explored'] = nodes_explored - sum(e.get('nodes_explored', 0) for e in episodes_list[:-1]) if episodes_list else nodes_explored
                episodes_list[-1]['nodes_pruned'] = nodes_pruned - sum(e.get('nodes_pruned', 0) for e in episodes_list[:-1]) if episodes_list else nodes_pruned
        times.append(total_time)
        moves_total += moves

    episodes_done = episodes
    report = {
        'episodes': episodes_done,
        'wins_ai': stats['wins_ai'],
        'wins_opponent': stats['wins_opponent'],
        'draws': stats['draws'],
        'avg_moves_per_game': moves_total / episodes_done if episodes_done else 0,
        'avg_time_per_game': sum(times) / episodes_done if episodes_done else 0,
        'avg_nodes_explored_per_game': nodes_explored / episodes_done if episodes_done else 0,
        'avg_nodes_pruned_per_game': nodes_pruned / episodes_done if episodes_done else 0,
    }
    if record_episodes:
        return report, episodes_list
    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate algorithms')
    parser.add_argument('--episodes', '-n', type=int, default=100, help='Number of games per matchup')
    parser.add_argument('--depth', type=int, default=5, help='Search depth for minimax/alphabeta')
    parser.add_argument('--qtable', type=str, default='q_table.pkl', help='Path to q-table pickle')
    parser.add_argument('--out', '-o', type=str, default=None, help='Optional JSON output file to write results')
    parser.add_argument('--out-detailed', '-D', type=str, default=None, help='Optional detailed JSON output file with per-episode records')
    args = parser.parse_args()

    # Prepare q-agent
    q_agent = QLearningAgent()
    try:
        q_agent.load(args.qtable)
        print(f'Loaded Q-table from {args.qtable}')
    except Exception:
        print(f'Could not load Q-table from {args.qtable}; running with untrained agent')

    q_search = q_search_factory(q_agent)

    print('Evaluating: AlphaBeta vs Plain Minimax')
    r1 = play_and_collect(alphabeta_search, minimax_search, episodes=args.episodes, depth_a=args.depth, depth_b=args.depth, record_episodes=bool(args.out_detailed))
    print(r1)

    print('\nEvaluating: AlphaBeta vs Q-Learning')
    r2 = play_and_collect(alphabeta_search, q_search, episodes=args.episodes, depth_a=args.depth, depth_b=0, record_episodes=bool(args.out_detailed))
    print(r2)

    print('\nEvaluating: Plain Minimax vs Q-Learning')
    r3 = play_and_collect(minimax_search, q_search, episodes=args.episodes, depth_a=args.depth, depth_b=0, record_episodes=bool(args.out_detailed))
    print(r3)

    # Normalize results to include optional per-episode lists
    def normalize(res):
        if isinstance(res, tuple):
            report, episodes = res
            return {'report': report, 'episodes': episodes}
        return {'report': res, 'episodes': None}

    results = {
        'AlphaBeta vs Plain Minimax': normalize(r1),
        'AlphaBeta vs Q-Learning': normalize(r2),
        'Plain Minimax vs Q-Learning': normalize(r3),
    }

    if args.out:
        try:
            import json
            with open(args.out, 'w', encoding='utf-8') as jf:
                json.dump(results, jf, indent=2)
            print(f'Wrote results JSON to {args.out}')
        except Exception as e:
            print(f'Failed to write JSON to {args.out}: {e}')
    if args.out_detailed:
        try:
            import json
            with open(args.out_detailed, 'w', encoding='utf-8') as jf:
                json.dump(results, jf, indent=2)
            print(f'Wrote detailed results JSON to {args.out_detailed}')
        except Exception as e:
            print(f'Failed to write detailed JSON to {args.out_detailed}: {e}')


if __name__ == '__main__':
    main()
