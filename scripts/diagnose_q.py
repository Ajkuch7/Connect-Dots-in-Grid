import json
import argparse
import os
import sys

# Ensure project root is on sys.path so top-level modules import correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from q_learning import QLearningAgent
from headless_qlearning import evaluate_agent, is_ai_turn
from core import State, make_move, make_move_opponent, minimax_search, column_from_move


def play_and_collect(agent, games=100, depth_op=5, collect_wins=5):
    samples = []
    collected = 0
    for g in range(1, games + 1):
        who_first = -1 if g % 2 == 0 else 0
        state = State(0, 0, depth=0)
        moves = []
        while True:
            if state.terminal_node_test():
                status = state.status
                if status == -1:
                    winner = 'ai'
                elif status == 1:
                    winner = 'opponent'
                else:
                    winner = 'draw'
                if winner == 'ai' and collected < collect_wins:
                    samples.append({'game': g, 'moves': moves, 'winner': winner})
                    collected += 1
                break

            if is_ai_turn(state, who_first):
                # minimax (ai) move
                res = minimax_search(state, who_first, d=depth_op)
                if res is None:
                    # choose random legal (shouldn't happen often)
                    legal = [c for c in range(7) if not (state.game_position & (1 << (7 * c + 5)))]
                    if not legal:
                        break
                    action = legal[0]
                else:
                    action = column_from_move(state.ai_position, res.ai_position)
                moves.append(('minimax', action))
                new_ai_pos, new_mask = make_move(state.ai_position, state.game_position, action)
                state = State(new_ai_pos, new_mask, state.depth + 1)
            else:
                # Q-agent (opponent) move; present state from agent's perspective
                transformed_state = State(state.player_position, state.game_position, state.depth)
                transformed_first = 0 if who_first == -1 else -1
                action = agent.choose_action(transformed_state, transformed_first, epsilon=0.0)
                moves.append(('q', action))
                new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, action)
                state = State(new_ai_pos, new_mask, state.depth + 1)

        if collected >= collect_wins:
            break

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qtable', default='q_table_trained_10k.pkl')
    parser.add_argument('--eval-games', type=int, default=200)
    parser.add_argument('--sample-games', type=int, default=5)
    parser.add_argument('--depth', type=int, default=5)
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.load(args.qtable)

    print('Running greedy eval vs random...')
    res = evaluate_agent(agent, opponent='random', depth_op=args.depth, games=args.eval_games)
    print('Eval vs random:', res)

    print(f'Collecting up to {args.sample_games} sample games where minimax beats the Q-agent...')
    samples = play_and_collect(agent, games=1000, depth_op=args.depth, collect_wins=args.sample_games)

    out = {'eval_vs_random': res, 'sample_games_minimax_wins': samples}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
