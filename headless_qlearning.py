"""Train a Q-learning agent for Connect Four in headless mode.

This script is intentionally simple: it uses the `State`, `make_move`, and
`make_move_opponent` helpers from `core.py`. It supports training the agent
via self-play where the opponent is either random or a search-based player
(minimax/alphabeta) provided by `core`.
"""
import random
import time
import argparse
from q_learning import QLearningAgent
from core import State, make_move, make_move_opponent, minimax_search, alphabeta_search


def is_ai_turn(state, who_went_first):
    return (who_went_first == -1 and state.depth % 2 == 0) or (who_went_first == 0 and state.depth % 2 == 1)


def train(agent, episodes=2000, opponent='random', depth_op=3, who_first_policy='alternate', save_path='q_table.pkl'):
    wins = 0
    losses = 0
    draws = 0
    start = time.time()

    for ep in range(1, episodes + 1):
        # alternate who moves first for fairness
        if who_first_policy == 'alternate':
            who_first = -1 if ep % 2 == 0 else 0
        else:
            who_first = random.choice([-1, 0])

        state = State(0, 0, depth=0)

        prev_state = None
        prev_action = None

        while True:
            if state.terminal_node_test():
                # terminal reached; determine reward for agent perspective
                status = state.status
                if status == -1:
                    reward = 1.0
                    wins += 1
                elif status == 1:
                    reward = -1.0
                    losses += 1
                else:
                    reward = 0.0
                    draws += 1

                # update last transition
                if prev_state is not None:
                    agent.update(prev_state, prev_action, reward, state, True)
                break

            # decide whose policy to use for this turn
            if is_ai_turn(state, who_first):
                # agent's turn: agent expects state.ai_position to be agent's pieces
                action = agent.choose_action(state, who_first, epsilon=agent.epsilon)
                if action is None:
                    # no legal moves
                    if prev_state is not None:
                        agent.update(prev_state, prev_action, 0.0, None, True)
                    break

                # apply action
                new_ai_pos, new_mask = make_move(state.ai_position, state.game_position, action)
                next_state = State(new_ai_pos, new_mask, state.depth + 1)

                # update Q for previous step
                if prev_state is not None:
                    agent.update(prev_state, prev_action, 0.0, next_state, False)

                # store as prev for next step
                prev_state = state
                prev_action = action
                state = next_state
            else:
                # opponent's turn
                if opponent == 'random':
                    legal = [c for c in range(7) if not (state.game_position & (1 << (7 * c + 5)))]
                    if not legal:
                        break
                    action = random.choice(legal)
                    # opponent move applied as make_move_opponent
                    new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, action)
                    state = State(new_ai_pos, new_mask, state.depth + 1)
                elif opponent == 'minimax':
                    # Use minimax_search where the opponent is treated as 'ai' in transformed state
                    transformed_state = State(state.player_position, state.game_position, state.depth)
                    transformed_first = 0 if who_first == -1 else -1
                    res = minimax_search(transformed_state, transformed_first, d=depth_op)
                    if res is None:
                        # fallback random
                        legal = [c for c in range(7) if not (state.game_position & (1 << (7 * c + 5)))]
                        if not legal:
                            break
                        action = random.choice(legal)
                    else:
                        # determine column from transformed ai_position diff
                        col_bit = transformed_state.ai_position ^ res.ai_position
                        action = (col_bit.bit_length() - 1) // 7
                    new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, action)
                    state = State(new_ai_pos, new_mask, state.depth + 1)
                elif opponent == 'alphabeta':
                    # Use alphabeta_search where the opponent is treated as 'ai' in transformed state
                    transformed_state = State(state.player_position, state.game_position, state.depth)
                    transformed_first = 0 if who_first == -1 else -1
                    res = alphabeta_search(transformed_state, transformed_first, d=depth_op)
                    if res is None:
                        # fallback random
                        legal = [c for c in range(7) if not (state.game_position & (1 << (7 * c + 5)))]
                        if not legal:
                            break
                        action = random.choice(legal)
                    else:
                        col_bit = transformed_state.ai_position ^ res.ai_position
                        action = (col_bit.bit_length() - 1) // 7
                    new_ai_pos, new_mask = make_move_opponent(state.ai_position, state.game_position, action)
                    state = State(new_ai_pos, new_mask, state.depth + 1)
                else:
                    raise ValueError('Unknown opponent type')

        # decay epsilon slowly
        agent.epsilon *= 0.9995

        if ep % max(1, episodes // 10) == 0 and ep >= 1:
            elapsed = time.time() - start
            print(f"Ep {ep}/{episodes}  wins={wins} losses={losses} draws={draws}  eps={agent.epsilon:.4f}  time={elapsed:.1f}s")

    # save q-table
    agent.save(save_path)
    print(f"Training finished. Q-table saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Q-learning agent')
    parser.add_argument('--episodes', '-n', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--opponent', '-o', choices=['random', 'minimax', 'alphabeta'], default='random', help='Opponent type')
    parser.add_argument('--depth-op', type=int, default=3, help='Depth for opponent search')
    parser.add_argument('--save', type=str, default='q_table.pkl', help='Path to save q-table')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Starting epsilon')
    args = parser.parse_args()

    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    train(agent, episodes=args.episodes, opponent=args.opponent, depth_op=args.depth_op, save_path=args.save)
