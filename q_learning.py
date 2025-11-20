import random
import pickle
from collections import defaultdict


class QLearningAgent:
    """Simple tabular Q-learning agent for Connect-Four bitboard states.

    States are stored as a tuple (ai_position, game_position). The agent always
    views the side it controls as the "ai" side when calling `choose_action`.
    """

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: maps state_key -> list of 7 Q-values (one per column)
        self.q_table = defaultdict(lambda: [0.0] * 7)

    def state_key(self, state):
        # store canonical state as tuple of ints
        return (int(state.ai_position), int(state.game_position))

    def available_actions(self, state):
        # return list of legal columns (0..6)
        return [c for c in range(7) if not (state.game_position & (1 << (7 * c + 5)))]

    def choose_action(self, state, who_went_first, epsilon=None):
        """Epsilon-greedy choice of column. The provided `state` must be arranged
        so that the agent's pieces are in `state.ai_position` (i.e. transform before
        calling if necessary).
        """
        if epsilon is None:
            epsilon = self.epsilon

        actions = self.available_actions(state)
        if not actions:
            return None

        key = self.state_key(state)

        # exploration
        if random.random() < epsilon:
            return random.choice(actions)

        # exploitation: pick argmax among legal actions
        qvals = self.q_table[key]
        best_val = None
        best_actions = []
        for a in actions:
            v = qvals[a]
            if best_val is None or v > best_val:
                best_val = v
                best_actions = [a]
            elif v == best_val:
                best_actions.append(a)
        return random.choice(best_actions)

    def update(self, prev_state, action, reward, next_state, done):
        """Update Q-table using Bellman equation."""
        if action is None:
            return
        prev_key = self.state_key(prev_state)
        next_key = self.state_key(next_state) if next_state is not None else None

        prev_q = self.q_table[prev_key][action]
        if done or next_key is None:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_key])

        self.q_table[prev_key][action] = prev_q + self.alpha * (target - prev_q)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: [0.0] * 7, data)


def train_self_play(agent, episodes=1000, opponent='random', alpha=None, epsilon_decay=0.999, verbose=100):
    """Placeholder: training helpers are provided in `headless_qlearning.py`.
    """
    raise NotImplementedError("Use the `headless_qlearning.py` script to train.")
