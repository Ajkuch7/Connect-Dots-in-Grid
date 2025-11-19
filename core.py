infinity = float('inf')


class State:
    """
    State class (copied from main.py, GUI removed)
    """

    status = 3

    def __init__(self, ai_position, game_position, depth=0):
        self.ai_position = ai_position
        self.game_position = game_position
        self.depth = depth

    @property
    def player_position(self):
        return self.ai_position ^ self.game_position

    @staticmethod
    def is_winning_state(position):
        # Horizontal check
        m = position & (position >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \
        m = position & (position >> 6)
        if m & (m >> 12):
            return True
        # Diagonal /
        m = position & (position >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = position & (position >> 1)
        if m & (m >> 2):
            return True
        # Nothing found
        return False

    @staticmethod
    def is_draw(position):
        return all(position & (1 << (7 * column + 5)) for column in range(0, 7))

    def terminal_node_test(self):
        """ Test if current state is a terminal node """
        if self.is_winning_state(self.ai_position):
            # AI Wins
            self.status = -1
            return True
        elif self.is_winning_state(self.player_position):
            # Player Wins
            self.status = 1
            return True
        elif self.is_draw(self.game_position):
            # Draw
            self.status = 0
            return True
        else:
            return False

    def calculate_heuristic(self):
        """
        Score based on who can win. Score computed as 22 minus number of moves played
        i.e. AI wins with 4th move, score = 22 - 4 = 18
        """
        if self.status == -1:
            # AI Wins
            return 22 - (self.depth // 2)
        elif self.status == 1:
            # Player Wins
            return -1 * (22 - (self.depth // 2))
        elif self.status == 0:
            # Draw
            return 0
        elif self.depth % 2 == 0:
            # MAX node returns
            return infinity
        else:
            # MIN node returns
            return -infinity

    def generate_children(self, who_went_first):
        """ For each column entry, generate a new State if the new position is valid"""
        for i in range(0, 7):
            # Select column starting from the middle and then to the edges index order [3,2,4,1,5,0,6]
            column = 3 + (1 - 2 * (i % 2)) * (i + 1) // 2
            if not self.game_position & (1 << (7 * column + 5)):
                if (who_went_first == -1 and self.depth % 2 == 0) or (who_went_first == 0 and self.depth % 2 == 1):
                    # AI (MAX) Move
                    new_ai_position, new_game_position = make_move(self.ai_position, self.game_position, column)
                else:
                    # Player (MIN) move
                    new_ai_position, new_game_position = make_move_opponent(self.ai_position, self.game_position,
                                                                            column)
                yield State(new_ai_position, new_game_position, self.depth + 1)

    def __str__(self):
        return '{0:049b}'.format(self.ai_position) + ' ; ' + '{0:049b}'.format(self.game_position)

    def __hash__(self):
        return hash((self.ai_position, self.game_position, self.depth % 2))

    def __eq__(self, other):
        return (self.ai_position, self.game_position, self.depth % 2) == (
            other.ai_position, other.game_position, other.depth % 2)


def make_move(position, mask, col):
    """ Helper method to make a move and return new position along with new board position """
    opponent_position = position ^ mask
    new_mask = mask | (mask + (1 << (col * 7)))
    return opponent_position ^ new_mask, new_mask


def make_move_opponent(position, mask, col):
    """ Helper method to only return new board position """
    new_mask = mask | (mask + (1 << (col * 7)))
    return position, new_mask


# Alphabeta search (copied from main.py)
def alphabeta_search(state, turn=-1, d=7):
    """Search game state to determine best action; use alpha-beta pruning. Returns best child state with .metrics attached."""
    # Track metrics
    search_metrics = {'nodes_explored': 0, 'nodes_pruned': 0}

    def max_value(state, alpha, beta, depth):
        search_metrics['nodes_explored'] += 1
        if cutoff_search(state, depth):
            return state.calculate_heuristic()

        v = -infinity
        for child in state.generate_children(turn):
            if child in seen:
                continue
            v = max(v, min_value(child, alpha, beta, depth + 1))
            seen[child] = alpha
            if v >= beta:
                search_metrics['nodes_pruned'] += 1
                return v
            alpha = max(alpha, v)
        if v == -infinity:
            return infinity
        return v

    def min_value(state, alpha, beta, depth):
        search_metrics['nodes_explored'] += 1
        if cutoff_search(state, depth):
            return state.calculate_heuristic()

        v = infinity
        for child in state.generate_children(turn):
            if child in seen:
                continue
            v = min(v, max_value(child, alpha, beta, depth + 1))
            seen[child] = alpha
            if v <= alpha:
                search_metrics['nodes_pruned'] += 1
                return v
            beta = min(beta, v)
        if v == infinity:
            return -infinity
        return v

    seen = {}
    cutoff_search = (lambda state, depth: depth > d or state.terminal_node_test())
    best_score = -infinity
    beta = infinity
    best_action = None
    for child in state.generate_children(turn):
        v = min_value(child, best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = child
    # Attach metrics to the result
    if best_action:
        best_action.metrics = search_metrics
    return best_action


# Plain minimax without alpha-beta pruning
def minimax_search(state, turn=-1, d=5):
    """Depth-limited minimax without alpha-beta pruning. Returns best child State with .metrics attached."""
    search_metrics = {'nodes_explored': 0, 'nodes_pruned': 0}

    def max_value(state, depth):
        search_metrics['nodes_explored'] += 1
        if cutoff_search(state, depth):
            return state.calculate_heuristic()
        v = -infinity
        for child in state.generate_children(turn):
            val = min_value(child, depth + 1)
            v = max(v, val)
        if v == -infinity:
            return infinity
        return v

    def min_value(state, depth):
        search_metrics['nodes_explored'] += 1
        if cutoff_search(state, depth):
            return state.calculate_heuristic()
        v = infinity
        for child in state.generate_children(turn):
            val = max_value(child, depth + 1)
            v = min(v, val)
        if v == infinity:
            return -infinity
        return v

    cutoff_search = (lambda state, depth: depth > d or state.terminal_node_test())
    best_score = -infinity
    best_action = None
    for child in state.generate_children(turn):
        v = min_value(child, 1)
        if v > best_score:
            best_score = v
            best_action = child
    # Attach metrics to result
    if best_action:
        best_action.metrics = search_metrics
    return best_action


def column_from_move(prev_ai_pos, new_ai_pos):
    col_bit = prev_ai_pos ^ new_ai_pos
    if col_bit == 0:
        return None
    column = (col_bit.bit_length() - 1) // 7
    return column
