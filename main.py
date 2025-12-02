# main.py

from fourInARowGUI import fourInARowGUI as GUI

# import search implementations from core (headless runner and experiments use core.py)
try:
    from core import minimax_search as core_minimax_search, alphabeta_search as core_alphabeta_search
except Exception:
    # fall back to functions defined in this file if core import not available
    core_minimax_search = None
    core_alphabeta_search = None

# Optional Q-learning agent
try:
    from q_learning import QLearningAgent
except Exception:
    QLearningAgent = None
infinity = float('inf')


class Game:
    AI = -1
    PLAYER = 0

    def __init__(self, game_board, ai_algo='alphabeta', player_is_ai=False):
        self.current_state = State(0, 0)
        self.turn = self.AI
        self.first = self.turn
        self.board = game_board
        # ai_algo: 'alphabeta' or 'plain'
        self.ai_algo = ai_algo
        # If True, the human player is controlled by an AI (AI vs AI mode)
        self.player_is_ai = player_is_ai
        # Q-learning agent instance (optional)
        self.q_agent = None
        if self.ai_algo == 'qlearning' and QLearningAgent is not None:
            try:
                self.q_agent = QLearningAgent()
                # Try to load a pre-trained table if present
                try:
                    self.q_agent.load('q_table.pkl')
                except Exception:
                    # no pre-trained table available — continue with default random Q
                    pass
            except Exception:
                self.q_agent = None

    def is_game_over(self):
        if self.has_winning_state():
            """Display who won"""
            print("AI Bot won!") if ~self.turn == self.AI else print("Congratulations, you won!")
            return True
        elif self.draw():
            print("Draw...Thank you...come again")
            return True
        return False

    def draw(self):
        """Check current state to determine if it is in a draw"""
        return State.is_draw(self.current_state.game_position) and not self.has_winning_state()

    def has_winning_state(self):
        return State.is_winning_state(self.current_state.ai_position) or State.is_winning_state(
            self.current_state.player_position)

    def next_turn(self):
        if self.turn == self.AI:
            self.query_AI()
        else:
            if self.player_is_ai:
                self.query_player_ai()
            else:
                self.query_player()

        # self.turn = (self.turn + 1) % 2
        # Apply one's complement (invert bits); 0 ~= -1
        self.turn = ~self.turn

    def query_player(self):
        """Make a move by querying standard input."""
        print("\nPlayer's Move...")
        column = None
        while column is None:
            # column = input('Your move identify column [0-6]? ')
            try:
                # column = int(column)
                column = GUI.getHumanInteraction(self.board)
                # Check if move is legal
                if not 0 <= column <= 6:
                    raise ValueError
                if self.current_state.game_position & (1 << (7 * column + 5)):
                    raise IndexError
            except (ValueError, IndexError):
                print("Invalid move. Try again...")
                column = None

        GUI.dropHumanToken(self.board, column)

        new_position, new_game_position = make_move(self.current_state.player_position,self.current_state.game_position, column)
        self.current_state = State(self.current_state.ai_position, new_game_position, self.current_state.depth + 1)

    def query_AI(self):
        """ AI Bot chooses next best move from current state """
        import time
        print("\nAI's Move...")
        temp_position = self.current_state.ai_position
        # If using Q-learning agent, use it to choose action
        if self.ai_algo == 'qlearning' and self.q_agent is not None:
            # deterministic play in GUI: epsilon=0
            column = self.q_agent.choose_action(self.current_state, self.first, epsilon=0.0)
            if column is None:
                # fallback: first legal move
                fallback_child = None
                for c in self.current_state.generate_children(self.first):
                    fallback_child = c
                    break
                if fallback_child is None:
                    return
                self.current_state = fallback_child
                column = (temp_position ^ self.current_state.ai_position).bit_length() - 1
                column = column // 7 if column >= 0 else None
            else:
                # apply chosen column
                new_ai_pos, new_mask = make_move(self.current_state.ai_position, self.current_state.game_position, column)
                self.current_state = State(new_ai_pos, new_mask, self.current_state.depth + 1)

            # No search metrics for Q-learning
            print(f"[Q-Learning] Chosen column: {column}")
            GUI.animateComputerMoving(self.board, column)
            GUI.makeMove(self.board, GUI.BLACK, column)
            return

        # choose search implementation depending on selected AI algorithm
        if self.ai_algo == 'plain':
            # use plain minimax (no alpha-beta). Use a smaller depth to avoid long delays.
            search_fn = core_minimax_search if core_minimax_search is not None else minimax_search
            depth = 5
        else:
            # default: alpha-beta minimax
            search_fn = core_alphabeta_search if core_alphabeta_search is not None else alphabeta_search
            depth = 7

        # Run search and measure time
        start_time = time.time()
        result = search_fn(self.current_state, self.first, d=depth)
        elapsed_time = time.time() - start_time

        # Extract metrics if available
        metrics = getattr(result, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0}) if result else {'nodes_explored': 0, 'nodes_pruned': 0}

        if result is None:
            # fallback: pick the first generated child (if any)
            fallback_child = None
            for c in self.current_state.generate_children(self.first):
                fallback_child = c
                break
            if fallback_child is None:
                # no legal moves (should be terminal) — nothing to do
                return
            self.current_state = fallback_child
        else:
            self.current_state = result

        # Get column for GUI (old ai position XOR new ai position)
        column = temp_position ^ self.current_state.ai_position
        column = (column.bit_length() - 1) // 7
        
        # Print AI metrics
        algo_name = "AlphaBeta" if self.ai_algo == 'alphabeta' else "Plain Minimax"
        print(f"[{algo_name}] Depth: {depth} | Nodes Explored: {metrics['nodes_explored']} | Nodes Pruned: {metrics['nodes_pruned']} | Time: {elapsed_time:.4f}s")

        GUI.animateComputerMoving(self.board, column)
        GUI.makeMove(self.board, GUI.BLACK, column)

    def query_player_ai(self):
        """Use AI to choose the player's (MIN) move in AI-vs-AI mode."""
        import time
        print("\nPlayer (AI) Move...")
        old_mask = self.current_state.game_position
        # If using a Q-learning agent for the player's side (AI-vs-AI), let it pick
        if self.ai_algo == 'qlearning' and self.q_agent is not None:
            # transformed state: treat player's pieces as 'ai' for the agent
            transformed_first = 0 if self.first == -1 else -1
            transformed_state = State(self.current_state.player_position, self.current_state.game_position,
                                      self.current_state.depth)
            column = self.q_agent.choose_action(transformed_state, transformed_first, epsilon=0.0)
            if column is None:
                # fallback to first legal
                fallback_child = None
                for c in self.current_state.generate_children(self.first):
                    fallback_child = c
                    break
                if fallback_child is None:
                    return
                self.current_state = fallback_child
            else:
                new_ai_pos, new_mask = make_move_opponent(self.current_state.ai_position, self.current_state.game_position, column)
                self.current_state = State(new_ai_pos, new_mask, self.current_state.depth + 1)

            # Determine which column changed by mask diff
            col_bits = old_mask ^ self.current_state.game_position
            column = (col_bits.bit_length() - 1) // 7
            print(f"[Player-QLearning] Chosen column: {column}")

            GUI.animateDroppingToken(self.board, column, GUI.RED)
            GUI.makeMove(self.board, GUI.RED, column)
            return

        # choose search implementation depending on selected AI algorithm
        if self.ai_algo == 'plain':
            search_fn = core_minimax_search if core_minimax_search is not None else minimax_search
            depth = 5
        else:
            search_fn = core_alphabeta_search if core_alphabeta_search is not None else alphabeta_search
            depth = 7

        # We'll reuse the same search by transforming the state so MAX==original player
        start_time = time.time()
        # transformed state: make the 'ai_position' be the original player positions
        transformed_first = 0 if self.first == -1 else -1
        transformed_state = State(self.current_state.player_position, self.current_state.game_position,
                                  self.current_state.depth)
        result_trans = search_fn(transformed_state, transformed_first, d=depth)
        elapsed_time = time.time() - start_time

        # Extract metrics if available
        metrics = getattr(result_trans, 'metrics', {'nodes_explored': 0, 'nodes_pruned': 0}) if result_trans else {'nodes_explored': 0, 'nodes_pruned': 0}

        if result_trans is None:
            # fallback: pick the first generated child (if any)
            fallback_child = None
            for c in self.current_state.generate_children(self.first):
                fallback_child = c
                break
            if fallback_child is None:
                return
            self.current_state = fallback_child
        else:
            # Map transformed result back to original coordinates
            new_game_position = result_trans.game_position
            new_ai_pos_trans = result_trans.ai_position
            # original ai_position after the player's move = new_game_position XOR transformed_ai_position
            new_ai_pos_orig = new_game_position ^ new_ai_pos_trans
            self.current_state = State(new_ai_pos_orig, new_game_position, result_trans.depth)
            # attach metrics
            self.current_state.metrics = metrics

        # Determine which column changed by mask diff
        col_bits = old_mask ^ self.current_state.game_position
        column = (col_bits.bit_length() - 1) // 7

        algo_name = "AlphaBeta" if self.ai_algo == 'alphabeta' else "Plain Minimax"
        print(f"[Player-{algo_name}] Depth: {depth} | Nodes Explored: {metrics['nodes_explored']} | Nodes Pruned: {metrics['nodes_pruned']} | Time: {elapsed_time:.4f}s")

        # Animate as player's (red) move
        GUI.animateDroppingToken(self.board, column, GUI.RED)
        GUI.makeMove(self.board, GUI.RED, column)


class State:
    """
    State class
    Each position is a 6x7 board with top row as sentinel row of 0's; so a 7x7 bitboard
    Bit positions corresponding to the board are as follows...
    -  -   -   -   -   -   -
    5  12  19  26  33  40  47
    4  11  18  25  32  39  46
    3  10  17  24  31  38  45
    2  9   16  23  30  37  44
    1  8   15  22  29  36  43
    0  7   14  21  28  35  42
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


def alphabeta_search(state, turn=-1, d=7):
    """Search game state to determine best action; use alpha-beta pruning. Returns best child state with .metrics attached."""
    # Track metrics
    search_metrics = {'nodes_explored': 0, 'nodes_pruned': 0}

    # Functions used by alpha beta
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
                # Min is going to completely ignore this route
                # since v will not get any lower than beta
                search_metrics['nodes_pruned'] += 1
                return v
            alpha = max(alpha, v)
        if v == -infinity:
            # If win/loss/draw not found, don't return -infinity to MIN node
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
                # Max is going to completely ignore this route
                # since v will not get any higher than alpha
                search_metrics['nodes_pruned'] += 1
                return v
            beta = min(beta, v)
        if v == infinity:
            # If win/loss/draw not found, don't return infinity to MAX node
            return -infinity
        return v

    # Keep track of seen states using their hash
    seen = {}

    # Body of alpha beta_search:
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


# Plain minimax without alpha-beta (fallback if core.minimax_search not available)
def minimax_search(state, turn=-1, d=5):
    """Depth-limited minimax without alpha-beta pruning. Returns best child State with .metrics attached."""
    search_metrics = {'nodes_explored': 0, 'nodes_pruned': 0}
    cutoff_search = (lambda state, depth: depth > d or state.terminal_node_test())

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


def make_move(position, mask, col):
    """ Helper method to make a move and return new position along with new board position """
    opponent_position = position ^ mask
    new_mask = mask | (mask + (1 << (col * 7)))
    return opponent_position ^ new_mask, new_mask


def make_move_opponent(position, mask, col):
    """ Helper method to only return new board position """
    new_mask = mask | (mask + (1 << (col * 7)))
    return position, new_mask


def print_board(state):
    """
    Helper method to pretty print binary board (6x7 board with top sentinel row of 0's)
    """
    ai_board, total_board = state.ai_position, state.game_position
    for row in range(5, -1, -1):
        print("")
        for column in range(0, 7):
            if ai_board & (1 << (7 * column + row)):
                print("1", end='')
            elif total_board & (1 << (7 * column + row)):
                print("2", end='')
            else:
                print("0", end='')
    print("")


if __name__ == "__main__":
    print("Welcome to Connect Four!")
    # Ask user which AI algorithm to use in the GUI: alpha-beta or plain minimax
    ai_choice = None
    try:
        choice = input("Choose AI algorithm: (1) AlphaBeta (default)  (2) Plain Minimax  (3) Q-Learning : ")
        c = choice.strip().lower()
        if c == '2':
            ai_choice = 'plain'
        elif c == '3' or c == 'q' or c == 'q-learning' or c == 'qlearning':
            ai_choice = 'qlearning'
        else:
            ai_choice = 'alphabeta'
    except Exception:
        ai_choice = 'alphabeta'

    # Ask whether to run AI vs AI or Human vs AI
    player_is_ai = False
    try:
        mode = input("Choose mode: (1) Human vs AI (default)  (2) AI vs AI : ")
        if mode.strip() == '2':
            player_is_ai = True
    except Exception:
        player_is_ai = False

    GUI.run()
    
    # Give user time to switch to the window and bring it to focus
    import time
    import pygame
    print("\nUI window initializing... (5 second delay to let you switch tabs)")
    for i in range(5, 0, -1):
        print(f"Starting in {i}s...", end='\r', flush=True)
        pygame.event.pump()  # Keep event queue responsive during delay
        time.sleep(1)
    print("Starting now!          ")
    
    while True:
        # Set up a blank board data structure.
        game_board = GUI.getNewBoard()
        GUI.drawBoard(game_board)
        GUI.updateDisplay()
        # Start game data structure
        game = Game(game_board, ai_algo=ai_choice, player_is_ai=player_is_ai)
        while not game.is_game_over():
            game.next_turn()
            print_board(game.current_state)
            GUI.drawBoard(game.board)
            GUI.updateDisplay()

        # Necessary for GUI
        WINNER = '' if game.draw() else GUI.COMPUTER if ~game.turn == -1 else GUI.HUMAN
        GUI.processGameOver(WINNER, game.board)
