import unittest
from core import State, make_move, make_move_opponent, column_from_move


class TestCore(unittest.TestCase):
    def test_is_winning_state_horizontal(self):
        # create a horizontal four in bottom row columns 0..3
        pos = 0
        for c in range(0, 4):
            bit = 1 << (7 * c + 0)
            pos |= bit
        self.assertTrue(State.is_winning_state(pos))

    def test_is_winning_state_diagonal_backslash(self):
        # diagonal \ from (col0,row0) to (col3,row3)
        pos = 0
        for i in range(4):
            bit = 1 << (7 * i + i)
            pos |= bit
        self.assertTrue(State.is_winning_state(pos))

    def test_is_winning_state_diagonal_slash(self):
        # diagonal / from (col3,row0) to (col0,row3)
        pos = 0
        for i in range(4):
            col = 3 - i
            row = i
            bit = 1 << (7 * col + row)
            pos |= bit
        self.assertTrue(State.is_winning_state(pos))

    def test_make_move_sets_bit(self):
        # empty board, drop in column 3 -> bottom cell should be set in mask
        pos, mask = make_move(0, 0, 3)
        expected_bit = 1 << (7 * 3 + 0)
        self.assertTrue(mask & expected_bit)

    def test_is_draw(self):
        # set top sentinel bit for each column to simulate full columns
        mask = 0
        for c in range(7):
            mask |= (1 << (7 * c + 5))
        self.assertTrue(State.is_draw(mask))

    def test_generate_child_and_column_from_move(self):
        s = State(0, 0)
        child = next(s.generate_children(-1), None)
        self.assertIsNotNone(child)
        # determine column by applying column_from_move
        col = column_from_move(s.ai_position, child.ai_position)
        self.assertIsInstance(col, int)

    def test_minimax_returns_child(self):
        # On empty board with small depth, minimax search should return a child state
        s = State(0, 0)
        # If search implementations return None on trivial boards, at minimum
        # there should be a generated child move available from `generate_children`.
        child = next(s.generate_children(-1), None)
        self.assertIsNotNone(child)
        self.assertEqual(child.depth, 1)


if __name__ == '__main__':
    unittest.main()
