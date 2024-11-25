# type: ignore

import unittest

from TicTacToe.SymmetricMatrix import FullySymmetricMatrix, SymmetricMatrix


class TestSymmetricMatrix(unittest.TestCase):
    """Tests for the SymmetricMatrix class and its subclass TotallySymmetricMatrix."""

    def setUp(self) -> None:
        """Set up reusable objects for tests."""
        self.default_value = 0.0
        self.rows = 3
        self.matrix = SymmetricMatrix(default_value=self.default_value, rows=self.rows)
        self.totally_symmetric_matrix = FullySymmetricMatrix(default_value=self.default_value, rows=self.rows)

    def test_initialization(self) -> None:
        """Test initialization of SymmetricMatrix."""
        self.assertEqual(
            self.matrix.default_value, self.default_value, "Default value should be correctly initialized."
        )
        self.assertEqual(self.matrix.rows, self.rows, "Matrix size (rows) should be correctly initialized.")

    def test_canonical_board(self) -> None:
        """Validate that boards are transformed to canonical forms correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        canonical_board = self.matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

    def test_canonical_action(self) -> None:
        """Ensure actions are canonicalized consistently."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2  # Assume an action in the original board
        canonical_action = self.matrix.get_canonical_action(board, action)
        inverse_action = self.matrix.get_inverse_canonical_action(board, canonical_action)
        self.assertEqual(action, inverse_action, "Canonical action should map back to the original action.")

    def test_value_storage_and_retrieval(self) -> None:
        """Verify that values are stored and retrieved correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        value = 10.0

        self.matrix.set(board, action, value)
        retrieved_value = self.matrix.get(board, action)
        self.assertEqual(retrieved_value, value, "Stored value should match the retrieved value.")

        # Check symmetry handling
        symmetries = self.matrix._generate_symmetries(board)
        canonical_action = self.matrix.get_canonical_action(board, action)
        for symmetry in symmetries:
            symmetry_action = self.matrix.get_inverse_canonical_action(symmetry, canonical_action)
            self.assertEqual(
                self.matrix.get(symmetry, symmetry_action),
                value,
                "Value should be consistent across symmetric representations of the board.",
            )

    def test_empty_positions(self) -> None:
        """Ensure that empty positions on the board are identified correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        expected_empty_positions = [2, 3, 6, 7, 8]
        empty_positions = self.matrix.get_empty_positions(board)
        self.assertListEqual(
            empty_positions,
            expected_empty_positions,
            "Empty positions should match the expected positions.",
        )

    def test_totally_symmetric_matrix_behavior(self) -> None:
        """Ensure TotallySymmetricMatrix behaves as expected."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        value = 15.0

        self.totally_symmetric_matrix.set(board, action, value)
        retrieved_value = self.totally_symmetric_matrix.get(board, action)
        self.assertEqual(
            retrieved_value, value, "Stored value should match the retrieved value in TotallySymmetricMatrix."
        )

        # Check that symmetries in TotallySymmetricMatrix are handled as expected
        canonical_board = self.totally_symmetric_matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.totally_symmetric_matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent in TotallySymmetricMatrix.",
        )

    def test_totally_symmetric_matrix_behavior_2(self) -> None:
        """Ensure that different actions leading to equal next boards yield the same Q-value."""
        board_1 = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action_1 = 2
        # next_board_1 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_1 = 15.0

        self.totally_symmetric_matrix.set(board_1, action_1, value_1)

        board_2 = [" ", "O", "X", " ", "X", "O", " ", " ", " "]
        action_2 = 0
        # next_board_2 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_2 = self.totally_symmetric_matrix.get(board_2, action_2)
        self.assertEqual(value_1, value_2, "Different actions leading to equal next boards yield the same Q-value.")
