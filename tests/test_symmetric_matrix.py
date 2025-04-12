# type: ignore

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from TicTacToe.game_types import Action, Board
from TicTacToe.SymmetricMatrix import BaseMatrix, FullySymmetricMatrix, Matrix, SymmetricMatrix


class TestSymmetricMatrix(unittest.TestCase):
    """Tests for the SymmetricMatrix class and its subclass TotallySymmetricMatrix."""

    def setUp(self) -> None:
        """Set up reusable objects for tests."""
        self.default_value = 0.0
        self.rows = 3
        self.matrix = Matrix(default_value=self.default_value)
        self.symmetric_matrix = SymmetricMatrix(default_value=self.default_value, rows=self.rows)
        self.fully_symmetric_matrix = FullySymmetricMatrix(default_value=self.default_value, rows=self.rows)

        self.symmetric_matrix_non_lazy = SymmetricMatrix(default_value=self.default_value, rows=self.rows, lazy=False)
        self.fully_symmetric_matrix_non_lazy = FullySymmetricMatrix(
            default_value=self.default_value, rows=self.rows, lazy=False
        )

    def test_initialization(self) -> None:
        """Test initialization of SymmetricMatrix."""
        self.assertEqual(
            self.matrix.default_value, self.default_value, "Default value should be correctly initialized."
        )
        self.assertEqual(
            self.symmetric_matrix.default_value, self.default_value, "Default value should be correctly initialized."
        )
        self.assertEqual(
            self.fully_symmetric_matrix.default_value,
            self.default_value,
            "Default value should be correctly initialized.",
        )
        self.assertEqual(
            self.symmetric_matrix_non_lazy.default_value,
            self.default_value,
            "Default value should be correctly initialized.",
        )
        self.assertEqual(
            self.fully_symmetric_matrix_non_lazy.default_value,
            self.default_value,
            "Default value should be correctly initialized.",
        )

        self.assertEqual(self.symmetric_matrix.rows, self.rows, "Matrix size (rows) should be correctly initialized.")
        self.assertEqual(
            self.fully_symmetric_matrix.rows, self.rows, "Matrix size (rows) should be correctly initialized."
        )
        self.assertEqual(
            self.symmetric_matrix_non_lazy.rows, self.rows, "Matrix size (rows) should be correctly initialized."
        )
        self.assertEqual(
            self.fully_symmetric_matrix_non_lazy.rows, self.rows, "Matrix size (rows) should be correctly initialized."
        )

        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2  # Assume an action in the original board
        self.assertEqual(
            self.matrix.get(tuple(board), action),
            self.default_value,
            "All values should be initialized to the default value.",
        )
        self.assertEqual(
            self.symmetric_matrix.get(board, action),
            self.default_value,
            "All values should be initialized to the default value.",
        )
        self.assertEqual(
            self.fully_symmetric_matrix.get(board, action),
            self.default_value,
            "All values should be initialized to the default value.",
        )
        self.assertEqual(
            self.symmetric_matrix_non_lazy.get(board, action),
            self.default_value,
            "All values should be initialized to the default value.",
        )
        self.assertEqual(
            self.fully_symmetric_matrix_non_lazy.get(board, action),
            self.default_value,
            "All values should be initialized to the default value.",
        )

    def test_canonical_board(self) -> None:
        """Validate that boards are transformed to canonical forms correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        canonical_board = self.symmetric_matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.symmetric_matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

        canonical_board = self.fully_symmetric_matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.fully_symmetric_matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

        canonical_board = self.symmetric_matrix_non_lazy.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.symmetric_matrix_non_lazy.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

        canonical_board = self.fully_symmetric_matrix_non_lazy.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.fully_symmetric_matrix_non_lazy.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

    def test_canonical_action(self) -> None:
        """Ensure actions are canonicalized consistently."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2  # Assume an action in the original board

        canonical_action = self.symmetric_matrix.get_canonical_action(board, action)
        inverse_action = self.symmetric_matrix.get_inverse_canonical_action(board, canonical_action)
        self.assertEqual(action, inverse_action, "Canonical action should map back to the original action.")

    def test_value_storage_and_retrieval(self) -> None:
        """Verify that values are stored and retrieved correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        value = 10.0

        self.symmetric_matrix.set(board, action, value)
        retrieved_value = self.symmetric_matrix.get(board, action)
        self.assertEqual(retrieved_value, value, "Stored value should match the retrieved value.")

        # Check symmetry handling
        symmetries = self.symmetric_matrix._generate_symmetries(board)
        canonical_action = self.symmetric_matrix.get_canonical_action(board, action)
        for symmetry in symmetries:
            symmetry_action = self.symmetric_matrix.get_inverse_canonical_action(symmetry, canonical_action)
            self.assertEqual(
                self.symmetric_matrix.get(symmetry, symmetry_action),
                value,
                "Value should be consistent across symmetric representations of the board.",
            )

    def test_empty_positions(self) -> None:
        """Ensure that empty positions on the board are identified correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        expected_empty_positions = [2, 3, 6, 7, 8]
        empty_positions = self.symmetric_matrix.get_empty_positions(board)
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

        self.fully_symmetric_matrix.set(board, action, value)
        retrieved_value = self.fully_symmetric_matrix.get(board, action)
        self.assertEqual(
            retrieved_value, value, "Stored value should match the retrieved value in TotallySymmetricMatrix."
        )

        # Check that symmetries in TotallySymmetricMatrix are handled as expected
        canonical_board = self.fully_symmetric_matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.fully_symmetric_matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent in TotallySymmetricMatrix.",
        )

    def test_totally_symmetric_matrix_behavior_2(self) -> None:
        """Ensure that different actions leading to equal next boards yield the same Q-value."""
        board_1 = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action_1 = 2
        # next_board_1 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_1 = 15.0

        self.fully_symmetric_matrix.set(board_1, action_1, value_1)

        board_2 = [" ", "O", "X", " ", "X", "O", " ", " ", " "]
        action_2 = 0
        # next_board_2 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_2 = self.fully_symmetric_matrix.get(board_2, action_2)
        self.assertEqual(value_1, value_2, "Different actions leading to equal next boards yield the same Q-value.")


class MatrixTestHelper(BaseMatrix):
    def get(self, board: Board, action: Action) -> float:
        return self.qMatrix[board][action]

    def set(self, board: Board, action: Action, value: float) -> None:
        self.qMatrix[board][action] = value


# Test suite
class TestBaseMatrix(unittest.TestCase):
    def setUp(self):
        self.default_value = 0.5
        self.matrix = MatrixTestHelper(default_value=self.default_value)

    def test_initialization_without_file(self):
        """Test initialization without a file."""
        self.assertIsInstance(self.matrix.qMatrix, defaultdict)
        self.assertEqual(self.matrix.default_value, self.default_value)

    @patch("builtins.open", new_callable=MagicMock)
    @patch("dill.load")
    def test_initialization_with_file(self, mock_dill_load, mock_open):
        """Test initialization with a file."""
        mock_dill_load.return_value = {"mock_key": "mock_value"}
        matrix = MatrixTestHelper(file="dummy_path")
        mock_open.assert_called_once_with("dummy_path", "rb")
        self.assertEqual(matrix.qMatrix, {"mock_key": "mock_value"})

    def test_initialize_q_matrix(self):
        """Test that _initialize_q_matrix returns the correct default structure."""
        initial_matrix = self.matrix._initialize_q_matrix()
        self.assertIsInstance(initial_matrix, defaultdict)
        self.assertEqual(initial_matrix["nonexistent_key"], self.default_value)

    def test_get_and_set(self):
        """Test get and set methods through the concrete subclass."""
        board = (" ") * 9
        action = 6
        value = 1.23

        # Initially, the value should be the default
        self.assertEqual(self.matrix.get(board, action), self.default_value)

        # After setting, the value should be updated
        self.matrix.set(board, action, value)
        self.assertEqual(self.matrix.get(board, action), value)
