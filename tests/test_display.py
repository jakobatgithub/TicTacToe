# type: ignore

import unittest
from typing import Literal
from unittest.mock import MagicMock, patch

from TicTacToe.Display import ConsoleDisplay, TicTacToeDisplay


class TestDisplay(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the common test environment."""
        self.rows = 3
        self.cols = 3
        self.board = ["X", "O", " ", " ", "X", "O", " ", " ", "X"]
        self.empty_board = [" " for _ in range(self.rows * self.cols)]
        self.outcome_x: Literal["X", "O", "D"] | None = "X"
        self.outcome_draw: Literal["X", "O", "D"] | None = "D"

    @patch("time.sleep", return_value=None)  # To skip delays during tests
    def test_console_display_outcome(self, _) -> None:
        """Test the ConsoleDisplay handles outcomes correctly."""
        display = ConsoleDisplay(rows=self.rows, cols=self.cols)
        with patch("builtins.print") as print_mock:
            display.update_display(self.board, outcome=self.outcome_x)
            print_mock.assert_any_call("Player X wins!")

            display.update_display(self.board, outcome=self.outcome_draw)
            print_mock.assert_any_call("It's a draw!")

    def test_console_display_message(self) -> None:
        """Test ConsoleDisplay message setting."""
        display = ConsoleDisplay(rows=self.rows, cols=self.cols)
        with patch("builtins.print") as print_mock:
            display.set_message("Player X's turn")
            print_mock.assert_called_with("Player X's turn")

    def test_tkinter_display_initialization(self) -> None:
        """Test TicTacToeDisplay initializes labels correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        self.assertEqual(len(display.labels), self.rows * self.cols)
        for label in display.labels:
            self.assertEqual(label.cget("text"), " ")

    def test_tkinter_display_update(self) -> None:
        """Test TicTacToeDisplay updates board correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        for label in display.labels:
            label.config = MagicMock()

        display.update_display(self.board)
        for i, label in enumerate(display.labels):
            expected_text = self.board[i] if self.board[i] in ["X", "O"] else " "
            label.config.assert_called_with(text=expected_text)

    def test_tkinter_display_outcome(self) -> None:
        """Test TicTacToeDisplay handles outcomes correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.set_message = MagicMock()

        display.update_display(self.board, outcome=self.outcome_x)
        display.set_message.assert_called_with("Player X wins!")

        display.update_display(self.board, outcome=self.outcome_draw)
        display.set_message.assert_called_with("It's a draw!")

    def test_tkinter_display_message(self) -> None:
        """Test TicTacToeDisplay sets messages correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.message_label.config = MagicMock()

        display.set_message("Player X's turn")
        display.message_label.config.assert_called_with(text="Player X's turn")

    def test_tkinter_wait_for_action(self) -> None:
        """Test TicTacToeDisplay waits for player action."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.wait_variable = MagicMock()

        display.wait_for_player_action()
        display.wait_variable.assert_called_with(display.action_complete)

    def test_tkinter_click_handler(self) -> None:
        """Test TicTacToeDisplay click handler functionality."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        mock_handler = MagicMock()
        display.bind_click_handler(mock_handler)

        event_mock = MagicMock()
        display.handle_click(event_mock, 5)
        mock_handler.assert_called_with(5)

    def test_tkinter_display_quit_on_outcome(self) -> None:
        """Test TicTacToeDisplay calls quit after an outcome."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.quit = MagicMock()

        display.update_display(self.board, outcome=self.outcome_x)
        display.quit.assert_called_once()
