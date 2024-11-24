import unittest
from unittest.mock import MagicMock, patch
from TicTacToe import TicTacToe
from Agent import RandomAgent, MouseAgent
from Display import TicTacToeDisplay, ConsoleDisplay


class TestTicTacToe(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the common test environment."""
        self.agent1 = RandomAgent(player="X")
        self.agent2 = RandomAgent(player="O")
        self.display = MagicMock(spec=TicTacToeDisplay)
        self.game = TicTacToe(self.agent1, self.agent2)

    def test_initialize_board(self) -> None:
        """Test board initialization."""
        board = TicTacToe.initialize_board(3, 3)
        self.assertEqual(len(board), 9)
        self.assertTrue(all(cell == " " for cell in board))

    def test_valid_actions(self) -> None:
        """Test valid actions are calculated correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        valid_actions = TicTacToe.get_valid_actions_from_board(board)
        self.assertEqual(valid_actions, [2, 3, 6, 7, 8])

    def test_win_conditions(self) -> None:
        """Test win condition generation."""
        game = TicTacToe(self.agent1, self.agent2, rows=3, cols=3, win_length=3)
        self.assertIn([0, 1, 2], game._win_conditions)  # Horizontal win
        self.assertIn([0, 3, 6], game._win_conditions)  # Vertical win
        self.assertIn([0, 4, 8], game._win_conditions)  # Diagonal win

    def test_is_won(self) -> None:
        """Test win condition detection."""
        self.game._board = ["X", "X", "X", "O", " ", "O", " ", " ", " "]
        self.assertTrue(self.game._is_won("X"))
        self.assertFalse(self.game._is_won("O"))

    def test_is_draw(self) -> None:
        """Test draw detection."""
        self.game._board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game._is_draw())

    def test_make_move(self) -> None:
        """Test making valid and invalid moves."""
        self.game._board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game._current_player = "O"
        self.assertTrue(self.game._make_move(2))  # Valid move
        self.assertFalse(self.game._make_move(0))  # Invalid move

    def test_switch_player(self) -> None:
        """Test player switching logic."""
        self.game._current_player = "X"
        self.game._switch_player()
        self.assertEqual(self.game._current_player, "O")
        self.game._switch_player()
        self.assertEqual(self.game._current_player, "X")

    def test_game_over(self) -> None:
        """Test game-over detection."""
        self.game._board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertTrue(self.game._is_game_over())  # Win detected

        self.game._board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game._is_game_over())  # Draw detected

    def test_get_outcome(self) -> None:
        """Test outcome determination."""
        self.game._board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertEqual(self.game._get_outcome(), "X")

        self.game._board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertEqual(self.game._get_outcome(), "D")

    def test_display_board(self) -> None:
        """Test board display updates."""
        board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game.display = self.display
        self.game.display_board(board)
        self.display.update_display.assert_called_with(board, None)

    def test_terminal_rewards(self) -> None:
        """Test terminal rewards calculation."""
        rewards = self.game._get_terminal_rewards("X")
        self.assertEqual(rewards, (1.0, -1.0))

        rewards = self.game._get_terminal_rewards("O")
        self.assertEqual(rewards, (-1.0, 1.0))

        rewards = self.game._get_terminal_rewards("D")
        self.assertEqual(rewards, (0.0, 0.0))

    def test_agent_interaction(self) -> None:
        """Test interaction with agents."""
        self.agent1.get_action = MagicMock(return_value=0)
        self.agent2.get_action = MagicMock(return_value=1)

        outcome = self.game.play()
        self.agent1.get_action.assert_called()
        self.agent2.get_action.assert_called()
        self.assertIn(outcome, ["X", "O", "D"])

    def test_mouse_agent_requires_display(self) -> None:
        """Test MouseAgent raises an error without a display."""
        agent1 = MouseAgent(player="X")
        display = ConsoleDisplay(rows=3, cols=3, waiting_time=0.25)
        with self.assertRaises(ValueError):
            TicTacToe(agent1, self.agent2, display=display)


class TestDisplay(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the common test environment."""
        self.rows = 3
        self.cols = 3
        self.board = ["X", "O", " ", " ", "X", "O", " ", " ", "X"]
        self.empty_board = [" " for _ in range(self.rows * self.cols)]
        self.outcome_x = "X"
        self.outcome_draw = "D"

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


if __name__ == "__main__":
    unittest.main()
