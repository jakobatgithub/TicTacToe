# type: ignore

import unittest
from unittest.mock import MagicMock

from TicTacToe.Agent import MouseAgent, RandomAgent
from TicTacToe.Display import ConsoleDisplay, TicTacToeDisplay
from TicTacToe.TicTacToe import TicTacToe


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
        self.assertIn([0, 1, 2], game.win_conditions)  # Horizontal win
        self.assertIn([0, 3, 6], game.win_conditions)  # Vertical win
        self.assertIn([0, 4, 8], game.win_conditions)  # Diagonal win

    def test_is_won(self) -> None:
        """Test win condition detection."""
        self.game.board = ["X", "X", "X", "O", " ", "O", " ", " ", " "]
        self.assertTrue(self.game.is_won("X"))
        self.assertFalse(self.game.is_won("O"))

    def test_is_draw(self) -> None:
        """Test draw detection."""
        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game.is_draw())

    def test_make_move(self) -> None:
        """Test making valid and invalid moves."""
        self.game.board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game.current_player = "O"
        self.assertTrue(self.game.make_move(2))  # Valid move
        self.assertFalse(self.game.make_move(0))  # Invalid move

    def test_switch_player(self) -> None:
        """Test player switching logic."""
        self.game.current_player = "X"
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "O")
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "X")

    def test_game_over(self) -> None:
        """Test game-over detection."""
        self.game.board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertTrue(self.game.is_game_over())  # Win detected

        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game.is_game_over())  # Draw detected

    def test_get_outcome(self) -> None:
        """Test outcome determination."""
        self.game.board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertEqual(self.game.get_outcome(), "X")

        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertEqual(self.game.get_outcome(), "D")

    def test_display_board(self) -> None:
        """Test board display updates."""
        board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game.display = self.display
        self.game.display_board(board)
        self.display.update_display.assert_called_with(board, None)

    def test_terminal_rewards(self) -> None:
        """Test terminal rewards calculation."""
        rewards = self.game.get_terminal_rewards("X")
        self.assertEqual(rewards, (1.0, -1.0))

        rewards = self.game.get_terminal_rewards("O")
        self.assertEqual(rewards, (-1.0, 1.0))

        rewards = self.game.get_terminal_rewards("D")
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

    def test_non_quadratic_board(self) -> None:
        """Test non-quadratic board raises an error."""
        with self.assertRaises(ValueError):
            TicTacToe(self.agent1, self.agent2, rows=3, cols=4)
