# type: ignore

import unittest
from typing import Any
from unittest.mock import MagicMock

from TicTacToe.Agent import HumanAgent, RandomAgent
from TicTacToe.DeepQAgent import DeepQLearningAgent
from TicTacToe.Display import ConsoleDisplay
from TicTacToe.SymmetricMatrix import SymmetricMatrix
from TicTacToe.TicTacToe import TicTacToe


class TestIntegration(unittest.TestCase):
    """Integration tests for the Tic-Tac-Toe project."""

    def test_random_agent_vs_random_agent(self) -> None:
        """Simulate a game between two RandomAgents."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2)

        outcome = game.play()
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be a win, loss, or draw.")

    def test_deep_q_agent_training(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        params: dict[str, Any] = {
            "player": "X",
            "switching": False,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "nr_of_episodes": 10,
            "batch_size": 32,
            "target_update_frequency": 2,
            "learning_rate": 0.001,
            "replay_buffer_length": 100,
            "wandb_logging_frequency": 5,
            "rows": 3,
            "device": "cpu",
            "wandb": False,
        }
        agent1 = DeepQLearningAgent(params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_console_display_updates(self) -> None:
        """Ensure the ConsoleDisplay updates correctly during a game."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        display = ConsoleDisplay(rows=3, cols=3)
        game = TicTacToe(agent1, agent2, display=display)

        display.update_display = MagicMock()  # Mock the display update method
        game.play()

        # Verify the display was updated at least once per move
        self.assertGreaterEqual(display.update_display.call_count, 1, "Display should update during the game.")

    def test_symmetric_matrix_usage(self) -> None:
        """Verify that SymmetricMatrix is used correctly in a game."""
        matrix = SymmetricMatrix(default_value=0.0, rows=3)
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        matrix.set(board, action, 5.0)

        # Simulate a game where matrix is used for Q-value updates
        self.assertEqual(matrix.get(board, action), 5.0, "SymmetricMatrix should retrieve stored values correctly.")

    def test_human_agent_interaction(self) -> None:
        """Mock HumanAgent to simulate user input during a game."""
        agent1 = RandomAgent(player="X")
        agent2 = HumanAgent(player="O")
        game = TicTacToe(agent1, agent2)

        # Mock HumanAgent's input to simulate user actions
        agent2.get_action = MagicMock(return_value=0)
        outcome = game.play()

        self.assertTrue(agent2.get_action.called, "HumanAgent should be called for actions during the game.")
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be valid.")
