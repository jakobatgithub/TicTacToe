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

    def setUp(self) -> None:
        self.params: dict[str, Any] = {
            # Game settings
            "player": "X",  # Player symbol for the agent
            "rows": 3,  # Board size (rows x rows)
            "win_length": 3,  # Number of in-a-row needed to win
            "rewards": {
                "W": 1.0,  # Reward for a win
                "L": -1.0,  # Reward for a loss
                "D": 0.5,  # Reward for a draw
            },

            # Training settings
            "nr_of_episodes": 15000,  # Number of training games
            "learning_rate": 0.0001,  # Optimizer learning rate
            "gamma": 0.95,  # Discount factor for future rewards
            "switching": True,  # Whether players switch turns
            "target_update_frequency": 25,  # Frequency to sync target network

            # Evaluation settings
            "evaluation_frequency": 100,  # Episodes between evaluations
            "evaluation_batch_size": 300,  # Games to evaluate per round
            "wandb_logging": False,  # Enable Weights & Biases logging
            "wandb_logging_frequency": 25,  # Logging frequency (in episodes)

            # Exploration rate settings
            "epsilon_start": 0.925,  # Initial exploration rate
            "epsilon_min": 0.01,  # Minimum exploration rate
            "set_exploration_rate_externally": True,  # Adaptive epsilon enabled
            "epsilon_update_threshold": 0.025,  # Epsilon adjustment sensitivity
            "epsilon_decay": 0.95,  # Decay rate for epsilon
            "win_rate_deque_length": 5,  # Length of win rate deques

            # Device settings
            "device": "cpu",  # Device: "cuda", "mps", or "cpu"

            # Replay buffer settings
            "replay_buffer_type": "prioritized",  # "uniform" or "prioritized"
            "replay_buffer_length": 10000,  # Max length of replay buffer
            "batch_size": 256,  # Batch size for training updates
            "priority_alpha": 0.6,
            "priority_beta": 0.4,
            "shared_replay_buffer": False,  # Share replay buffer between agents

            # Q Network settings
            "network_type": "FullyCNN",  # Network architecture: 'Equivariant', 'FullyCNN', 'FCN', 'CNN'
            "periodic": False,  # Periodic boundary conditions
            "load_network": False,  # Whether to load pretrained weights
            "save_models": "/Users/jakob/TicTacToe/models/",  # Save weights after training
            "symmetrized_loss": False,  # Use symmetrized loss
            "state_shape": "one-hot",  # state representation: 'flat' with shape (batch_size, rows * rows), 
                                    # '2D' with shape (batch_size, 1, rows, rows), 
                                    # 'one-hot' with shape (batch_size, 3, rows, rows)
        }


    def test_random_agent_vs_random_agent(self) -> None:
        """Simulate a game between two RandomAgents."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        outcome = game.play()
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be a win, loss, or draw.")

    def test_deep_q_agent_training_FCN(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FCN"
        self.params["periodic"] = False
        self.params["state_shape"] = "flat"
        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_FCN_more_rows(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FCN"
        self.params["periodic"] = False
        self.params["state_shape"] = "flat"
        self.params["rows"] = 5
        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_CNN(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "CNN"
        self.params["periodic"] = False
        self.params["state_shape"] = "flat"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_CNN_2D_state(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "CNN"
        self.params["periodic"] = False
        self.params["state_shape"] = "2D"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_CNN_one_hot_state(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "CNN"
        self.params["periodic"] = False
        self.params["state_shape"] = "one-hot"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_CNN_more_rows(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "CNN"
        self.params["periodic"] = False
        self.params["state_shape"] = "flat"
        self.params["rows"] = 5

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_CNN_periodic(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "CNN"
        self.params["periodic"] = True
        self.params["state_shape"] = "flat"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)


        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_FullyCNN(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FullyCNN"
        self.params["periodic"] = False
        self.params["state_shape"] = "flat"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_FullyCNN_periodic(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FullyCNN"
        self.params["periodic"] = True
        self.params["state_shape"] = "flat"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_FullyCNN_periodic_2D_state(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FullyCNN"
        self.params["periodic"] = True
        self.params["state_shape"] = "2D"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_deep_q_agent_training_FullyCNN_periodic_one_hot_state(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        self.params["network_type"] = "FullyCNN"
        self.params["periodic"] = True
        self.params["state_shape"] = "one-hot"
        self.params["rows"] = 3

        agent1 = DeepQLearningAgent(params=self.params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2, params=self.params)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_console_display_updates(self) -> None:
        """Ensure the ConsoleDisplay updates correctly during a game."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        display = ConsoleDisplay(rows=self.params["rows"], cols=self.params["rows"])
        game = TicTacToe(agent1, agent2, display=display, params=self.params)

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
        game = TicTacToe(agent1, agent2, params=self.params)

        # Mock HumanAgent's input to simulate user actions
        agent2.get_action = MagicMock(return_value=0)
        outcome = game.play()

        self.assertTrue(agent2.get_action.called, "HumanAgent should be called for actions during the game.")
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be valid.")
