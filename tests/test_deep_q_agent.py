# type: ignore

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from TicTacToe.DeepQAgent import DeepQLearningAgent, DeepQPlayingAgent, ReplayBuffer


class TestDeepQLearningAgent(unittest.TestCase):
    """Tests for the DeepQLearningAgent class."""

    def setUp(self) -> None:
        """Set up common parameters and objects for the tests."""
        self.params: dict[str, Any] = {
            "player": "X",
            "switching": False,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "nr_of_episodes": 100,
            "batch_size": 32,
            "target_update_frequency": 10,
            "learning_rate": 0.001,
            "replay_buffer_length": 100,
            "wandb_logging_frequency": 5,
            "rows": 3,
            "device": "cpu",
            "wandb": False,
            "load_network": False,
            "shared_replay_buffer": False,
            "network_type": "FCN",
            "set_exploration_rate_externally": False,
            "state_shape": "flat",        
        }
        self.agent = DeepQLearningAgent(self.params)

    def test_initialization(self) -> None:
        """Check initialization of networks, replay buffer, and parameters."""
        self.assertIsInstance(self.agent.q_network, nn.Module, "Q-network should be an instance of nn.Module.")
        self.assertIsInstance(
            self.agent.target_network, nn.Module, "Target network should be an instance of nn.Module."
        )
        self.assertIsInstance(
            self.agent.replay_buffer, ReplayBuffer, "Replay buffer should be an instance of ReplayBuffer."
        )
        self.assertEqual(self.agent.epsilon, self.params["epsilon_start"], "Epsilon should match the starting value.")
        self.assertEqual(self.agent.gamma, self.params["gamma"], "Gamma should be correctly initialized.")

    def test_initialization_parameters(self):
        """Test initialization of DeepQLearningAgent parameters."""
        self.assertEqual(self.agent.gamma, self.params["gamma"], "Gamma should be initialized correctly.")
        self.assertEqual(self.agent.epsilon, self.params["epsilon_start"], "Epsilon should match the starting value.")
        self.assertEqual(self.agent.nr_of_episodes, self.params["nr_of_episodes"], "Number of episodes should match.")
        self.assertEqual(self.agent.batch_size, self.params["batch_size"], "Batch size should match.")
        self.assertEqual(self.agent.target_update_frequency, self.params["target_update_frequency"], "Target update frequency should match.")
        self.assertEqual(self.agent.learning_rate, self.params["learning_rate"], "Learning rate should match.")
        self.assertEqual(self.agent.replay_buffer_length, self.params["replay_buffer_length"], "Replay buffer length should match.")
        self.assertEqual(self.agent.rows, self.params["rows"], "Number of rows should match.")
        self.assertEqual(self.agent.device.type, self.params["device"], "Device should match.")
        self.assertIsInstance(self.agent.q_network, nn.Module, "Q-network should be an instance of nn.Module.")
        self.assertIsInstance(self.agent.target_network, nn.Module, "Target network should be an instance of nn.Module.")
        self.assertIsInstance(self.agent.replay_buffer, ReplayBuffer, "Replay buffer should be an instance of ReplayBuffer.")
        self.assertIsNotNone(self.agent.state_converter, "State converter should be initialized.")

    def test_action_selection_exploration(self) -> None:
        """Test action selection in exploration mode (random)."""
        self.agent.epsilon = 1.0  # Force exploration
        board = [" " for _ in range(self.params["rows"] ** 2)]
        action = self.agent.choose_action(board, epsilon=self.agent.epsilon)
        self.assertIn(action, self.agent.get_valid_actions(board), "Action should be valid for the given board.")

    def test_action_selection_exploitation(self) -> None:
        """Test action selection in exploitation mode (greedy)."""
        self.agent.epsilon = 0.0  # Force exploitation
        board = [" " for _ in range(self.params["rows"] ** 2)]
        # Mock the Q-network output to test greedy selection
        self.agent.q_network = MagicMock()
        self.agent.q_network.return_value = torch.tensor([0.1, 0.9, -0.2] + [-float("inf")] * 6)  # Mock 3 valid actions
        action = self.agent.choose_action(board, epsilon=self.agent.epsilon)
        self.assertEqual(action, 1, "Agent should select the action with the highest Q-value.")

    def test_add_experience_to_replay_buffer(self) -> None:
        """Ensure experiences are added to the replay buffer correctly."""
        board = [" " for _ in range(self.params["rows"] ** 2)]
        next_board = board[:]
        next_board[0] = "X"
        self.agent.episode_history = [(board, 0)]  # Simulate a previous move
        self.agent.get_action((next_board, 1.0, False), game=None)  # Add experience
        self.assertEqual(len(self.agent.replay_buffer), 1, "Replay buffer should contain one experience.")

    def test_training_step_updates_network(self) -> None:
        """Validate that the training step updates the Q-network weights."""
        # Fill the episode history
        board = [" " for _ in range(self.params["rows"] ** 2)]
        next_board = board[:]
        next_board[0] = "X"
        self.agent.episode_history = [(board, 0)]  # Simulate a previous move
        # Fill the replay buffer
        for _ in range(self.agent.batch_size):
            self.agent.replay_buffer.add(
                state=np.random.rand(self.params["rows"] ** 2),
                action=0,
                reward=1.0,
                next_state=np.random.rand(self.params["rows"] ** 2),
                done=False,
            )

        initial_weights = [param.clone().detach() for param in self.agent.q_network.parameters()]
        self.agent.get_action((None, 0, True), None)  # Trigger training
        updated_weights = [param.clone().detach() for param in self.agent.q_network.parameters()]

        # Compare initial and updated weights
        for initial, updated in zip(initial_weights, updated_weights):
            assert not torch.equal(initial, updated), "Training step did not update the Q-network weights."

    def test_target_network_update(self) -> None:
        """Check that the target network updates at specified intervals."""
        # Mock Q-network to track its state_dict updates
        self.agent.q_network.load_state_dict = MagicMock()
        self.agent.target_network.load_state_dict = MagicMock()

        for i in range(1, 21):  # Simulate 20 episodes
            self.agent.episode_count = i
            if i % self.agent.target_update_frequency == 0:
                self.agent.get_action((None, 0, True), None)  # Simulate an action
                self.agent.target_network.load_state_dict.assert_called()
            else:
                self.agent.target_network.load_state_dict.assert_not_called()
            self.agent.target_network.load_state_dict.reset_mock()

    def test_update_state_transitions_and_replay_buffer(self):
        # Mock inputs
        reward = 1
        done = False
        board = ["X", "O", " ", "O", "X", " ", " ", " ", " "]
        action = 2
        next_board = ["X", "O", "X", "O", "X", " ", " ", " ", " "]
        self.agent.episode_history = [(board, action)]

        # Create fake state arrays (matching shape of board_to_state)
        fake_state = np.ones((1, 9), dtype=np.float32)        # e.g., from board
        fake_next_state = np.full((1, 9), 2.0, dtype=np.float32)  # e.g., from next_board

        # Mock the board_to_state method
        self.agent.board_to_state = MagicMock(side_effect=[fake_state, fake_next_state])

        # Call the method
        self.agent._update_state_transitions_and_replay_buffer(next_board, reward, done)

        # Assertions
        self.agent.board_to_state.assert_called()
        self.assertEqual(len(self.agent.state_transitions), 1)

    def test_handle_incomplete_game(self):
        # Mock inputs
        next_board = ["X", "O", " ", "O", "X", " ", " ", " ", " "]
        self.agent.choose_action = MagicMock(return_value=4)

        # Call the method
        action = self.agent._handle_incomplete_game(next_board)

        # Assertions
        self.agent.choose_action.assert_called_with(next_board, epsilon=self.agent.epsilon)
        self.assertEqual(action, 4)
        self.assertEqual(len(self.agent.episode_history), 1)
        self.assertEqual(self.agent.games_moves_count, 1)

    def test_handle_game_completion(self):
        # Mock attributes
        self.agent.episode_count = 5
        self.agent.target_update_frequency = 5
        self.agent.update_exploration_rate = MagicMock()

        # Call the method
        self.agent._handle_game_completion()

        # Assertions
        self.agent.update_exploration_rate.assert_called_with(6)  # Incremented episode count
        self.assertEqual(len(self.agent.episode_history), 0)

    def test_state_to_board(self):
        mock_state = MagicMock()
        mock_state.flatten.return_value = [0, 1, -1, 1, 0]
        expected_board = [" ", "X", "O", "X", " "]
        result = self.agent.state_to_board(mock_state)
        self.assertEqual(result, expected_board)

    def test_compute_loss(self):
        size = 8
        state_dim = 9
        # Define mock behaviors for q_network and target_network
        self.agent.q_network = MagicMock()
        self.agent.target_network = MagicMock()

        q_network_return_value = torch.zeros((size, state_dim))
        q_network_return_value[3, 1] = 1
        self.agent.q_network.return_value = q_network_return_value
        target_network_return_value = torch.zeros((size, state_dim))
        target_network_return_value[1, 5] = 2
        self.agent.target_network.return_value = target_network_return_value

        states = torch.zeros((size, state_dim), dtype=torch.float32)
        actions = torch.zeros(size, dtype=torch.int64)
        actions[3] = 1
        rewards = torch.zeros(size, dtype=torch.float32)
        rewards[4] = 1
        next_states = torch.zeros((size, state_dim), dtype=torch.float32)
        dones = torch.zeros(size, dtype=torch.bool)

        samples = (states, actions, rewards, next_states, dones)

        # Expected outputs
        expected_q_values = torch.zeros(size, dtype=torch.float32)
        expected_q_values[3] = 1
        expected_next_q_values = torch.zeros(size, dtype=torch.float32)
        expected_next_q_values[1] = 2
        expected_targets = rewards + (~dones) * self.agent.gamma * expected_next_q_values

        # Compute loss using the mocked function
        loss = self.agent.compute_standard_loss(samples)

        # Assert correct computations
        self.assertAlmostEqual(
            loss.item(), nn.MSELoss()(expected_q_values, expected_targets).item() / self.agent.batch_size, places=5
        )

    @patch("torch.randint", wraps=torch.randint)
    def test_single_max_q_value(self, mock_randint):
        # Mock the QNetwork output
        mock_qnet = MagicMock()
        mock_qnet.return_value = torch.tensor([[1.0, 2.0, 3.0, 5.0]])

        self.agent.board_to_state = MagicMock()
        # Call the method
        result = self.agent.get_best_action("mock_board", mock_qnet)

        # Assertions
        self.assertEqual(result, 3)  # Index of the maximum Q-value
        mock_randint.assert_not_called()  # No randomness needed when there's a single max

    @patch("torch.randint", wraps=torch.randint)
    def test_multiple_max_q_values(self, mock_randint):
        # Mock the QNetwork output with multiple max values
        mock_qnet = MagicMock()
        mock_qnet.return_value = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
        self.agent.board_to_state = MagicMock()

        # Call the method
        result = self.agent.get_best_action("mock_board", mock_qnet)

        # Assertions
        self.assertIn(result, [1, 2])  # Result must be one of the max Q-value indices
        mock_randint.assert_called_once_with(2, (1,))  # Called to resolve the tie

    def test_permutation_generation(self):
        board = np.array(["X", " ", " ", " ", " ", " ", " ", " ", " "])
        action = 1
        transformations = [lambda x: x, lambda x: np.rot90(x, k=3)]
        permutations, inverse_permutations = self.agent.generate_permutations(transformations, 3)
        self.assertListEqual(list(board[permutations[1]]), [" ", " ", "X", " ", " ", " ", " ", " ", " "])
        self.assertEqual(inverse_permutations[1][action], 5)

    def test_symmetrized_loss(self):
        self.agent.q_network = MagicMock()
        self.agent.target_network = MagicMock()

        q_network_return_value = torch.zeros((1, 9))
        q_network_return_value[0, 1] = 1
        self.agent.q_network.return_value = q_network_return_value

        target_network_return_value = torch.zeros((1, 9))
        target_network_return_value[0, 5] = 2
        self.agent.target_network.return_value = target_network_return_value

        board = ["X", " ", " ", " ", " ", " ", " ", " ", " "]
        action = torch.tensor([1], dtype=torch.int64)
        reward = torch.tensor([0.0], dtype=torch.float32)
        next_board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        done = torch.tensor([False], dtype=torch.bool)

        state = torch.tensor(self.agent.board_to_state(board)[0], dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(self.agent.board_to_state(next_board)[0], dtype=torch.float32).unsqueeze(0)

        samples = (state, action, reward, next_state, done)
        symmetrized_loss = self.agent.compute_loss(samples)

        transformed_board = [" ", " ", "X", " ", " ", " ", " ", " ", " "]
        transformed_action = torch.tensor([5], dtype=torch.int64)
        transformed_next_board = [" ", " ", "X", " ", " ", "O", " ", " ", " "]
        transformed_reward = torch.tensor([0.0], dtype=torch.float32)
        transformed_done = torch.tensor([False], dtype=torch.bool)

        transformed_state = torch.tensor(self.agent.board_to_state(transformed_board)[0], dtype=torch.float32).unsqueeze(0)
        transformed_next_state = torch.tensor(self.agent.board_to_state(transformed_next_board)[0], dtype=torch.float32).unsqueeze(0)

        transformed_samples = (
            transformed_state,
            transformed_action,
            transformed_reward,
            transformed_next_state,
            transformed_done,
        )

        transformed_symmetrized_loss = self.agent.compute_loss(transformed_samples)
        self.assertAlmostEqual(symmetrized_loss.item(), transformed_symmetrized_loss.item(), places=5)

    def test_set_exploration_rate(self):
        """Test setting exploration rate."""
        self.agent.set_exploration_rate(0.5)
        self.assertEqual(self.agent.epsilon, 0.5, "Exploration rate should be updated correctly.")

    def test_update_exploration_rate(self):
        """Test updating exploration rate based on episode."""
        self.agent.update_exploration_rate(50)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.params["epsilon_min"], "Epsilon should not go below minimum.")
        self.assertLessEqual(self.agent.epsilon, self.agent.params["epsilon_start"], "Epsilon should not exceed start value.")

    def test_flat_state_converter_roundtrip(self):
        board = ["X", "O", " ", "X", " ", "O", " ", "X", "O"]
        state = self.agent.state_converter.board_to_state(board)
        restored = self.agent.state_converter.state_to_board(state)
        self.assertEqual(restored, board)

    def test_init_cnn_onehot(self):
        params = self.params.copy()
        params["network_type"] = "CNN"
        params["state_shape"] = "one-hot"
        agent = DeepQLearningAgent(params)
        self.assertEqual(agent.state_converter.__class__.__name__, "OneHotStateConverter")

    def test_init_fullycnn_flat(self):
        params = self.params.copy()
        params["network_type"] = "FullyCNN"
        params["state_shape"] = "flat"
        agent = DeepQLearningAgent(params)
        self.assertEqual(agent.state_converter.__class__.__name__, "FlatStateConverter")

    def test_init_equivariant_invalid_shape(self):
        params = self.params.copy()
        params["network_type"] = "Equivariant"
        params["state_shape"] = "2D"
        with self.assertRaises(ValueError):
            DeepQLearningAgent(params)

    def test_init_equivariant_invalid_rows(self):
        params = self.params.copy()
        params["network_type"] = "Equivariant"
        params["state_shape"] = "flat"
        params["rows"] = 4  # Even number
        with self.assertRaises(ValueError):
            DeepQLearningAgent(params)

    def test_init_invalid_network_type(self):
        params = self.params.copy()
        params["network_type"] = "UnsupportedNet"
        with self.assertRaises(ValueError):
            DeepQLearningAgent(params)

    def test_init_invalid_state_shape(self):
        params = self.params.copy()
        params["state_shape"] = "weird"
        with self.assertRaises(ValueError):
            DeepQLearningAgent(params)

    def test_handle_incomplete_game_with_none_board(self):
        action = self.agent._handle_incomplete_game(None)
        self.assertEqual(action, -1)

    def test_handle_game_completion_external_epsilon(self):
        self.agent.set_exploration_rate_externally = True
        self.agent._handle_game_completion()
        # Check that exploration rate update wasn't called
        self.assertEqual(self.agent.episode_count, 1)

    def test_board_to_state_wrapper(self):
        board = ["X"] * 9
        state = self.agent.board_to_state(board)
        self.assertEqual(state.shape[1], 9)

    @patch("torch.randint", return_value=torch.tensor([1]))
    def test_get_best_action_multiple_max_values(self, mock_randint):
        board = [" "] * 9
        state = np.array([[0.1, 0.5, 0.5, -0.1, 0, 0, 0, 0, 0]])
        self.agent.board_to_state = MagicMock(return_value=state)
        self.agent.device = torch.device("cpu")

        mock_net = MagicMock()
        mock_net.return_value = torch.tensor([[0.1, 0.5, 0.5, -0.1, 0, 0, 0, 0, 0]])
        action = self.agent.get_best_action(board, mock_net)
        self.assertIn(action, [1, 2])      

    def test_prioritized_and_standard_loss_shapes_and_values(self):
        batch_size = self.agent.batch_size
        state_dim = 9  # Assuming flat state

        # Set dummy linear networks for q_network and target_network
        self.agent.q_network = nn.Linear(state_dim, 9)
        self.agent.target_network = nn.Linear(state_dim, 9)

        # Generate dummy sample data
        states = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, 9, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, state_dim)
        dones = torch.randint(0, 2, (batch_size,), dtype=torch.bool)

        samples = (states, actions, rewards, next_states, dones)

        # Set dummy weights and indices in replay buffer
        self.agent.replay_buffer.last_sampled_weights = torch.ones(batch_size)
        self.agent.replay_buffer.last_sampled_indices = list(range(batch_size))
        self.agent.replay_buffer.update_priorities = MagicMock()

        # Compute losses
        prioritized_loss = self.agent.compute_prioritized_loss(samples)
        standard_loss = self.agent.compute_standard_loss(samples)

        self.assertEqual(prioritized_loss.shape, torch.Size([]), "Prioritized loss should be a scalar.")
        self.assertEqual(standard_loss.shape, torch.Size([]), "Standard loss should be a scalar.")
        self.assertIsInstance(prioritized_loss.item(), float, "Prioritized loss should be a float.")
        self.assertIsInstance(standard_loss.item(), float, "Standard loss should be a float.")

        # Print for manual inspection if needed
        print(f"Prioritized Loss: {prioritized_loss.item():.6f}")
        print(f"Standard Loss: {standard_loss.item():.6f}")


class TestStateConverters(unittest.TestCase):
    def test_flat_state_converter_roundtrip(self):
        from TicTacToe.DeepQAgent import FlatStateConverter
        board = ["X", "O", " ", "X", " ", "O", " ", "X", "O"]
        converter = FlatStateConverter()
        state = converter.board_to_state(board)
        restored = converter.state_to_board(state)
        self.assertEqual(restored, board)

    def test_grid_state_converter_roundtrip(self):
        from TicTacToe.DeepQAgent import GridStateConverter
        board = ["X", "O", " ", "X", " ", "O", " ", "X", "O"]
        converter = GridStateConverter(shape=(3, 3))
        state = converter.board_to_state(board)
        restored = converter.state_to_board(state)
        self.assertEqual(restored, board)

    def test_onehot_state_converter_roundtrip(self):
        from TicTacToe.DeepQAgent import OneHotStateConverter
        board = ["X", "O", " ", "X", " ", "O", " ", "X", "O"]
        converter = OneHotStateConverter(rows=3)
        state = converter.board_to_state(board)
        restored = converter.state_to_board(state)
        self.assertEqual(restored, board)

# Mock classes for dependencies
class MockQNetwork(torch.nn.Module):
    def forward(self, state_tensor):
        return torch.tensor([[0.1, 0.5, 0.3, 0.7]])


class MockTicTacToe:
    def get_board(self):
        return ["X", " ", "O", " ", " ", "X", "O", " ", " "]


class TestDeepQPlayingAgent(unittest.TestCase):
    def setUp(self):
        self.q_network = MockQNetwork()
        self.agent = DeepQPlayingAgent(q_network=self.q_network, player="X", switching=True)

    def test_board_to_state(self):
        board = ["X", " ", "O", " ", " ", "X", "O", " ", " "]
        expected_state = np.array([[1, 0, -1, 0, 0, 1, -1, 0, 0]])
        np.testing.assert_array_equal(self.agent.board_to_state(board), expected_state)

    def test_state_to_board(self):
        state = np.array([[1, 0, -1, 0, 0, 1, -1, 0, 0]])
        expected_board = ["X", " ", "O", " ", " ", "X", "O", " ", " "]
        self.assertEqual(self.agent.state_to_board(state), expected_board)

    def test_get_valid_actions(self):
        board = ["X", " ", "O", " ", " ", "X", "O", " ", " "]
        expected_actions = [1, 3, 4, 7, 8]
        self.assertEqual(self.agent.get_valid_actions(board), expected_actions)

    def test_choose_action(self):
        board = ["X", " ", "O", " ", " ", "X", "O", " ", " "]
        action = self.agent.choose_action(board)
        self.assertEqual(action, 3)  # Mock network outputs highest Q-value for index 3.

    @patch("torch.load", return_value=MockQNetwork())
    def test_q_network_loading(self, mock_load):
        agent = DeepQPlayingAgent(q_network="mock_path.pth", player="X", switching=False)
        self.assertIsInstance(agent.q_network, MockQNetwork)
        mock_load.assert_called_once_with("mock_path.pth", weights_only=False)

    def test_get_action_not_done(self):
        mock_game = MockTicTacToe()
        state_transition = (None, None, False)  # State, action, and done flag.
        action = self.agent.get_action(state_transition, mock_game)
        self.assertEqual(action, 3)  # Mock network predicts action 3.

    def test_get_action_done(self):
        mock_game = MockTicTacToe()
        state_transition = (None, None, True)  # Done flag is True.
        action = self.agent.get_action(state_transition, mock_game)
        self.assertEqual(action, -1)  # Game is over, no action taken.

    def test_on_game_end(self):
        initial_player = self.agent.player
        initial_opponent = self.agent.opponent
        self.agent.on_game_end(None)  # Pass None for game, not used.
        self.assertEqual(self.agent.player, initial_opponent)
        self.assertEqual(self.agent.opponent, initial_player)