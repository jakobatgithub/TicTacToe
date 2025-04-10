# type: ignore

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from TicTacToe.DeepQAgent import DeepQLearningAgent, DeepQPlayingAgent, QNetwork, CNNQNetwork, ReplayBuffer, FullyConvQNetwork


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer class."""

    def test_initialization(self) -> None:
        """Test ReplayBuffer initialization with proper size and device allocation."""
        buffer = ReplayBuffer(size=10, state_dim=4, device="cpu")
        self.assertEqual(buffer.size, 10, "Buffer size should be correctly initialized.")
        self.assertEqual(buffer.device, "cpu", "Buffer device should be correctly initialized.")
        self.assertEqual(len(buffer), 0, "Buffer should initially have zero stored experiences.")
        self.assertTrue(torch.is_tensor(buffer.states), "States should be stored as a torch tensor.")

    def test_add_experience(self) -> None:
        """Ensure experiences are added correctly, including overwriting behavior."""
        buffer = ReplayBuffer(size=3, state_dim=4, device="cpu")
        for i in range(5):  # Add more experiences than the buffer size
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        self.assertEqual(len(buffer), 3, "Buffer size should not exceed its capacity.")
        self.assertTrue(
            torch.equal(buffer.states[0], torch.tensor([3, 4, 5, 6], dtype=torch.float32)),
            "Oldest experiences should be overwritten in circular fashion.",
        )
        self.assertTrue(
            torch.equal(buffer.states[1], torch.tensor([4, 5, 6, 7], dtype=torch.float32)),
            "Oldest experiences should be overwritten in circular fashion.",
        )
        self.assertFalse(buffer.dones[0], "Stored 'done' value should match the input.")

    def test_sample_experiences(self) -> None:
        """Check that the sampling returns correct shapes and values, including the last added experience."""
        buffer = ReplayBuffer(size=5, state_dim=4, device="cpu")
        for i in range(5):
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        batch_size = 3
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # Check shapes
        self.assertEqual(states.shape, (batch_size, 4), "Sampled states should have correct shape.")
        self.assertEqual(actions.shape, (batch_size,), "Sampled actions should have correct shape.")
        self.assertEqual(rewards.shape, (batch_size,), "Sampled rewards should have correct shape.")
        self.assertEqual(next_states.shape, (batch_size, 4), "Sampled next states should have correct shape.")
        self.assertEqual(dones.shape, (batch_size,), "Sampled dones should have correct shape.")

        # Ensure the last added experience is included
        last_state = np.array([4, 5, 6, 7])
        last_action = 4
        last_reward = 4.0
        last_next_state = np.array([5, 6, 7, 8])
        last_done = True

        # Convert sampled tensors back to numpy for easier comparison
        states_np = states.numpy()
        actions_np = actions.numpy()
        rewards_np = rewards.numpy()
        next_states_np = next_states.numpy()
        dones_np = dones.numpy()

        # Assert that the last experience is present in the sampled batch
        self.assertTrue(
            any(
                np.array_equal(state, last_state)
                and action == last_action
                and reward == last_reward
                and np.array_equal(next_state, last_next_state)
                and done == last_done
                for state, action, reward, next_state, done in zip(
                    states_np, actions_np, rewards_np, next_states_np, dones_np
                )
            ),
            "The most recently added experience must be in the sampled batch.",
        )

    def test_buffer_length(self) -> None:
        """Verify __len__ returns the correct number of stored experiences."""
        buffer = ReplayBuffer(size=5, state_dim=4, device="cpu")
        self.assertEqual(len(buffer), 0, "Initial buffer length should be zero.")
        for i in range(7):  # Add more experiences than buffer size
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )
        self.assertEqual(len(buffer), 5, "Buffer length should match its capacity after being filled.")


class TestQNetwork(unittest.TestCase):
    """Tests for the QNetwork class."""

    def test_initialization(self) -> None:
        """Test QNetwork initialization with correct input and output dimensions."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)

        self.assertEqual(model.fc[0].in_features, input_dim, "Input dimension of the first layer is incorrect.")
        self.assertEqual(model.fc[-1].out_features, output_dim, "Output dimension of the last layer is incorrect.")

    def test_forward_pass(self) -> None:
        """Ensure the forward pass produces output of the correct shape."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))  # Batch of 5 inputs
        output = model(test_input)

        self.assertEqual(output.shape, (5, output_dim), "Output shape is incorrect.")

    def test_gradient_flow(self) -> None:
        """Confirm that gradients flow correctly during backpropagation."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))
        target = torch.randn((5, output_dim))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")

    def test_parameter_count(self) -> None:
        """Verify that the number of trainable parameters is as expected."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # n1, n2 = 128, 64
        n1, n2 = 49, 49
        expected_params = (input_dim * n1 + n1) + (n1 * n2 + n2) + (n2 * output_dim + output_dim)
        self.assertEqual(total_params, expected_params, "The number of trainable parameters is incorrect.")

    def test_reproducibility(self) -> None:
        """Check that the network produces consistent outputs in evaluation mode."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)

        self.assertTrue(torch.equal(output1, output2), "Outputs should be consistent in evaluation mode.")


class TestCNNQNetwork(unittest.TestCase):
    """Tests for the CNNQNetwork class."""

    def test_initialization(self):
        """Test CNNQNetwork initialization with correct input and output dimensions."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, grid_size=grid_size, output_dim=output_dim)

        self.assertEqual(
            model.conv_layers[0].in_channels, input_dim, "Input channels of the first convolutional layer are incorrect."
        )
        # Find the last Conv2d layer in the conv_layers sequence
        last_conv_layer = [layer for layer in model.conv_layers if isinstance(layer, nn.Conv2d)][-1]
        self.assertEqual(
            last_conv_layer.out_channels, 64, "Output channels of the last convolutional layer are incorrect."
        )
        self.assertEqual(
            model.fc_layers[-1].out_features, output_dim, "Output dimension of the final fully connected layer is incorrect."
        )

    def test_forward_pass(self):
        """Ensure the forward pass produces output of the correct shape."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, grid_size=grid_size, output_dim=output_dim)
        test_input = torch.randn((5, input_dim, grid_size, grid_size))  # Batch of 5 inputs
        output = model(test_input)

        self.assertEqual(output.shape, (5, output_dim), "Output shape is incorrect.")

    def test_gradient_flow(self):
        """Confirm that gradients flow correctly during backpropagation."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, grid_size=grid_size, output_dim=output_dim)
        test_input = torch.randn((5, input_dim, grid_size, grid_size))
        target = torch.randn((5, output_dim))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")

    def test_parameter_count(self):
        """Verify that the number of trainable parameters is as expected."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, grid_size=grid_size, output_dim=output_dim)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate expected parameters
        conv1_params = (input_dim * 32 * 3 * 3) + 32
        conv2_params = (32 * 64 * 3 * 3) + 64
        conv3_params = (64 * 64 * 3 * 3) + 64
        fc1_params = (64 * grid_size * grid_size * 128) + 128
        fc2_params = (128 * output_dim) + output_dim
        expected_params = conv1_params + conv2_params + conv3_params + fc1_params + fc2_params

        self.assertEqual(total_params, expected_params, "The number of trainable parameters is incorrect.")


class TestFullyCNNQNetwork(unittest.TestCase):
    """Tests for the FullyConvQNetwork class."""

    def test_output_dimensions(self):
        """Test that the output dimension is correct."""
        batch_size, input_dim, rows = 1, 1, 5  # Example grid size
        model = FullyConvQNetwork(input_dim=input_dim)

        # Create a test input tensor
        test_input = torch.zeros((batch_size, input_dim, rows, rows))
        test_input[0, 0, 2, 2] = 1.0  # Set a single active cell in the center
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, rows * rows), "Output shape is incorrect.")

    def test_shift_invariance(self):
        """Test that shifting the input and then applying conv_layers yields the same result as applying conv_layers and then shifting the output."""
        batch_size, input_dim, rows = 1, 1, 5  # Example grid size
        model = FullyConvQNetwork(input_dim=input_dim)

        # Create a test input tensor
        test_input = torch.zeros((batch_size, input_dim, rows, rows))
        test_input[0, 0, 2, 2] = 1.0  # Set a single active cell in the center

        # Define a shift (x, y)
        shift_x, shift_y = 1, 2

        # Shift the input tensor
        shifted_input = torch.roll(test_input, shifts=(shift_x, shift_y), dims=(2, 3))

        # Apply conv_layers to both the original and shifted inputs
        original_output = model.conv_layers(test_input)
        shifted_output = model.conv_layers(shifted_input)

        # Shift the original output by the same (x, y) vector
        shifted_original_output = torch.roll(original_output, shifts=(shift_x, shift_y), dims=(2, 3))

        # Compare the results
        self.assertTrue(
            torch.allclose(shifted_output, shifted_original_output, atol=1e-3),
            "Shifting the input and then applying conv_layers should yield the same result as applying conv_layers and then shifting the output.",
        )

    def test_random_input(self):
        """Test that the network produces consistent outputs for the same random input."""
        batch_size, input_dim, rows = 1, 1, 5  # Example grid size
        model = FullyConvQNetwork(input_dim=input_dim)

        # Create a random input tensor
        test_input = torch.randn((batch_size, input_dim, rows, rows))

        # Apply the network twice
        output1 = model(test_input)
        output2 = model(test_input)

        # Assert that the outputs are identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5), "Outputs should be consistent for the same input.")

    def test_gradient_flow(self):
        """Test that gradients flow correctly during backpropagation."""
        batch_size, input_dim, rows = 1, 1, 5  # Example grid size
        model = FullyConvQNetwork(input_dim=input_dim)

        # Create a random input tensor
        test_input = torch.randn((batch_size, input_dim, rows, rows), requires_grad=True)
        target = torch.randn((batch_size, rows * rows))

        # Define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Perform a forward pass
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)

        # Perform a backward pass
        loss.backward()

        # Check that gradients exist for all parameters
        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")


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
        self.agent.board_to_state = MagicMock(side_effect=[1, 2])

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
        self.agent.update_rates = MagicMock()

        # Call the method
        self.agent._handle_game_completion()

        # Assertions
        self.agent.update_rates.assert_called_with(6)  # Incremented episode count
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
        loss = self.agent.compute_loss(samples)

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
        action = 1
        next_board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]

        state = torch.tensor(self.agent.board_to_state(board)[0], dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(0.0, dtype=torch.float32)
        next_state = torch.tensor(self.agent.board_to_state(next_board)[0], dtype=torch.float32)
        done = torch.tensor(False, dtype=torch.bool)

        states = torch.vstack([state])
        actions = torch.vstack([action])
        rewards = torch.vstack([reward])
        next_states = torch.vstack([next_state])
        dones = torch.vstack([done])

        samples = (states, actions, rewards, next_states, dones)
        symmetrized_loss = self.agent.compute_symmetrized_loss(samples)

        transformed_board = [" ", " ", "X", " ", " ", " ", " ", " ", " "]
        transformed_action = 5
        transformed_next_board = [" ", " ", "X", " ", " ", "O", " ", " ", " "]

        transformed_state = torch.tensor(self.agent.board_to_state(transformed_board)[0], dtype=torch.float32)
        transformed_action = torch.tensor(transformed_action, dtype=torch.int64)
        transformed_reward = torch.tensor(0.0, dtype=torch.float32)
        transformed_next_state = torch.tensor(self.agent.board_to_state(transformed_next_board)[0], dtype=torch.float32)
        transformed_done = torch.tensor(False, dtype=torch.bool)

        transformed_states = torch.vstack([transformed_state])
        transformed_actions = torch.vstack([transformed_action])
        transformed_rewards = torch.vstack([transformed_reward])
        transformed_next_states = torch.vstack([transformed_next_state])
        transformed_dones = torch.vstack([transformed_done])

        transformed_samples = (
            transformed_states,
            transformed_actions,
            transformed_rewards,
            transformed_next_states,
            transformed_dones,
        )
        transformed_symmetrized_loss = self.agent.compute_symmetrized_loss(transformed_samples)
        self.assertAlmostEqual(symmetrized_loss.item(), transformed_symmetrized_loss.item(), places=5)


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
