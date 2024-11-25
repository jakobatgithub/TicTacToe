# type: ignore

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import torch

from TicTacToe.DeepQAgent import DeepQLearningAgent, QNetwork, ReplayBuffer


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
        """Check that the sampling returns correct shapes and values."""
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
        self.assertEqual(states.shape, (batch_size, 4), "Sampled states should have correct shape.")
        self.assertEqual(actions.shape, (batch_size,), "Sampled actions should have correct shape.")
        self.assertEqual(rewards.shape, (batch_size,), "Sampled rewards should have correct shape.")
        self.assertEqual(next_states.shape, (batch_size, 4), "Sampled next states should have correct shape.")
        self.assertEqual(dones.shape, (batch_size,), "Sampled dones should have correct shape.")

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

        expected_params = (input_dim * 128 + 128) + (128 * 64 + 64) + (64 * output_dim + output_dim)
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
            "debug": False,
            "rows": 3,
            "device": "cpu",
        }
        self.agent = DeepQLearningAgent(self.params)

    def test_initialization(self) -> None:
        """Check initialization of networks, replay buffer, and parameters."""
        self.assertIsInstance(self.agent.q_network, QNetwork, "Q-network should be an instance of QNetwork.")
        self.assertIsInstance(self.agent.target_network, QNetwork, "Target network should be an instance of QNetwork.")
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

        initial_weights = [param.clone() for param in self.agent.q_network.parameters()]
        self.agent.get_action((None, 0, True), None)  # Trigger training
        updated_weights = [param for param in self.agent.q_network.parameters()]

        for init, updated in zip(initial_weights, updated_weights):
            self.assertFalse(torch.equal(init, updated), "Weights should update after training.")

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
