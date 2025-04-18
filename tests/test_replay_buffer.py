# type: ignore

import unittest

import numpy as np
import torch

from TicTacToe.DeepQAgent import ReplayBuffer
from TicTacToe.DeepQAgent import PrioritizedReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer class."""

    def test_initialization(self) -> None:
        """Test ReplayBuffer initialization with proper size and device allocation."""
        buffer = ReplayBuffer(size=10, state_shape=(4, ), device="cpu")
        self.assertEqual(buffer.size, 10, "Buffer size should be correctly initialized.")
        self.assertEqual(buffer.device, "cpu", "Buffer device should be correctly initialized.")
        self.assertEqual(len(buffer), 0, "Buffer should initially have zero stored experiences.")
        self.assertTrue(torch.is_tensor(buffer.states), "States should be stored as a torch tensor.")

    def test_add_experience(self) -> None:
        """Ensure experiences are added correctly, including overwriting behavior."""
        buffer = ReplayBuffer(size=3, state_shape=(4, ), device="cpu")
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
        buffer = ReplayBuffer(size=5, state_shape=(4, ), device="cpu")
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
        buffer = ReplayBuffer(size=5, state_shape=(4, ), device="cpu")
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


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests for the PrioritizedReplayBuffer class."""

    def test_initialization(self) -> None:
        """Test PrioritizedReplayBuffer initialization with proper size and device allocation."""
        buffer = PrioritizedReplayBuffer(size=10, state_shape=(4,), device="cpu")
        self.assertEqual(buffer.size, 10, "Buffer size should be correctly initialized.")
        self.assertEqual(buffer.device, "cpu", "Buffer device should be correctly initialized.")
        self.assertEqual(len(buffer), 0, "Buffer should initially have zero stored experiences.")
        self.assertTrue(torch.is_tensor(buffer.priorities), "Priorities should be stored as a torch tensor.")

    def test_add_experience(self) -> None:
        """Ensure experiences are added correctly with priorities."""
        buffer = PrioritizedReplayBuffer(size=3, state_shape=(4,), device="cpu")
        for i in range(5):  # Add more experiences than the buffer size
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        self.assertEqual(len(buffer), 3, "Buffer size should not exceed its capacity.")

    def test_sample_experiences(self) -> None:
        """Check that sampling returns correct shapes and values with priority-based sampling."""
        buffer = PrioritizedReplayBuffer(size=5, state_shape=(4,), device="cpu")
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

    def test_update_priorities(self) -> None:
        """Ensure priorities are updated correctly."""
        buffer = PrioritizedReplayBuffer(size=5, state_shape=(4,), device="cpu")
        for i in range(5):
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        td_errors = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
        buffer.update_priorities(indices, td_errors)

        self.assertTrue(
            torch.equal(buffer.priorities[indices], td_errors.abs() + 1e-5),
            "Priorities should be updated based on TD errors.",
        )
