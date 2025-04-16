import torch

from TicTacToe.game_types import (
    Action,
    Reward,
    State,
)


class ReplayBuffer:
    """
    A replay buffer for storing and sampling experiences.
    """

    def __init__(self, size: int, state_dim: int, device: str = "cpu") -> None:
        """
        Initialize the ReplayBuffer.

        Args:
            size: Maximum number of experiences to store.
            state_dim: Dimension of the state tensor.
            device: The device to store the buffer on.
        """
        self.size = size
        self.current_size = 0  # Tracks the current number of stored experiences
        self.index = 0  # Tracks the insertion index for circular overwriting
        self.device = device

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)

    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        """
        Add a new experience to the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observed.
            done: Whether the episode is done.
        """
        self.states[self.index] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.index] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.index] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.index] = torch.tensor(done, dtype=torch.bool, device=self.device)

        # Update the index and current size
        self.index = (self.index + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer directly on the GPU.
        Ensure the most recently added experience is always included.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            A tuple of sampled tensors (states, actions, rewards, next_states, dones).
        """
        if self.current_size < batch_size:
            raise ValueError("Not enough experiences in the buffer to sample a batch.")

        # Ensure the last added element is always included
        last_index = self.current_size - 1  # Index of the most recent addition
        batch_size -= 1  # Reduce batch size for random sampling

        # Randomly sample the remaining indices
        indices = torch.randint(0, self.current_size - 1, (batch_size,), device=self.device)  # Exclude the last element
        indices = torch.cat([indices, torch.tensor([last_index], device=self.device)])  # Add the last element

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            The number of experiences currently stored in the buffer.
        """
        return self.current_size