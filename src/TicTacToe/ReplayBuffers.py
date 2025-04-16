import torch

from typing import Tuple

from TicTacToe.game_types import (
    Action,
    Reward,
    State
)


class ReplayBuffer:
    """
    A generalized replay buffer for storing and sampling experiences with arbitrary state shapes.
    """

    def __init__(self, size: int, state_shape: Tuple[int, ...], device: str = "cpu") -> None:
        """
        Initialize the ReplayBuffer.

        Args:
            size: Maximum number of experiences to store.
            state_shape: Shape of a single state tensor (e.g., (4, 4) or (3, 84, 84)).
            device: The device to store the buffer on.
        """
        self.size = size
        self.current_size = 0
        self.index = 0
        self.device = device

        # Pre-allocate tensors with arbitrary state shape
        self.states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)

    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        """
        Add a new experience to the buffer.
        """
        self.states[self.index] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.index] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.index] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        self.index = (self.index + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer. Always includes the most recent.
        """
        if self.current_size < batch_size:
            raise ValueError("Not enough experiences in the buffer to sample a batch.")

        last_index = (self.index - 1) % self.current_size
        batch_size -= 1

        indices = torch.randint(0, self.current_size - 1, (batch_size,), device=self.device)
        indices = torch.cat([indices, torch.tensor([last_index], device=self.device)])

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.current_size
