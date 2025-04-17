# ReplayBuffers.py
import torch
from typing import Tuple
from TicTacToe.game_types import Action, Reward, State

class BaseReplayBuffer:
    """
    Base class interface for all replay buffers.
    """
    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ReplayBuffer(BaseReplayBuffer):
    """
    Standard uniform sampling replay buffer.
    """
    def __init__(self, size: int, state_shape: Tuple[int, ...], device: str = "cpu") -> None:
        self.size = size
        self.current_size = 0
        self.index = 0
        self.device = device

        self.states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)

    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        self.states[self.index] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.index] = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.index] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.index] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.index] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        self.index = (self.index + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size: int):
        if self.current_size < batch_size:
            raise ValueError("Not enough experiences in the buffer to sample a batch.")

        self.last_sampled_weights = torch.ones(batch_size, device=self.device)

        last_index = (self.index - 1) % self.current_size
        batch_size -= 1

        indices = torch.randint(0, self.current_size - 1, (batch_size,), device=self.device)
        indices = torch.cat([indices, torch.tensor([last_index], device=self.device)])

        self.last_sampled_indices = indices

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        pass

    def __len__(self) -> int:
        return self.current_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    """
    def __init__(self, size, state_shape, device="cpu", alpha=0.6, beta=0.4):
        super().__init__(size, state_shape, device)
        self.alpha = alpha
        self.beta = beta
        self.priorities = torch.zeros(size, dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        max_prio = self.priorities.max().item() if self.current_size > 0 else 1.0
        self.priorities[self.index - 1] = max_prio

    def sample(self, batch_size):
        if self.current_size < batch_size:
            raise ValueError("Not enough samples to draw batch.")

        prios = self.priorities[:self.current_size] ** self.alpha
        if prios.sum() == 0:
            probs = torch.ones_like(prios) / self.current_size
        else:
            probs = prios / prios.sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self.current_size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
        self.last_sampled_indices = indices
        self.last_sampled_weights = weights
        return samples

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        self.priorities[indices] = td_errors.abs() + 1e-5
