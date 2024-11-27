import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from TicTacToe.Agent import Agent

if TYPE_CHECKING:
    from TicTacToe.TicTacToe import TwoPlayerBoardGame  # Import only for type hinting

from TicTacToe.game_types import (
    Action,
    Actions,
    Board,
    History,
    Player,
    Reward,
    State,
    StateTransition,
    StateTransitions2,
)


class ReplayBuffer:
    def __init__(self, size: int, state_dim: int, device: str = "cpu") -> None:
        """
        Initialize the ReplayBuffer with a fixed size and GPU storage.

        :param size: Maximum number of experiences to store in the buffer.
        :param state_dim: Dimension of the state tensor.
        :param device: The device to store the buffer on ("cuda" for GPU or "cpu").
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

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state observed.
        :param done: Whether the episode is done.
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

        :param batch_size: Number of experiences to sample.
        :return: A tuple of sampled tensors (states, actions, rewards, next_states, dones).
        """
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
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

        :return: The number of experiences currently stored in the buffer.
        """
        return self.current_size


# Neural Network for Q-function
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(QNetwork, self).__init__()  # type: ignore
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepQLearningAgent(Agent):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(player=params["player"], switching=params["switching"])
        self.params = params
        self.debug = params["debug"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon_start"]
        self.nr_of_episodes = params["nr_of_episodes"]
        self.batch_size = params["batch_size"]
        self.target_update_frequency = params["target_update_frequency"]
        self.learning_rate = params["learning_rate"]
        self.replay_buffer_length = params["replay_buffer_length"]
        self.wandb_logging_frequency = params["wandb_logging_frequency"]

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0
        self.target_update_count = 0

        wandb.init(config=params)  # type: ignore

        self.episode_history: History = []
        self.state_transitions: StateTransitions2 = []
        self.rows = params["rows"]
        (state_size, action_size) = (self.rows**2, self.rows**2)
        self.device = torch.device(params["device"])
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_length, self.rows**2, device=params["device"])

        self.board_to_state_translation = {"X": 1, "O": -1, " ": 0}
        self.state_to_board_translation = {1: "X", -1: "O", 0: " "}

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")

        self.evaluation_data: dict[str, Any] = {
            "loss": [],
            "action_value": [],
            "histories": [],
            "rewards": [],
            "valid_actions": [],
        }

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        next_board, reward, done = state_transition

        if len(self.episode_history) > 0:
            self._update_state_transitions_and_replay_buffer(next_board, reward, done)
            if len(self.replay_buffer) >= self.batch_size:
                self._train_network(reward)

        if not done:
            return self._handle_incomplete_game(next_board)
        else:
            self._handle_game_completion()
            return -1

    def _update_state_transitions_and_replay_buffer(self, next_board: Board | None, reward: Reward, done: bool) -> None:
        board, action = self.episode_history[-1]
        self.state_transitions.append((board, action, next_board, reward, done))

        state = self.board_to_state(board)
        next_state = (
            self.board_to_state(next_board) if next_board is not None else self.board_to_state(["X"] * (self.rows**2))
        )

        self.replay_buffer.add(state, action, reward, next_state, done)

    def compute_loss(
        self, samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, float]:
        states, actions, rewards, next_states, dones = samples
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].squeeze(1)
        targets = rewards + (~dones) * self.gamma * next_q_values
        avg_next_q_value = next_q_values.mean().item()
        return nn.MSELoss()(q_values, targets), avg_next_q_value

    def _train_network(self, reward: Reward) -> None:
        samples = self.replay_buffer.sample(self.batch_size)

        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].squeeze(1)
        # targets = rewards + (~dones) * self.gamma * next_q_values

        # loss = nn.MSELoss()(q_values, targets)
        loss, next_q_values = self.compute_loss(samples)

        self._log_training_metrics(loss.item(), next_q_values, reward)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore
        self.train_step_count += 1

    def _log_training_metrics(self, loss: float, next_q_values: float, reward: Reward) -> None:
        self.evaluation_data["loss"].append(loss)
        self.evaluation_data["action_value"].append(next_q_values)
        self.evaluation_data["rewards"].append(reward)

        if self.train_step_count % self.wandb_logging_frequency == 0:
            wandb.log(
                {
                    "loss": sum(self.evaluation_data["loss"][-self.wandb_logging_frequency :])
                    / self.wandb_logging_frequency,
                    "action_value": sum(self.evaluation_data["action_value"][-self.wandb_logging_frequency :])
                    / self.wandb_logging_frequency,
                    "reward": sum(self.evaluation_data["rewards"][-self.wandb_logging_frequency :])
                    / self.wandb_logging_frequency,
                    "episode_count": self.episode_count,
                }
            )

    def _handle_incomplete_game(self, next_board: Board | None) -> Action:
        if next_board is not None:
            action = self.choose_action(next_board, epsilon=self.epsilon)
            self.episode_history.append((next_board, action))
            self.games_moves_count += 1
            return action
        return -1

    def _handle_game_completion(self) -> None:
        if self.episode_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_count += 1

        self.episode_count += 1
        self.update_rates(self.episode_count)
        self.evaluation_data["histories"].append(self.episode_history)
        self.episode_history = []

    def board_to_state(self, board: Board):
        return np.array([self.board_to_state_translation[cell] for cell in board]).reshape(1, -1)
        # return [self.board_to_state_translation[cell] for cell in board]

    def state_to_board(self, state: State) -> Board:
        flat_state = state.flatten()
        board = [self.state_to_board_translation[cell] for cell in flat_state]
        return board

    def update_rates(self, episode: int) -> None:
        self.epsilon = max(
            self.params["epsilon_min"], self.params["epsilon_start"] / (1 + episode / self.nr_of_episodes)
        )

    def get_valid_actions(self, board: Board):
        return [i for i, cell in enumerate(board) if cell == " "]

    # Choose an action based on Q-values
    def choose_action(self, board: Board, epsilon: float) -> Action:
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            action = self.get_best_action(board, self.q_network)
            return action

    def get_best_action(self, board: Board, QNet: QNetwork) -> Action:
        state = self.board_to_state(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = QNet(state_tensor).squeeze()
            max_q = torch.max(q_values)
            max_q_indices = torch.nonzero(q_values == max_q, as_tuple=False)
            if len(max_q_indices) > 1:
                action = int(max_q_indices[torch.randint(len(max_q_indices), (1,))].item())
            else:
                action = int(max_q_indices)

        return action


class DeepQPlayingAgent(Agent):
    def __init__(self, q_network: QNetwork | str, player: Player = "X", switching: bool = False) -> None:
        super().__init__(player=player, switching=switching)
        # self.device = torch.device('mps')
        self.device = torch.device("cpu")
        if isinstance(q_network, torch.nn.Module):
            self.q_network: QNetwork = q_network.to(self.device)
        else:
            self.q_network = torch.load(q_network).to(self.device)  # type: ignore
            self.q_network.eval()

        self.state_to_board_translation = {"X": 1, "O": -1, " ": 0}
        self.board_to_state_translation: dict[int, str] = {}
        for key, value in self.state_to_board_translation.items():
            self.board_to_state_translation[value] = key

    def board_to_state(self, board: Board):
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)

    def state_to_board(self, state: State) -> Board:
        flat_state = state.flatten()
        board = [self.board_to_state_translation[cell] for cell in flat_state]
        return board

    # Generate all empty positions on the board
    def get_valid_actions(self, board: Board) -> Actions:
        return [i for i, cell in enumerate(board) if cell == " "]

    def choose_action(self, board: Board) -> int:
        # Exploitation: Choose the best known move
        action = self.get_best_action(board, self.q_network)
        return action

    def get_best_action(self, board: Board, QNet: QNetwork) -> Action:
        state = self.board_to_state(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = QNet(state_tensor).squeeze()
            max_q = torch.max(q_values)
            max_q_indices = torch.nonzero(q_values == max_q, as_tuple=False)
            if len(max_q_indices) > 1:
                action = int(max_q_indices[torch.randint(len(max_q_indices), (1,))].item())
            else:
                action = int(max_q_indices)

        return action

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        _, _, done = state_transition
        if not done:
            board = game.get_board()
            action = self.choose_action(board)
            return action
        else:
            self.on_game_end(game)
            return -1

    def on_game_end(self, game: "TwoPlayerBoardGame") -> None:
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
