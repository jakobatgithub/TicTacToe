import random
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from TicTacToe.Agent import Agent
from TicTacToe.QNetworks import QNetwork, CNNQNetwork, FullyConvQNetwork, EquivariantNN
from TicTacToe.EvaluationMixin import EvaluationMixin
from TicTacToe.ReplayBuffers import ReplayBuffer, PrioritizedReplayBuffer

if TYPE_CHECKING:
    from TicTacToe.TicTacToe import TwoPlayerBoardGame  # Import only for type hinting

from TicTacToe.game_types import (
    Action,
    Actions,
    Board,
    History,
    Player,
    Reward,
    StateTransition,
    StateTransitions2,
)


class StateConverter(Protocol):
    def board_to_state(self, board: Board) -> np.ndarray:
        ...
    
    def state_to_board(self, state: np.ndarray) -> Board:
        ...

class FlatStateConverter:
    def __init__(self, state_to_board_translation=None):
        self.state_to_board_translation = state_to_board_translation or {"X": 1, "O": -1, " ": 0}
        self.board_to_state_translation = {v: k for k, v in self.state_to_board_translation.items()}

    def board_to_state(self, board: Board) -> np.ndarray:
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)

    def state_to_board(self, state: np.ndarray) -> Board:
        flat_state = state.flatten()
        return [self.board_to_state_translation[cell] for cell in flat_state]
    
class GridStateConverter:
    def __init__(self, shape: tuple[int, int], state_to_board_translation=None):
        self.shape = shape
        self.state_to_board_translation = state_to_board_translation or {"X": 1, "O": -1, " ": 0}
        self.board_to_state_translation = {v: k for k, v in self.state_to_board_translation.items()}

    def board_to_state(self, board: Board) -> np.ndarray:
        grid = np.array([self.state_to_board_translation[cell] for cell in board])
        return grid.reshape(1, 1, *self.shape)

    def state_to_board(self, state: np.ndarray) -> Board:
        flat_state = state.flatten()
        return [self.board_to_state_translation[cell] for cell in flat_state]

class OneHotStateConverter:
    """
    Converts board states to one-hot encoded numpy arrays of shape (1, 3, rows, rows),
    where channels represent 'X', 'O', and empty respectively.
    """

    def __init__(self, rows: int):
        self.rows = rows
        self.cell_to_index = {"X": 0, "O": 1, " ": 2}
        self.index_to_cell = {v: k for k, v in self.cell_to_index.items()}

    def board_to_state(self, board: Board) -> np.ndarray:
        """
        Convert a board to a one-hot encoded state of shape (1, 3, rows, rows).
        """
        state = np.zeros((3, self.rows, self.rows), dtype=np.float32)

        for i, cell in enumerate(board):
            row, col = divmod(i, self.rows)
            channel = self.cell_to_index[cell]
            state[channel, row, col] = 1.0

        return state[np.newaxis, ...]  # shape: (1, 3, rows, rows)

    def state_to_board(self, state: np.ndarray) -> Board:
        """
        Convert a one-hot encoded state back to a flat board list.
        """
        _, channels, rows, cols = state.shape
        decoded = []
        one_hot = state.squeeze(0)  # shape: (3, rows, rows)
        for row in range(rows):
            for col in range(cols):
                channel = np.argmax(one_hot[:, row, col])
                decoded.append(self.index_to_cell[int(channel)])
        return decoded

class DeepQLearningAgent(Agent, EvaluationMixin):
    """
    A Deep Q-Learning agent for playing Tic Tac Toe.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        """
        Initialize the DeepQLearningAgent with configuration parameters.

        Args:
            params: A dictionary of parameters for the agent.
        """
        super().__init__(player=params["player"], switching=params["switching"])
        self.params = params
        self._init_config(params)
        self._init_wandb(params)
        self._init_group_matrices()
        self._init_networks(params)
        self._load_pretrained_weights(params)
        self._init_optimizer()
        self._init_state_converter_and_buffer(params)
        self._override_with_shared_replay_buffer(params)
        self._init_symmetrized_loss(params)
        EvaluationMixin.__init__(
            self, wandb_enabled=params["wandb"], wandb_logging_frequency=params["wandb_logging_frequency"]
        )

    def _init_config(self, params: dict[str, Any]) -> None:
        """
        Initialize internal variables and counters from configuration.

        Args:
            params: The configuration dictionary.
        """
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon_start"]
        self.set_exploration_rate_externally = params["set_exploration_rate_externally"]
        self.nr_of_episodes = params["nr_of_episodes"]
        self.batch_size = params["batch_size"]
        self.target_update_frequency = params["target_update_frequency"]
        self.learning_rate = params["learning_rate"]
        self.replay_buffer_length = params["replay_buffer_length"]
        self.wandb_logging_frequency = params["wandb_logging_frequency"]
        self.wandb = params["wandb"]
        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0
        self.target_update_count = 0
        self.episode_history: History = []
        self.state_transitions: StateTransitions2 = []
        self.rows = params["rows"]
        self.device = torch.device(params["device"])

    def _init_wandb(self, params: dict[str, Any]) -> None:
        """
        Initialize Weights & Biases logging if enabled.

        Args:
            params: The configuration dictionary.
        """
        if self.wandb:
            wandb.init(config=params)  # type: ignore

    def _init_group_matrices(self) -> None:
        """
        Initialize the 2D transformation matrices used in equivariant networks.
        """
        Bs = [
            [[1, 0], [0, 1]], [[-1, 0], [0, -1]], [[-1, 0], [0, 1]], [[1, 0], [0, -1]],
            [[0, 1], [1, 0]], [[0, -1], [1, 0]], [[0, 1], [-1, 0]], [[0, -1], [-1, 0]],
        ]
        self.groupMatrices = [np.array(B) for B in Bs]

    def _init_networks(self, params: dict[str, Any]) -> None:
        """
        Initialize Q-network and target network based on the selected architecture.

        Args:
            params: The configuration dictionary.
        """
        network_type = params["network_type"]
        state_shape = params["state_shape"]
        periodic = params.get("periodic", False)

        if network_type == "Equivariant":
            if state_shape != "flat":
                raise ValueError("Equivariant network requires 'flat' state_shape.")
            if self.rows % 2 != 1:
                raise ValueError("Equivariant network requires an odd number of rows.")
            ms = ((self.rows - 1) / 2, 3, 3, (self.rows - 1) / 2)
            self.q_network = EquivariantNN(self.groupMatrices, ms=ms).to(self.device)
            self.target_network = EquivariantNN(self.groupMatrices, ms=ms).to(self.device)

        elif network_type == "CNN":
            input_dim = 3 if state_shape == "one-hot" else 1
            output_dim = self.rows**2
            self.q_network = CNNQNetwork(input_dim=input_dim, rows=self.rows, output_dim=output_dim).to(self.device)
            self.target_network = CNNQNetwork(input_dim=input_dim, rows=self.rows, output_dim=output_dim).to(self.device)

        elif network_type == "FCN":
            if state_shape != "flat":
                raise ValueError("Fully connected network requires 'flat' state_shape.")
            size = self.rows**2
            self.q_network = QNetwork(size, output_dim=size).to(self.device)
            self.target_network = QNetwork(size, output_dim=size).to(self.device)

        elif network_type == "FullyCNN":
            input_dim = 3 if state_shape == "one-hot" else 1
            padding_mode = "circular" if periodic else "zeros"
            self.q_network = FullyConvQNetwork(input_dim=input_dim, padding_mode=padding_mode).to(self.device)
            self.target_network = FullyConvQNetwork(input_dim=input_dim, padding_mode=padding_mode).to(self.device)

        else:
            raise ValueError(f"Unsupported network type: {network_type}")

    def _load_pretrained_weights(self, params: dict[str, Any]) -> None:
        """
        Load pretrained weights into the networks, if specified.

        Args:
            params: The configuration dictionary.
        """
        if params.get("load_network"):
            state_dict = torch.load(params["load_network"])
            self.q_network.load_state_dict(state_dict)
            self.target_network.load_state_dict(state_dict)

    def _init_optimizer(self) -> None:
        """
        Initialize the optimizer for training the Q-network.
        """
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def _init_state_converter_and_buffer(self, params: dict[str, Any]) -> None:
        """
        Initialize state representation converter and the replay buffer.

        Args:
            params: The configuration dictionary.
        """
        state_shape = params["state_shape"]
        if state_shape == "flat":
            self.state_converter = FlatStateConverter()
            shape = (self.rows**2,)
        elif state_shape == "2D":
            self.state_converter = GridStateConverter(shape=(self.rows, self.rows))
            shape = (1, self.rows, self.rows)
        elif state_shape == "one-hot":
            self.state_converter = OneHotStateConverter(rows=self.rows)
            shape = (3, self.rows, self.rows)
        else:
            raise ValueError(f"Unsupported state shape: {state_shape}")

        buffer_type = params.get("replay_buffer_type", "uniform")
        if buffer_type == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer_length, shape, device=params["device"],
                alpha=params.get("priority_alpha", 0.6),
                beta=params.get("priority_beta", 0.4),
            )
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_length, shape, device=params["device"])

    def _override_with_shared_replay_buffer(self, params: dict[str, Any]) -> None:
        """
        Override the local replay buffer with a shared buffer if provided.

        Args:
            params: The configuration dictionary.
        """
        if params.get("shared_replay_buffer"):
            self.replay_buffer = params["shared_replay_buffer"]

    def _init_symmetrized_loss(self, params: dict[str, Any]) -> None:
        """
        Initialize the symmetrized loss function based on reflection and rotation transformations.

        Args:
            params: The configuration dictionary.
        """
        self.transformations: list[Any] = [
            lambda x: x,
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.flipud(np.fliplr(x)),
            lambda x: np.transpose(x),
            lambda x: np.fliplr(np.transpose(x)),
            lambda x: np.flipud(np.transpose(x)),
            lambda x: np.flipud(np.fliplr(np.transpose(x))),
        ]
        if params.get("symmetrized_loss", True):
            self.compute_loss = self.create_symmetrized_loss(
                self.compute_standard_loss, self.transformations, self.rows
            )
        elif params.get("replay_buffer_type", "uniform") == "prioritized":
            self.compute_loss = self.compute_prioritized_loss
        else:
            self.compute_loss = self.compute_standard_loss

    def set_exploration_rate(self, epsilon: float) -> None:
        """
        Set the exploration rate (epsilon).

        Args:
            epsilon: The exploration rate.
        """
        self.epsilon = epsilon

    def generate_permutations(self, transformations: list[Any], rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate permutations and their inverses for symmetrized loss computation.

        Args:
            transformations: List of transformations.
            rows: Number of rows in the board.

        Returns:
            A tuple of permutations and inverse permutations.
        """
        indices = np.array(object=list(range(rows * rows)), dtype=int).reshape(rows, rows)
        permutations = np.array([transform(indices).flatten().tolist() for transform in transformations], dtype=int)
        inverse_permutations_list: list[Any] = []
        for permutation in permutations:
            inverse_permutation = np.empty_like(permutation)
            inverse_permutation[permutation] = np.arange(len(permutation))
            inverse_permutations_list.append(inverse_permutation)

        permutations = torch.tensor(permutations)
        inverse_permutations = torch.tensor(np.array(inverse_permutations_list, dtype=int))
        return permutations, inverse_permutations

    def create_symmetrized_loss(
        self, loss: Callable[..., Any], transformations: list[Any], rows: int
    ) -> Callable[..., Any]:
        """
        Create a symmetrized loss function.

        Args:
            loss: Original loss function.
            transformations: List of transformations.
            rows: Number of rows in the board.

        Returns:
            A symmetrized loss function.
        """
        permutations, inverse_permutations = self.generate_permutations(transformations, rows)
        permutations = [p.clone().detach().to(self.device) for p in permutations]
        inverse_permutations = [ip.clone().detach().to(self.device) for ip in inverse_permutations]

        def apply_permutation(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            if x.dim() == 2:
                # Shape: (B, rows*rows)
                return x[:, perm]
            elif x.dim() == 4:
                # Shape: (B, C, rows, rows)
                B, C, H, W = x.shape
                flat = x.view(B, C, -1)  # (B, C, rows*rows)
                permuted = flat[:, :, perm]  # Apply permutation to last dim
                return permuted.view(B, C, H, W)
            else:
                raise ValueError(f"Unsupported tensor shape: {x.shape}")

        def symmetrized_loss(samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
            states, actions, rewards, next_states, dones = samples

            total_loss = 0.0
            for p, ip in zip(permutations, inverse_permutations):
                ts = apply_permutation(states, p)
                ta = ip[actions]
                tns = apply_permutation(next_states, p)
                total_loss += loss((ts, ta, rewards, tns, dones))

            return total_loss / len(permutations)

        return symmetrized_loss

    def compute_standard_loss(
        self, samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss for a batch of samples.

        Args:
            samples: A tuple of tensors (states, actions, rewards, next_states, dones).

        Returns:
            The computed loss.
        """
        states, actions, rewards, next_states, dones = samples
        # print(f"states.shape = {states.shape}")
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].squeeze(1)
        targets = rewards + (~dones) * self.gamma * next_q_values
        return nn.MSELoss()(q_values, targets) / self.batch_size
    
    def _compute_td_errors(self, samples):
        states, actions, rewards, next_states, dones = samples
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (~dones) * self.gamma * next_q_values
        return targets - q_values
    
    def compute_prioritized_loss(
        self, samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss for a batch of samples using prioritized experience replay.

        Args:
            samples: A tuple of tensors (states, actions, rewards, next_states, dones).

        Returns:
            The computed loss.
        """
        weights = self.replay_buffer.last_sampled_weights
        td_errors = self._compute_td_errors(samples)
        loss = (weights * td_errors.pow(2)).mean()
        self.replay_buffer.update_priorities(self.replay_buffer.last_sampled_indices, td_errors.detach())
        return loss

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Get the next action for the agent.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action.
        """
        next_board, reward, done = state_transition
        self.record_eval_data("rewards", reward)

        if len(self.episode_history) > 0:
            self._update_state_transitions_and_replay_buffer(next_board, reward, done)
            if len(self.replay_buffer) >= self.batch_size:
                self._train_network()

        if not done:
            return self._handle_incomplete_game(next_board)
        else:
            self._handle_game_completion()
            return -1

    def _update_state_transitions_and_replay_buffer(self, next_board: Board | None, reward: Reward, done: bool) -> None:
        """
        Update state transitions and add to the replay buffer.

        Args:
            next_board: The next board state.
            reward: The reward received.
            done: Whether the episode is done.
        """
        board, action = self.episode_history[-1]
        self.state_transitions.append((board, action, next_board, reward, done))

        state = self.board_to_state(board)
        next_state = self.board_to_state(next_board) if next_board is not None else self.board_to_state(["X"] * (self.rows**2))
        self.replay_buffer.add(state.squeeze(0), action, reward, next_state.squeeze(0), done)

    def _train_network(self) -> None:
        samples = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_eval_data("loss", loss.item())
        self.maybe_log_metrics()
        self.train_step_count += 1

    def _handle_incomplete_game(self, next_board: Board | None) -> Action:
        """
        Handle the case where the game is not yet complete.

        Args:
            next_board: The next board state.

        Returns:
            The chosen action.
        """
        if next_board is not None:
            action = self.choose_action(next_board, epsilon=self.epsilon)
            self.episode_history.append((next_board, action))
            self.games_moves_count += 1
            return action
        return -1

    def _handle_game_completion(self) -> None:
        """
        Handle the case where the game is complete.
        """
        if self.episode_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_count += 1

        self.episode_count += 1
        if not self.set_exploration_rate_externally:
            self.update_exploration_rate(self.episode_count)
        
        self.episode_history = []

    def board_to_state(self, board: Board) -> np.ndarray:
        return self.state_converter.board_to_state(board)

    def state_to_board(self, state: np.ndarray) -> Board:
        return self.state_converter.state_to_board(state)

    def update_exploration_rate(self, episode: int) -> None:
        """
        Update the exploration rate (epsilon) based on the current episode.

        Args:
            episode: The current episode number.
        """
        epsilon_0 = self.params["epsilon_start"]
        epsilon_min = self.params["epsilon_min"]
        T = self.params["nr_of_episodes"]
        t = episode
        # self.epsilon = max(
        #     epsilon_min,
        #     T * epsilon_0 * epsilon_min / (t * (epsilon_0 - epsilon_min) + T * epsilon_min),
        # )
        delta = (epsilon_0 - epsilon_min) / T
        self.epsilon = max(epsilon_min, epsilon_0 - delta * t)

    def get_valid_actions(self, board: Board):
        """
        Get the list of valid actions for the current board.

        Args:
            board: The board state.

        Returns:
            A list of valid actions.
        """
        return [i for i, cell in enumerate(board) if cell == " "]

    def choose_action(self, board: Board, epsilon: float) -> Action:
        """
        Choose an action based on Q-values.

        Args:
            board: The board state.
            epsilon: The exploration rate.

        Returns:
            The chosen action.
        """
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            action = self.get_best_action(board, self.q_network)
            return action

    def get_best_action(self, board: Board, q_network: nn.Module) -> Action:
        """
        Get the best action based on Q-values.

        Args:
            board: The board state.
            q_network: The Q-network.

        Returns:
            The best action.
        """
        state = self.board_to_state(board)
        state_tensor = torch.FloatTensor(state).to(self.device)
        # print(f"state_tensor.shape = {state_tensor.shape}")
        with torch.no_grad():
            q_values = q_network(state_tensor).squeeze()
            max_q, _ = torch.max(q_values, dim=0)

            self.record_eval_data("action_value", max_q.item())

            max_q_indices = torch.nonzero(q_values == max_q, as_tuple=False)
            if max_q_indices.size(0) > 1:
                action = int(max_q_indices[torch.randint(len(max_q_indices), (1,))].item())
            else:
                action = int(max_q_indices)

        return action

class DeepQPlayingAgent(Agent):
    """
    A Deep Q-Playing agent for playing Tic Tac Toe.
    """

    def __init__(self, 
                q_network: nn.Module | str,
                player: Player = "X",
                switching: bool = False,
                device : str = "cpu",
                state_shape: str = "flat") -> None:
        """
        Initialize the DeepQPlayingAgent.

        Args:
            q_network: The Q-network or path to the saved Q-network.
            player: The player symbol ("X" or "O").
            switching: Whether to switch players after each game.
        """
        super().__init__(player=player, switching=switching)
        self.device = torch.device(device)

        if isinstance(q_network, torch.nn.Module):
            self.q_network: nn.Module = q_network.to(self.device)
        else:
            self.q_network = torch.load(q_network, weights_only=False).to(self.device)  # type: ignore
            self.q_network.eval()

        if state_shape == "flat":
            self.state_converter = FlatStateConverter()
        elif state_shape == "2D":
            self.state_converter = GridStateConverter(shape=(3, 3))  # Assuming a 3x3 grid
        elif state_shape == "one-hot":
            self.state_converter = OneHotStateConverter(rows=3)  # Assuming a 3x3 grid
        else:
            raise ValueError(f"Unsupported state shape: {state_shape}")

    def board_to_state(self, board: Board) -> np.ndarray:
        return self.state_converter.board_to_state(board)

    def state_to_board(self, state: np.ndarray) -> Board:
        return self.state_converter.state_to_board(state)

    def get_valid_actions(self, board: Board) -> Actions:
        """
        Generate all empty positions on the board.

        Args:
            board: The board state.

        Returns:
            A list of valid actions.
        """
        return [i for i, cell in enumerate(board) if cell == " "]

    def choose_action(self, board: Board) -> int:
        """
        Choose an action based on Q-values.

        Args:
            board: The board state.

        Returns:
            The chosen action.
        """
        action = self.get_best_action(board, self.q_network)
        return action

    def get_best_action(self, board: Board, q_network: nn.Module) -> Action:
        """
        Get the best action based on Q-values.

        Args:
            board: The board state.
            q_network: The Q-network.

        Returns:
            The best action.
        """
        state = self.board_to_state(board)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = q_network(state_tensor).squeeze()
            max_q, _ = torch.max(q_values, dim=0)
            max_q_indices = torch.nonzero(q_values == max_q, as_tuple=False)
            if max_q_indices.size(0) > 1:
                action = int(max_q_indices[torch.randint(len(max_q_indices), (1,))].item())
            else:
                action = int(max_q_indices)

        return action

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Get the next action for the agent.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action.
        """
        _, _, done = state_transition
        if not done:
            board = game.get_board()
            action = self.choose_action(board)
            return action
        else:
            self.on_game_end(game)
            return -1

    def on_game_end(self, game: "TwoPlayerBoardGame") -> None:
        """
        Handle the end of the game.

        Args:
            game: The game instance.
        """
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
