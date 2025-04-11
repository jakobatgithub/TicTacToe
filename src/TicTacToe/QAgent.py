import random
from typing import TYPE_CHECKING, Any

import numpy as np

from TicTacToe.Agent import Agent
from TicTacToe.SymmetricMatrix import BaseMatrix, Matrix  # SymmetricMatrix, FullySymmetricMatrix

if TYPE_CHECKING:
    from TicTacToe.TicTacToe import TwoPlayerBoardGame  # Import only for type hinting

from TicTacToe.game_types import (
    Action,
    Actions,
    Board,
    History,
    Params,
    Player,
    Reward,
    StateTransition,
    StateTransitions2,
)


class QLearningAgent(Agent):
    """
    An agent that uses Q-learning to play Tic-Tac-Toe.
    """

    def __init__(self, params: Params) -> None:
        """
        Initialize the QLearningAgent.

        Args:
            params: A dictionary of parameters for the agent.
        """
        super().__init__(player=params["player"], switching=params["switching"])
        self.params = params
        self.debug = params["debug"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon_start"]
        self.alpha = params["alpha_start"]
        self.nr_of_episodes = params["nr_of_episodes"]
        self.terminal_q_updates = params["terminal_q_updates"]

        # Initialize matrices
        self.Q: BaseMatrix = Matrix(default_value=params["Q_initial_value"])
        # self.Q: BaseMatrix = SymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], rows=params['rows'])
        # self.Q: BaseMatrix = FullySymmetricMatrix(
        #     default_value=params["Q_initial_value"], lazy=params["lazy_evaluation"], rows=params["rows"]
        # )

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0

        self.episode_history: History = []
        self.state_transitions: StateTransitions2 = []

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")

        self.evaluation_data: dict[str, Any] = {"loss": [], "action_value": [], "histories": [], "rewards": []}

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Decide the next action based on the current state transition.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action.
        """
        next_board, reward, done = state_transition
        self.evaluation_data["rewards"].append(reward)
        if len(self.episode_history) > 0:
            board, action = self.episode_history[-1]
            self.state_transitions.append((board, action, next_board, reward, done))
            if not self.terminal_q_updates:
                _, _ = self.q_update(board, action, next_board, reward)

        if not done:
            board = next_board
            if board is not None:
                action = self.choose_action(board, epsilon=self.epsilon)
                self.episode_history.append((board, action))
                self.games_moves_count += 1
                return action
            else:
                return -1
        else:
            if self.terminal_q_updates:
                _, _ = self.q_update_backward(self.episode_history, reward)

            self.episode_count += 1
            self.update_rates(self.episode_count)
            self.evaluation_data["histories"].append(self.episode_history)
            self.episode_history = []
            return -1

    def q_update(self, board: Board, action: Action, next_board: Board | None, reward: Reward) -> tuple[float, float]:
        """
        Update the Q-value for a state-action pair.

        Args:
            board: The current board state.
            action: The action taken.
            next_board: The next board state.
            reward: The reward received.

        Returns:
            A tuple containing the loss and action value.
        """
        old_value = self.Q.get(tuple(board), action)
        if next_board is not None:
            # Calculate max Q-value for the next state over all possible actions
            next_actions = self.get_valid_actions(next_board)
            future_qs_data = [self.Q.get(tuple(next_board), next_action) for next_action in next_actions]
            future_qs = [x for x in future_qs_data if isinstance(x, float)]
            if self.debug:
                print(f"future_qs = {future_qs}")

            future_q = max(future_qs)
        else:
            future_q = 0.0

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_q)
        self.Q.set(tuple(board), action, new_value)
        if self.debug:
            print(f"{old_value}, {new_value}, {new_value - old_value}")

        self.q_update_count += 1
        loss = float(abs(old_value - new_value))
        action_value = future_q
        self.evaluation_data["loss"].append(loss / self.alpha)
        self.evaluation_data["action_value"].append(action_value)
        return loss, action_value

    def q_update_backward(self, history: History, terminal_reward: float) -> tuple[float, float]:
        """
        Update Q-values based on the game's outcome, with correct max_future_q.

        Args:
            history: The history of the game.
            terminal_reward: The reward at the terminal state.

        Returns:
            A tuple containing the average loss and action value.
        """
        avg_loss = 0
        action_value = 0
        for i in reversed(range(len(history))):
            board, action = history[i]
            if i == len(history) - 1:
                # Update the last state-action pair with the terminal reward
                loss, action_value = self.q_update(board, action, None, terminal_reward)
                avg_loss += loss
                action_value += action_value
            else:
                next_board, _ = history[i + 1]
                loss, action_value = self.q_update(board, action, next_board, 0.0)
                avg_loss += loss
                action_value += action_value

        self.train_step_count += 1
        return (avg_loss / (len(history) * self.alpha), action_value / (len(history)))

    def update_rates(self, episode: int) -> None:
        """
        Update the exploration and learning rates.

        Args:
            episode: The current episode number.
        """
        self.epsilon = max(
            self.params["epsilon_min"], self.params["epsilon_start"] / (1 + episode / self.nr_of_episodes)
        )
        self.alpha = max(self.params["alpha_min"], self.params["alpha_start"] / (1 + episode / self.nr_of_episodes))

    def get_valid_actions(self, board: Board) -> Actions:
        """
        Get all valid actions for the current board state.

        Args:
            board: The current board state.

        Returns:
            A list of valid actions.
        """
        return [i for i, cell in enumerate(board) if cell == " "]

    def get_best_actions(self, board: Board, Q: BaseMatrix) -> Actions:
        """
        Get the best actions based on Q-values.

        Args:
            board: The current board state.
            Q: The Q-value matrix.

        Returns:
            A list of the best actions.
        """
        actions = self.get_valid_actions(board)
        q_values = {action: Q.get(tuple(board), action) for action in actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return best_actions

    def get_best_action(self, board: Board, Q: BaseMatrix) -> Action:
        """
        Get the best action based on Q-values.

        Args:
            board: The current board state.
            Q: The Q-value matrix.

        Returns:
            The best action.
        """
        best_actions = self.get_best_actions(board, Q)
        return np.random.choice(best_actions)

    def choose_action(self, board: Board, epsilon: float) -> Action:
        """
        Choose an action based on Q-values and exploration rate.

        Args:
            board: The current board state.
            epsilon: The exploration rate.

        Returns:
            The chosen action.
        """
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best known move
            action = int(self.get_best_action(board, self.Q))
            return action


class QPlayingAgent(Agent):
    """
    An agent that plays Tic-Tac-Toe using a pre-trained Q-value matrix.
    """

    def __init__(self, Q: BaseMatrix, player: Player = "X", switching: bool = False) -> None:
        """
        Initialize the QPlayingAgent.

        Args:
            Q: The Q-value matrix.
            player: The player's symbol.
            switching: Whether the agent switches players after each game.
        """
        super().__init__(player=player, switching=switching)
        self.Q: BaseMatrix = Q

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Decide the next action based on the current state transition.

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
            self.on_game_end()
            return -1

    def get_valid_actions(self, board: Board) -> Actions:
        """
        Get all valid actions for the current board state.

        Args:
            board: The current board state.

        Returns:
            A list of valid actions.
        """
        return [i for i, cell in enumerate(board) if cell == " "]

    def get_best_actions(self, board: Board, Q: BaseMatrix) -> Actions:
        """
        Get the best actions based on Q-values.

        Args:
            board: The current board state.
            Q: The Q-value matrix.

        Returns:
            A list of the best actions.
        """
        actions = self.get_valid_actions(board)
        q_values = {action: Q.get(tuple(board), action) for action in actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return best_actions

    def get_best_action(self, board: Board, Q: BaseMatrix) -> Action:
        """
        Get the best action based on Q-values.

        Args:
            board: The current board state.
            Q: The Q-value matrix.

        Returns:
            The best action.
        """
        best_actions = self.get_best_actions(board, Q)
        return np.random.choice(best_actions)

    def choose_action(self, board: Board) -> Action:
        """
        Choose an action based on Q-values.

        Args:
            board: The current board state.

        Returns:
            The chosen action.
        """
        action = int(self.get_best_action(board, self.Q))
        return action

    def on_game_end(self) -> None:
        """
        Handle the end of the game.
        """
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
