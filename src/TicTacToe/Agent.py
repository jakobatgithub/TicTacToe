import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TicTacToe.TicTacToe import TwoPlayerBoardGame  # Import only for type hinting

from TicTacToe.Display import TicTacToeDisplay
from TicTacToe.game_types import Action, Player, Players, StateTransition


class Agent(ABC):
    def __init__(self, player: Player = "X", switching: bool = False) -> None:
        """
        Base class for all agents.
        :param player: 'X', 'O', or None (to be assigned later).
        """
        self.player: Player = player
        self.players: Players = ["X", "O"]
        self.opponent: Player = self.get_opponent(player)
        self.switching = switching

    def get_opponent(self, player: Player) -> Player:
        return self.players[1] if player == self.players[0] else self.players[0]

    @abstractmethod
    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Decides the next action based on the current game state.
        :param game: An instance of the Tic-Tac-Toe game.
        :return: A tuple (row, col) representing the agent's move.
        """
        pass


class RandomAgent(Agent):
    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        _, _, done = state_transition
        if not done:
            valid_actions = game.get_valid_actions()
            return random.choice(valid_actions)
        else:
            self.on_game_end(game)
            return -1

    def on_game_end(self, game: "TwoPlayerBoardGame") -> None:
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class HumanAgent(Agent):
    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        _, _, done = state_transition
        if not done:
            valid_actions = game.get_valid_actions()
            action = None
            while action is None:
                try:
                    user_input = input(f"Choose a number from the set {valid_actions}: ")
                    action = int(user_input)
                    if action not in valid_actions:
                        print(f"Invalid choice. Please choose a number from {valid_actions}.")
                        action = None
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
                    action = None

            return action
        else:
            self.on_game_end(game)
            return -1

    def on_game_end(self, game: "TwoPlayerBoardGame") -> None:
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class MouseAgent(Agent):
    def __init__(self, player: Player = "X") -> None:
        super().__init__(player)
        self.selected_action = None  # Stores the clicked action

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Waits for a mouse click and returns the corresponding position.
        :param state_transition: Not used for a human player.
        :param game: The game instance (TicTacToeDisplay) to monitor for input.
        :return: An integer representing the selected move.
        """
        self.selected_action = -1
        if isinstance(game.display, TicTacToeDisplay):
            game.display.bind_click_handler(self.handle_click)  # Bind the click handler
            game.display.wait_for_player_action()  # Wait for the user to select an action

        return self.selected_action

    def handle_click(self, action: Action) -> None:
        """
        Handles the mouse click on the board.
        :param row: The row of the clicked cell.
        :param col: The column of the clicked cell.
        """
        self.selected_action = action
