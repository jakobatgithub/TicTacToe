import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TicTacToe.TicTacToe import TwoPlayerBoardGame  # Import only for type hinting

from TicTacToe.Display import ScreenDisplay
from TicTacToe.game_types import Action, Player, Players, StateTransition


class Agent(ABC):
    """
    Abstract base class for all agents in the Tic-Tac-Toe game.
    """

    def __init__(self, player: Player = "X", switching: bool = False) -> None:
        """
        Initialize the Agent.

        Args:
            player: The player symbol ('X' or 'O').
            switching: Whether the agent switches players after each game.
        """
        self.player: Player = player
        self.players: Players = ["X", "O"]
        self.opponent: Player = self.get_opponent(player)
        self.switching = switching

    def get_opponent(self, player: Player) -> Player:
        """
        Get the opponent of the given player.

        Args:
            player: The player symbol.

        Returns:
            The opponent's symbol.
        """
        return self.players[1] if player == self.players[0] else self.players[0]

    @abstractmethod
    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Decide the next action based on the current game state.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action.
        """
        pass


class RandomAgent(Agent):
    """
    An agent that selects actions randomly from the set of valid actions.
    """

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Select a random valid action.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action or -1 if the game is over.
        """
        _, _, done = state_transition
        if not done:
            valid_actions = game.get_valid_actions()
            return random.choice(valid_actions)
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


class HumanAgent(Agent):
    """
    An agent that allows a human player to input actions.
    """

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Prompt the human player to select an action.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action or -1 if the game is over.
        """
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
        """
        Handle the end of the game.

        Args:
            game: The game instance.
        """
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class MouseAgent(Agent):
    """
    An agent that allows a human player to select actions using a mouse.
    """

    def __init__(self, player: Player = "X") -> None:
        """
        Initialize the MouseAgent.

        Args:
            player: The player symbol ('X' or 'O').
        """
        super().__init__(player)
        self.selected_action = None  # Stores the clicked action

    def get_action(self, state_transition: StateTransition, game: "TwoPlayerBoardGame") -> Action:
        """
        Wait for a mouse click and return the corresponding position.

        Args:
            state_transition: The current state transition.
            game: The game instance.

        Returns:
            The chosen action or -1 if the game is over.
        """
        self.selected_action = -1
        if isinstance(game.display, ScreenDisplay):
            game.display.bind_click_handler(self.handle_click)  # Bind the click handler
            game.display.wait_for_player_action()  # Wait for the user to select an action

        return self.selected_action

    def handle_click(self, action: Action) -> None:
        """
        Handle the mouse click on the board.

        Args:
            action: The action corresponding to the clicked cell.
        """
        self.selected_action = action
