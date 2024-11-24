import random
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, player="X", switching=False):
        """
        Base class for all agents.
        :param player: 'X', 'O', or None (to be assigned later).
        """
        self.players = ["X", "O"]
        self.player = player
        self.opponent = self.get_opponent(player)
        self.switching = switching

    def get_opponent(self, player):
        return self.players[1] if player == self.players[0] else self.players[0]

    @abstractmethod
    def get_action(self, state_transition, game):
        """
        Decides the next action based on the current game state.
        :param game: An instance of the Tic-Tac-Toe game.
        :return: A tuple (row, col) representing the agent's move.
        """
        pass


class RandomAgent(Agent):
    def get_action(self, state_transition, game):
        state, reward, done = state_transition
        if not done:
            valid_actions = game.get_valid_actions()
            return random.choice(valid_actions)
        else:
            self.on_game_end(game)
            return None

    def on_game_end(self, game):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class HumanAgent(Agent):
    def get_action(self, state_transition, game):
        state, reward, done = state_transition
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
            return None

    def on_game_end(self, game):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class MouseAgent(Agent):
    def __init__(self, player="X"):
        super().__init__(player)
        self.selected_action = None  # Stores the clicked (row, col) action

    def get_action(self, state_transition, game):
        """
        Waits for a mouse click and returns the corresponding (row, col) position.
        :param state_transition: Not used for a human player.
        :param game: The game instance (TicTacToeDisplay) to monitor for input.
        :return: A tuple (row, col) representing the selected move.
        """
        self.selected_action = None
        game.display.bind_click_handler(self.handle_click)  # Bind the click handler
        game.display.wait_for_player_action()  # Wait for the user to select an action
        return self.selected_action

    def handle_click(self, action):
        """
        Handles the mouse click on the board.
        :param row: The row of the clicked cell.
        :param col: The column of the clicked cell.
        """
        self.selected_action = action
