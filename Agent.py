import random

class Agent:
    def __init__(self, player='X', switching=False):
        """
        Base class for all agents.
        :param player: 'X', 'O', or None (to be assigned later).
        """
        self.players = ['X', 'O']
        self.player = player
        self.opponent = self.get_opponent(player)
        self.switching = switching

    def get_opponent(self, player):
        return self.players[1] if player == self.players[0] else self.players[0]

    def get_action(self, game):
        """
        Decides the next action based on the current game state.
        :param game: An instance of the Tic-Tac-Toe game.
        :return: A tuple (row, col) representing the agent's move.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def notify_result(self, game, outcome):
        """
        Notifies the agent about the result of the game.
        :param result: A string, e.g., "win", "loss", or "draw".
        :param outcome: The symbol ('X' or 'O' or 'D') of the outcome.
        """
        pass


class RandomAgent(Agent):
    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)
    
    def notify_result(self, game, outcome):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player


class HumanAgent(Agent):
    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        action = None
        while action is None:
            user_input = input(f"Choose a number from the set {valid_actions}: ")
            action = int(user_input)
            if action not in valid_actions:
                action = None

        return action