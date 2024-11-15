import random
from abc import ABC, abstractmethod

class Agent(ABC):
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
                user_input = input(f"Choose a number from the set {valid_actions}: ")
                action = int(user_input)
                if action not in valid_actions:
                    action = None

            return action
        else:
            self.on_game_end(game)
            return None
        
    def on_game_end(self, game):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player