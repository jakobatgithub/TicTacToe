import random

class Agent:
    def get_action(self, game):
        """
        Decides the next action based on the current game state.
        :param game: An instance of the Tic-Tac-Toe game.
        :return: A tuple (row, col) representing the agent's move.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class RandomAgent(Agent):
    def get_action(self, game):
        valid_moves = game.get_valid_actions()  # Assume this method exists in your Tic-Tac-Toe class
        return random.choice(valid_moves)
    

class HumanAgent(Agent):
    def get_action(self, game):
        # print("Current Board:")
        # print(game.display_board())  # Assume this method prints the board
        valid_moves = game.get_valid_actions()
        action = None
        while action is None:
            user_input = input(f"Choose a number from the set {valid_moves}: ")
            action = int(user_input)
            if action not in valid_moves:
                action = None

        return action