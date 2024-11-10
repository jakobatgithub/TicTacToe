import random

from Matrix import Matrix, QMatrix
from SymmetricMatrix import SymmetricMatrix, QSymmetricMatrix
from SymmetricMatrix import TotallySymmetricMatrix, QTotallySymmetricMatrix

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


class QLearningAgent(Agent):
    def __init__(self, params):
        # Initialize matrices
        # self.Q = QMatrix(default_value=params['Q_initial_value'])
        # self.Q = QSymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], width=params['width'])
        self.Q = QTotallySymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], width=params['width'])
        if params['Q_optimal']:
            self.evaluation = True
            self.Q_optimal = QTotallySymmetricMatrix(file=params['Q_optimal'])
        else:
            self.evaluation = False

        self.params = params
        super().__init__(player=params['player'], switching=params['switching'])
        self.set_rewards()
        self.debug = params['debug']
        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")
    
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.alpha = params['alpha_start']

        self.nr_of_episodes = params['nr_of_episodes']
        self.episode = 0

    def is_optimal(self, board, action):
        return action in self.Q_optimal.best_actions(board)

    def initialize(self):
        self.set_rewards()

    def get_action(self, game):
        board = game.get_board()
        action = self.choose_action(board, epsilon=self.epsilon)
        return action

    def notify_result(self, game, outcome):
        total_history = game.get_history()
        if self.player == 'X':
            history = total_history[0::2]
        else:
            history = total_history[1::2]

        terminal_reward = self.rewards[outcome]
        if self.debug:
            print(f"outcome = {outcome}, terminal_reward = {terminal_reward}")
            board, action = history[-1]
            print(f"board = {board}, action = {action}")

        diff = self.q_update_backward(history, terminal_reward)
        if self.evaluation:
            self.params['diffs'].append(diff)

        self.episode += 1
        self.update_rates(self.episode)
        self.params['history'] = history
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
            self.set_rewards()
            if self.debug:
                print(f"Player: {self.player}, opponent: {self.opponent}")

    def set_rewards(self):
        self.rewards = self.params['rewards'][self.player]

    # Choose an action based on Q-values
    def choose_action(self, board, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best known move
            action = int(self.Q.best_action(board))            
            if self.evaluation:
                if self.is_optimal(board, action):
                    self.params['optimal_actions'].append(1)
                else:
                    self.params['optimal_actions'].append(0)

            return action

    # Generate all empty positions on the board
    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']
    
    def update_rates(self, episode):
        self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))
        self.alpha = max(self.params['alpha_min'], self.params['alpha_start'] / (1 + episode/self.nr_of_episodes))

    def q_update(self, board, action, next_board, reward):
        old_value = self.Q.get(board, action)
        if next_board:
            # Calculate max Q-value for the next state over all possible actions
            next_actions = self.get_valid_actions(next_board)
            future_qs = [self.Q.get(next_board, next_action) for next_action in next_actions]
            if self.debug:
                print(f"future_qs = {future_qs}")
            
            future_q = max(future_qs)
        else:
            future_q = 0.0

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_q)
        self.Q.set(board, action, new_value)
        if self.debug:
            print(f"{old_value}, {new_value}, {new_value - old_value}")
    
        return abs(old_value - new_value)

    # Update Q-values based on the game's outcome, with correct max_future_q
    def q_update_backward(self, history, terminal_reward):
        diff = 0
        # print(f"terminal_reward = {terminal_reward}")
        for i in reversed(range(len(history))):
            board, action = history[i]
            if i == len(history) - 1:
                # Update the last state-action pair with the terminal reward
                diff += self.q_update(board, action, None, terminal_reward)
            else:
                next_board, _ = history[i + 1]
                diff += self.q_update(board, action, next_board, 0.0)
            
        return diff/(len(history) * self.alpha)

class QPlayingAgent(Agent):
    def __init__(self, Q, player='X', switching=False):
        super().__init__(player=player, switching=switching)
        self.Q = Q

    def get_action(self, game):
        board = game.get_board()
        action = self.choose_action(board)
        return action
    
    def notify_result(self, game, outcome):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player

    # Choose an action based on Q-values
    def choose_action(self, board):
        return int(self.Q.best_action(board))


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