import random

from Matrix import Matrix, QMatrix
from SymmetricMatrix import SymmetricMatrix, QSymmetricMatrix
from SymmetricMatrix import TotallySymmetricMatrix, QTotallySymmetricMatrix

class Agent:
    def __init__(self, player):
        players = ['X', 'O']
        self.player = player
        self.opponent = players[1] if player == players[0] else players[0]

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
        raise NotImplementedError("This method should be overridden by subclasses")


class LearningAgent(Agent):
    def __init__(self, player, params):
        super().__init__(player)
        # Initialize matrices

        # # Q = QMatrix(file='Q.pkl')
        # Q = QMatrix(default_value=params['Q_initial_value'])
        # Visits = Matrix(default_value=0)
        # Rewards = Matrix(default_value=0)

        # # Q = QSymmetricMatrix(file='SymmetricQ.pkl')
        # Q = QSymmetricMatrix(default_value=params['Q_initial_value'])
        # Visits = SymmetricMatrix(default_value=0)
        # Rewards = SymmetricMatrix(default_value=0.0)

        # Q = QTotallySymmetricMatrix(file='TotallySymmetricQ.pkl')
        Q = QTotallySymmetricMatrix(default_value=params['Q_initial_value'])
        Visits = TotallySymmetricMatrix(default_value=0)
        Rewards = TotallySymmetricMatrix(default_value=0.0)

        self.Q = Q
        self.Visits = Visits
        self.Rewards = Rewards

        self.params = params
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.alpha = params['alpha_start']

        self.rewards = {}        
        self.rewards[self.player] = params['rewards']['W']
        self.rewards[self.opponent] = params['rewards']['L']
        self.rewards['D'] = params['rewards']['D']

        self.nr_of_episodes = params['nr_of_episodes']
        self.episode = 0

    def get_action(self, game):
        board = game.get_board()
        action = self.choose_action(board, epsilon=self.epsilon)
        return action

    # Choose an action based on Q-values
    def choose_action(self, board, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best known move
            return int(self.Q.best_action(board))

    def notify_result(self, game, outcome):
        history = game.get_history()
        terminal_reward = self.rewards[outcome]
        self.q_update_backward(history, terminal_reward)
        self.episode += 1
        self.update_rates(self.episode)

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
            future_q = max(future_qs)
        else:
            future_q = 0.0

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * future_q)
        self.Q.set(board, action, new_value)
        return abs(old_value - new_value)

    # Update Q-values based on the game's outcome, with correct max_future_q
    def q_update_backward(self, history, terminal_reward):
        diff = 0
        for i in reversed(range(len(history))):
            board, action = history[i]
            if i == len(history) - 1:
                # Update the last state-action pair with the terminal reward
                diff += self.q_update(board, action, None, terminal_reward)
            else:
                next_board, _ = history[i + 1]
                diff += self.q_update(board, action, next_board, 0.0)
            
        return diff


class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions)
    
    def notify_result(self, game, outcome):
        pass


class HumanAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def get_action(self, game):
        valid_actions = game.get_valid_actions()
        action = None
        while action is None:
            user_input = input(f"Choose a number from the set {valid_actions}: ")
            action = int(user_input)
            if action not in valid_actions:
                action = None

        return action
    
    def notify_result(self, game, outcome):
        pass