import random

from Agent import Agent
from Matrix import Matrix, QMatrix
from SymmetricMatrix import SymmetricMatrix, QSymmetricMatrix
from SymmetricMatrix import TotallySymmetricMatrix, QTotallySymmetricMatrix


class QLearningAgent(Agent):
    def __init__(self, params):
        super().__init__(player=params['player'], switching=params['switching'])

        # Initialize matrices
        # self.Q = QMatrix(default_value=params['Q_initial_value'])
        self.Q = QSymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], width=params['width'])
        # self.Q = QTotallySymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], width=params['width'])

        self.params = params
        self.debug = params['debug']
        self.evaluation = params['evaluation']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.alpha = params['alpha_start']
        self.nr_of_episodes = params['nr_of_episodes']

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")

        self.evaluation_data = {'loss': [], 'avg_action_value': [], 'histories' : [], 'rewards': []}

    def is_optimal(self, board, action):
        return action in self.Q_optimal.best_actions(board)

    def get_action(self, state_transition, game):
        state, reward , done = state_transition
        board = state
        if not done:
            action = self.choose_action(board, epsilon=self.epsilon)
            self.games_moves_count += 1
            return action
        else:
            self.on_game_end(game, reward)
            return None

    def update_rates(self, episode):
        self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))
        self.alpha = max(self.params['alpha_min'], self.params['alpha_start'] / (1 + episode/self.nr_of_episodes))

    def on_game_end(self, game, reward):
        total_history = game.get_history()
        if self.player == 'X':
            history = total_history[0::2]
        else:
            history = total_history[1::2]

        terminal_reward = reward
        if self.debug:
            print(f"terminal_reward = {terminal_reward}")
            board, action = history[-1]
            print(f"board = {board}, action = {action}")

        (loss, avg_action_value) = self.q_update_backward(history, terminal_reward)
        if self.evaluation:
            self.evaluation_data['loss'].append(loss)
            self.evaluation_data['avg_action_value'].append(avg_action_value)
            self.evaluation_data['histories'] = history
            self.evaluation_data['rewards'].append(terminal_reward)

        self.episode_count += 1
        self.update_rates(self.episode_count)
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
            if self.debug:
                print(f"Player: {self.player}, opponent: {self.opponent}")

    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    # Choose an action based on Q-values
    def choose_action(self, board, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best known move
            action = int(self.Q.best_action(board))            
            return action
    
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
    
        self.q_update_count += 1
        return (abs(old_value - new_value), future_q)

    # Update Q-values based on the game's outcome, with correct max_future_q
    def q_update_backward(self, history, terminal_reward):
        avg_loss = 0
        avg_action_value = 0
        for i in reversed(range(len(history))):
            board, action = history[i]
            if i == len(history) - 1:
                # Update the last state-action pair with the terminal reward
                loss, action_value = self.q_update(board, action, None, terminal_reward)
                avg_loss += loss
                avg_action_value += action_value
            else:
                next_board, _ = history[i + 1]
                loss, action_value =  self.q_update(board, action, next_board, 0.0)
                avg_loss += loss
                avg_action_value += action_value
            
        self.train_step_count += 1
        return (avg_loss/(len(history) * self.alpha), avg_action_value/(len(history)))

class QPlayingAgent(Agent):
    def __init__(self, Q, player='X', switching=False):
        super().__init__(player=player, switching=switching)
        self.Q = Q

    def get_action(self, state_transition, game):
        state, reward, done = state_transition
        if not done:
            board = game.get_board()
            action = self.choose_action(board)
            return action
        else:
            self.on_game_end(game)
            return None
    
    # Choose an action based on Q-values
    def choose_action(self, board):
        return int(self.Q.best_action(board))
    
    def on_game_end(self, game):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
