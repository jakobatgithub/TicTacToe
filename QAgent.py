import random
import numpy as np
from Agent import Agent
from SymmetricMatrix import Matrix, SymmetricMatrix, TotallySymmetricMatrix


class QLearningAgent(Agent):
    def __init__(self, params):
        super().__init__(player=params['player'], switching=params['switching'])
        self.params = params
        self.debug = params['debug']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.alpha = params['alpha_start']
        self.nr_of_episodes = params['nr_of_episodes']
        self.terminal_q_updates = params['terminal_q_updates']

        # Initialize matrices
        # self.Q = Matrix(default_value=params['Q_initial_value'])
        # self.Q = SymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], rows=params['rows'])
        self.Q = TotallySymmetricMatrix(default_value=params['Q_initial_value'], lazy=params['lazy_evaluation'], rows=params['rows'])

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0

        self.episode_history = []
        self.state_transitions = []

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")

        self.evaluation_data = {'loss': [], 'action_value': [], 'histories' : [], 'rewards': []}

    def get_action(self, state_transition, game):
        next_board, reward, done = state_transition
        self.evaluation_data['rewards'].append(reward)
        BATCH_SIZE = 16
        if len(self.episode_history) > 0:
            board, action = self.episode_history[-1]
            self.state_transitions.append((board, action, next_board, reward, done))
            if not self.terminal_q_updates:
                loss, action_value = self.q_update(board, action, next_board, reward)

        if not done:
            board = next_board
            action = self.choose_action(board, epsilon=self.epsilon)
            self.episode_history.append((board, action))
            self.games_moves_count += 1
            return action
        else:
            if self.terminal_q_updates:
                loss, action_value = self.q_update_backward(self.episode_history, reward)

            self.episode_count += 1
            self.update_rates(self.episode_count)
            self.evaluation_data['histories'].append(self.episode_history)
            self.episode_history = []
            return None
        
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
        loss = abs(old_value - new_value)
        action_value = future_q
        self.evaluation_data['loss'].append(loss / self.alpha)
        self.evaluation_data['action_value'].append(action_value)
        return loss, action_value
        
    # Update Q-values based on the game's outcome, with correct max_future_q
    def q_update_backward(self, history, terminal_reward):
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
                loss, action_value =  self.q_update(board, action, next_board, 0.0)
                avg_loss += loss
                action_value += action_value
            
        self.train_step_count += 1
        return (avg_loss/(len(history) * self.alpha), action_value/(len(history)))

    def update_rates(self, episode):
        self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))
        self.alpha = max(self.params['alpha_min'], self.params['alpha_start'] / (1 + episode/self.nr_of_episodes))

    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']
    
    def get_best_actions(self, board, Q):
        actions = self.get_valid_actions(board)
        q_values = {action: Q.get(board, action) for action in actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return best_actions
    
    def get_best_action(self, board, Q):
        best_actions = self.get_best_actions(board, Q)
        return np.random.choice(best_actions)

    # Choose an action based on Q-values
    def choose_action(self, board, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            # Exploitation: Choose the best known move
            action = int(self.get_best_action(board, self.Q))
            return action


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
            self.on_game_end()
            return None

    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def get_best_actions(self, board, Q):
        actions = self.get_valid_actions(board)
        q_values = {action: Q.get(board, action) for action in actions}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return best_actions
    
    def get_best_action(self, board, Q):
        best_actions = self.get_best_actions(board, Q)
        return np.random.choice(best_actions)

    # Choose an action based on Q-values
    def choose_action(self, board):
        action = int(self.get_best_action(board, self.Q))
        return action
    
    def on_game_end(self):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
