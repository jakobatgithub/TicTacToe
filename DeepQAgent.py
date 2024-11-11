import random

import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models

from Agent import Agent

class DeepQLearningAgent(Agent):
    def __init__(self, params):
        super().__init__(player=params['player'], switching=params['switching'])
        self.Qmodel = models.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(9)  # Q-values for 9 actions
        ])
        self.Qmodel.compile(optimizer='adam', loss='mse')
        self.Qmodel.summary()
        self.target_model = tf.keras.models.clone_model(self.Qmodel)
        self.target_model.set_weights(self.Qmodel.get_weights())

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = params['batch_size']
        self.state_to_board_translation = {'X': 1, 'O': -1, ' ': 0}
        self.board_to_state_translation = {1: 'X', -1: 'O', 0: ' '}

        self.target_update_frequency = params['target_update_frequency']

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
        self.target_update_count = 0

        self.set_rewards()

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")
            self.verbose_level = 2 # verbose level for tensorflow
        else:
            self.verbose_level = 0

        self.evaluation_data = {'loss': [], 'avg_action_value': [], 'histories' : [], 'rewards': []}


    def set_rewards(self):
        self.rewards = self.params['rewards'][self.player]

    # Generate all empty positions on the board
    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def board_to_state(self, board):
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)
    
    def state_to_board(self, state):
        flat_state = state.flatten()
        board = [self.board_to_state_translation[cell] for cell in flat_state]
        return board
    
    def update_rates(self, episode):
        self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))
        self.alpha = max(self.params['alpha_min'], self.params['alpha_start'] / (1 + episode/self.nr_of_episodes))
    
    def mask_invalid_actions(self, q_values, valid_actions):
        masked_q_values = np.full_like(q_values, -np.inf)  # Initialize with -inf
        masked_q_values[valid_actions] = q_values[valid_actions]  # Keep only valid actions
        return masked_q_values

    def choose_action(self, board, epsilon):
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random move
            valid_actions = self.get_valid_actions(board)
            return random.choice(valid_actions)
        else:
            valid_actions = self.get_valid_actions(board)
            state = self.board_to_state(board)

            q_values = self.Qmodel.predict(state, verbose=self.verbose_level)
            q_values = q_values[0]
            if self.debug:
                print(f"valid_actions = {valid_actions}, state = {state}, q_values = {q_values}")
    
            masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
            action = np.argmax(masked_q_values)
            return action

    def sample_experiences(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return (None, None)
        
        batch = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array([state[0] for state in states])
        next_states = np.array([next_state[0] for next_state in next_states])
        rewards = np.array(rewards)
        dones = np.array(dones)

        if self.debug:
            print(f"states = {states}")
            print(f"actions = {actions}")
            print(f"rewards = {rewards}")
            print(f"next_states = {next_states}")
            print(f"dones = {dones}")

        all_valid_actions = [self.get_valid_actions(self.state_to_board(state)) for state in states]
        q_values = self.Qmodel.predict(states, verbose=self.verbose_level)
        # next_q_values = self.Qmodel.predict(next_states, verbose=self.verbose_level)
        next_q_values = self.target_model.predict(next_states, verbose=self.verbose_level)

        if self.debug:
            print(f"q_values = {q_values}")
            print(f"next_q_values = {next_q_values}")
        
        avg_action_value = 0
        for i, action in enumerate(actions):
            # Mask invalid actions in next_q_values
            masked_next_q_values = [next_q_values[i][a] if a in all_valid_actions[i] else -float('inf') for a in range(len(next_q_values[i]))]
            q_max_predicted = np.max(masked_next_q_values)
            avg_action_value += q_max_predicted
            if not dones[i]:
                target = rewards[i] + gamma * q_max_predicted
            else:
                target = rewards[i]
        
            q_values[i][action] = target
            self.q_update_count += 1

        history = self.Qmodel.fit(states, q_values, epochs=1, verbose=self.verbose_level)
        self.train_step_count += 1
        loss = history.history['loss'][0]
        avg_action_value /= len(states)
        return (loss, avg_action_value)

    def get_next_board(self, board, action):
        new_board = list(board)
        new_board[action] = self.player
        return tuple(new_board)

    def store_history(self, history, terminal_reward):
        for i in range(len(history)):
            board, action = history[i]
            state = self.board_to_state(board)
            next_board = self.get_next_board(board, action)
            next_state = self.board_to_state(next_board)
            if i == len(history) - 1:
                reward = terminal_reward
                done = True
            else:
                reward = 0.0
                done = False

            self.store_experience(state, action, reward, next_state, done)

    def get_action(self, game):
        board = game.get_board()
        action = self.choose_action(board, epsilon=self.epsilon)
        (loss, avg_action_value) = self.train_step(self.batch_size, self.gamma)
        self.games_moves_count += 1

        # Update target network
        if self.train_step_count % self.target_update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self.target_model.set_weights(self.Qmodel.get_weights())
            self.target_update_count += 1
            print(f"target_update_count = {self.target_update_count}, train_step_count = {self.train_step_count}, "
                  f"episode_count = {self.episode_count}, games_moves_count = {self.games_moves_count}, q_update_count = {self.q_update_count}")

        if self.evaluation:
            if loss:
                self.evaluation_data['loss'].append(loss)
            if avg_action_value:
                self.evaluation_data['avg_action_value'].append(avg_action_value)

        return action

    def notify_result(self, game, outcome):
        total_history = game.get_history()
        if self.player == 'X':
            history = total_history[0::2]
        else:
            history = total_history[1::2]

        terminal_reward = self.rewards[outcome]
        self.store_history(history, terminal_reward)
        self.update_rates(self.episode_count)
        self.episode_count += 1

        if self.debug:
            board, action = history[-1]
            print(f"outcome = {outcome}, terminal_reward = {terminal_reward}")
            board, action = history[-1]
            print(f"board = {board}, action = {action}")

        if self.evaluation:
            self.evaluation_data['histories'].append(history)
            self.evaluation_data['rewards'].append(terminal_reward)             

        if self.switching:
            self.player, self.opponent = self.opponent, self.player
            self.set_rewards()
            if self.debug:
                print(f"Player: {self.player}, opponent: {self.opponent}")


class DeepQPlayingAgent(Agent):
    def __init__(self, Qmodel, player='X', switching=False):
        super().__init__(player=player, switching=switching)
        self.Qmodel = Qmodel
        self.verbose_level = 0 # verbose level for tensorflow

        self.state_to_board_translation = {'X': 1, 'O': -1, ' ': 0}
        board_to_state_translation = {}
        for key, value in self.state_to_board_translation.items():
            board_to_state_translation[value] = key

    def board_to_state(self, board):
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)
    
    def state_to_board(self, state):
        flat_state = state.flatten()
        board = [self.board_to_state_translation[cell] for cell in flat_state]
        return board

    # Generate all empty positions on the board
    def get_valid_actions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']
    
    def mask_invalid_actions(self, q_values, valid_actions):
        masked_q_values = np.full_like(q_values, -np.inf)  # Initialize with -inf
        masked_q_values[valid_actions] = q_values[valid_actions]  # Keep only valid actions
        return masked_q_values

    def choose_action(self, board):
        valid_actions = self.get_valid_actions(board)
        state = self.board_to_state(board)
        q_values = self.Qmodel.predict(state, verbose=self.verbose_level)
        q_values = q_values[0]
        masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
        action = np.argmax(masked_q_values)
        return action
    
    def get_action(self, game):
        board = game.get_board()
        action = self.choose_action(board)
        return action
    
    def notify_result(self, game, outcome):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player