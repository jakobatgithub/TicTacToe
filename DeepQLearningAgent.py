import random

import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models

from Agent import QLearningAgent
from Matrix import QMatrix


class DeepQLearningAgent(QLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.Q = QMatrix(default_value=params['Q_initial_value'])

        self.model = models.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(9)  # Q-values for 9 actions
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = params['batch_size']
        self.state_to_board_translation = {'X': 1, 'O': -1, ' ': 0}
        self.board_to_state_translation = {1: 'X', -1: 'O', 0: ' '}


    def sample_experiences(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def board_to_state(self, board):
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)
    
    def state_to_board(self, state):
        flat_state = state.flatten()
        board = [self.board_to_state_translation[cell] for cell in flat_state]
        return board
    
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

            q_values = self.model.predict(state)
            q_values = q_values[0]
            if self.debug:
                print(f"valid_actions = {valid_actions}, state = {state}, q_values = {q_values}")
    
            masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
            action = np.argmax(masked_q_values)
            return action

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return None
        
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
        q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        if self.debug:
            print(f"q_values = {q_values}")
            print(f"next_q_values = {next_q_values}")
        
        for i, action in enumerate(actions):
            # Mask invalid actions in next_q_values
            masked_next_q_values = [next_q_values[i][a] if a in all_valid_actions[i] else -float('inf') for a in range(len(next_q_values[i]))]

            if not dones[i]:
                target = rewards[i] + gamma * np.max(masked_next_q_values)
            else:
                target = rewards[i]
        
            q_values[i][action] = target

        if self.debug:
            history = self.model.fit(states, q_values, epochs=1, verbose=2)
        else:
            history = self.model.fit(states, q_values, epochs=1, verbose='none')

        loss = history.history['loss'][0]
        return loss

    def get_next_board(self, board, action):
        new_board = list(board)
        new_board[action] = self.player
        return tuple(new_board)

    def notify_result(self, game, outcome):
        total_history = game.get_history()
        if self.player == 'X':
            history = total_history[0::2]
        else:
            history = total_history[1::2]

        for i in range(len(history)):
            board, action = history[i]
            state = self.board_to_state(board)
            next_board = self.get_next_board(board, action)
            next_state = self.board_to_state(next_board)
            if i == len(history) - 1:
                reward = self.rewards[outcome]
                done = True
            else:
                reward = 0.0
                done = False

            self.store_experience(state, action, reward, next_state, done)

        if self.debug:
            board, action = history[-1]
            terminal_reward = self.rewards[outcome]
            print(f"outcome = {outcome}, terminal_reward = {terminal_reward}")
            board, action = history[-1]
            print(f"board = {board}, action = {action}")

        diff = self.train_step(self.batch_size, self.gamma)
        if self.evaluation:
            if diff:
                self.params['diffs'].append(diff)

        self.episode += 1
        self.update_rates(self.episode)
        self.params['history'] = history
        if self.switching:
            self.player, self.opponent = self.opponent, self.player
            self.set_rewards()
            if self.debug:
                print(f"Player: {self.player}, opponent: {self.opponent}")
