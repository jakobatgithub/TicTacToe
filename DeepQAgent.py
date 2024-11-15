import random

import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models

import torch
import torch.nn as nn
import torch.optim as optim

from Agent import Agent


# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Neural Network for Q-function
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DeepQLearningAgent(Agent):
    def __init__(self, params):
        super().__init__(player=params['player'], switching=params['switching'])
        self.params = params
        self.debug = params['debug']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_start']
        self.alpha = params['alpha_start']
        self.nr_of_episodes = params['nr_of_episodes']
        self.terminal_q_updates = params['terminal_q_updates']

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0

        self.episode_history = []
        self.state_transitions = []
        (state_size, action_size) = (3*3, 3*3)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)

        self.state_to_board_translation = {'X': 1, 'O': -1, ' ': 0}
        self.board_to_state_translation = {1: 'X', -1: 'O', 0: ' '}

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
            state = self.board_to_state(board)
            if next_board is None:
                next_state = self.board_to_state(['X'] * 9) # is not needed
            else:
                next_state = self.board_to_state(next_board)

            # print(f"state = {state}")
            # print(f"next_state = {next_state}")
            self.replay_buffer.add((state, action, reward, next_state, done))
            if not self.terminal_q_updates:
                # loss, action_value = self.q_update(board, action, next_board, reward)

                if len(self.replay_buffer) >= BATCH_SIZE:
                    experiences = self.replay_buffer.sample(BATCH_SIZE)
                    # for (board1, action1, next_board1, reward1, done1) in experiences:
                    #     loss, action_value = self.q_update(board1, action1, next_board1, reward1)

                    states, actions, rewards, next_states, dones = zip(*experiences)
                    # print(f"states = {states[0:2]}")
                    # print(f"next_states = {next_states[0:2]}")

                    states = torch.FloatTensor(np.array(states))
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    q_values = self.q_network(states).gather(2, actions.unsqueeze(2)).squeeze(2)

                    # next_q_values = self.target_network(next_states).max(2, keepdim=True)[0].squeeze(2)
                    next_q_values = self.q_network(next_states).max(2, keepdim=True)[0].squeeze(2)

                    # print(f"next_q_values.shape = {next_q_values.shape}")

                    targets = rewards + (1 - dones) * self.gamma * next_q_values

                    # print(f"q_values.shape = {q_values.shape}")
                    # print(f"targets.shape = {targets.shape}")

                    # print(f"q_values = {q_values[0:2]}")
                    # print(f"targets = {targets[0:2]}")

                    loss = nn.MSELoss()(q_values, targets)
                    self.evaluation_data['loss'].append(loss.item())
                    self.evaluation_data['action_value'].append(next_q_values.mean().item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

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

    def board_to_state(self, board):
        return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)
    
    def state_to_board(self, state):
        flat_state = state.flatten()
        board = [self.board_to_state_translation[cell] for cell in flat_state]
        return board

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
            state = self.board_to_state(board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.q_network(state_tensor)).item()

            return action


# class DeepQLearningAgent(Agent):
#     def __init__(self, params):
#         super().__init__(player=params['player'], switching=params['switching'])
#         self.params = params
#         self.debug = params['debug']
#         self.evaluation = params['evaluation']
#         self.gamma = params['gamma']
#         self.epsilon = params['epsilon_start']
#         self.alpha = params['alpha_start']
#         self.nr_of_episodes = params['nr_of_episodes']
#         self.double_q_learning = params['double_q_learning']

#         self.Qmodel = models.Sequential([
#             layers.Input(shape=(9,)),
#             layers.Dense(16, activation='relu'),
#             layers.Dense(16, activation='relu'),
#             layers.Dense(9)  # Q-values for 9 actions
#         ])
#         lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate=self.alpha,
#             decay_steps=self.nr_of_episodes,
#             decay_rate=0.95)
#         # self.Qmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
#         self.Qmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
#         self.Qmodel.summary()
#         self.target_model = tf.keras.models.clone_model(self.Qmodel)
#         self.target_model.set_weights(self.Qmodel.get_weights())

#         self.replay_buffer = deque(maxlen=10000)
#         self.batch_size = params['batch_size']
#         self.state_to_board_translation = {'X': 1, 'O': -1, ' ': 0}
#         self.board_to_state_translation = {1: 'X', -1: 'O', 0: ' '}

#         self.target_update_frequency = params['target_update_frequency']

#         self.episode_count = 0
#         self.games_moves_count = 0
#         self.train_step_count = 0
#         self.q_update_count = 0
#         self.target_update_count = 0

#         self.board = None
#         self.next_board = None

#         if self.debug:
#             print(f"Player: {self.player}, opponent: {self.opponent}")
#             self.verbose_level = 2 # verbose level for tensorflow
#         else:
#             self.verbose_level = 0

#         self.evaluation_data = {'loss': [], 'action_value': [], 'histories' : [], 'rewards': []}

#     # Generate all empty positions on the board
#     def get_valid_actions(self, board):
#         return [i for i, cell in enumerate(board) if cell == ' ']

#     def board_to_state(self, board):
#         return np.array([self.state_to_board_translation[cell] for cell in board]).reshape(1, -1)
    
#     def state_to_board(self, state):
#         flat_state = state.flatten()
#         board = [self.board_to_state_translation[cell] for cell in flat_state]
#         return board
    
#     def update_rates(self, episode):
#         self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))
    
#     def mask_invalid_actions(self, q_values, valid_actions):
#         masked_q_values = np.full_like(q_values, -np.inf)  # Initialize with -inf
#         masked_q_values[valid_actions] = q_values[valid_actions]  # Keep only valid actions
#         return masked_q_values

#     def choose_action(self, board, epsilon):
#         if random.uniform(0, 1) < epsilon:
#             # Exploration: Choose a random move
#             valid_actions = self.get_valid_actions(board)
#             return random.choice(valid_actions)
#         else:
#             valid_actions = self.get_valid_actions(board)
#             state = self.board_to_state(board)

#             q_values = self.Qmodel.predict(state, verbose=self.verbose_level)
#             q_values = q_values[0]
#             if self.debug:
#                 print(f"valid_actions = {valid_actions}, state = {state}, q_values = {q_values}")
    
#             masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
#             action = np.argmax(masked_q_values)
#             return action

#     def sample_experiences(self, batch_size):
#         return random.sample(self.replay_buffer, batch_size)

#     def store_experience(self, state, action, reward, next_state, done):
#         if self.debug:
#             print(f"state = {(state, action, reward, next_state, done)}")
    
#         self.replay_buffer.append((state, action, reward, next_state, done))

#     def train_step(self, batch_size, gamma=0.99):
#         if len(self.replay_buffer) < batch_size:
#             return (None, None)
        
#         batch = self.sample_experiences(batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = np.array([state[0] for state in states])
#         next_states = np.array([next_state[0] for next_state in next_states])
#         rewards = np.array(rewards)
#         dones = np.array(dones)

#         if self.debug:
#             print(f"states = {states}")
#             print(f"actions = {actions}")
#             print(f"rewards = {rewards}")
#             print(f"next_states = {next_states}")
#             print(f"dones = {dones}")

#         all_valid_actions = [self.get_valid_actions(self.state_to_board(state)) for state in states]
#         q_values = self.Qmodel.predict(states, verbose=self.verbose_level)
#         next_target_q_values = self.target_model.predict(next_states, verbose=self.verbose_level)
#         if self.double_q_learning:
#             next_q_values = self.Qmodel.predict(next_states, verbose=self.verbose_level)

#         if self.debug:
#             print(f"q_values = {q_values}")
#             print(f"next_q_values = {next_target_q_values}")
        
#         action_value = 0
#         for i, action in enumerate(actions):
#             # Mask invalid actions in next_q_values
#             masked_next_target_q_values = [next_target_q_values[i][a] if a in all_valid_actions[i] else -float('inf') for a in range(len(next_target_q_values[i]))]
#             if self.double_q_learning:
#                 masked_next_q_values = [next_q_values[i][a] if a in all_valid_actions[i] else -float('inf') for a in range(len(next_q_values[i]))]
#                 q_max_predicted = masked_next_q_values[np.argmax(masked_next_target_q_values)]
#             else:
#                 q_max_predicted = np.max(masked_next_target_q_values)

#             action_value += q_max_predicted
#             if not dones[i]:
#                 target = rewards[i] + gamma * q_max_predicted
#             else:
#                 target = rewards[i]
        
#             q_values[i][action] = target
#             self.q_update_count += 1

#         history = self.Qmodel.fit(states, q_values, epochs=1, verbose=self.verbose_level)
#         self.train_step_count += 1
#         loss = history.history['loss'][0]
#         action_value /= len(states)
#         return (loss, action_value)

#     def get_action(self, state_transition, game):
#         if self.next_board is not None:
#             self.board = self.next_board
        
#         self.next_board, reward, done = state_transition
#         if self.board is not None and self.action is not None:
#             self.store_experience(self.board_to_state(self.board), self.action, reward, self.board_to_state(self.next_board), done)
        
#         self.action = None
#         if not done:
#             board = self.next_board
#             self.action = self.choose_action(board, epsilon=self.epsilon)
#             (loss, action_value) = self.train_step(self.batch_size, self.gamma)
#             self.games_moves_count += 1

#             # Update target network
#             if self.train_step_count % self.target_update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
#                 self.target_model.set_weights(self.Qmodel.get_weights())
#                 self.target_update_count += 1
#                 print(f"target_update_count = {self.target_update_count}, train_step_count = {self.train_step_count}, "
#                     f"episode_count = {self.episode_count}, games_moves_count = {self.games_moves_count}, q_update_count = {self.q_update_count}")

#             if self.evaluation:
#                 if loss:
#                     self.evaluation_data['loss'].append(loss)
#                 if action_value:
#                     self.evaluation_data['action_value'].append(action_value)

#             return self.action
#         else:
#             self.on_game_end(game, reward)
#             self.action = None
#             self.board = None
#             self.next_board = None
#             return None

#     def on_game_end(self, game, reward):
#         terminal_reward = reward
#         self.update_rates(self.episode_count)
#         self.episode_count += 1

#         if self.debug:
#             print(f"terminal_reward = {terminal_reward}")

#         if self.evaluation:
#             self.evaluation_data['rewards'].append(terminal_reward)

#         if self.switching:
#             self.player, self.opponent = self.opponent, self.player
#             if self.debug:
#                 print(f"Player: {self.player}, opponent: {self.opponent}")


class DeepQPlayingAgent(Agent):
    def __init__(self, q_network, player='X', switching=False):
        super().__init__(player=player, switching=switching)
        self.q_network = q_network
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

    def mask_invalid_actions(self, q_values, valid_actions):
        masked_q_values = [q_values[valid_action] for valid_action in valid_actions]  # Keep only valid actions
        return masked_q_values

    def choose_action(self, board):
        # Exploitation: Choose the best known move
        state = self.board_to_state(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        valid_actions = self.get_valid_actions(board)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze().tolist()
        
        masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
        action = valid_actions[np.argmax(masked_q_values)]
        return action
    
    def get_action(self, state_transition, game):
        state, reward, done = state_transition
        if not done:
            board = game.get_board()
            action = self.choose_action(board)
            return action
        else:
            self.on_game_end(game)
            return None

    def on_game_end(self, game):
        if self.switching:
            self.player, self.opponent = self.opponent, self.player