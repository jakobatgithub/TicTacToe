import random

import numpy as np

import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from Agent import Agent


class ReplayBuffer:
    def __init__(self, size, state_dim, device='cpu'):
        """
        Initialize the ReplayBuffer with a fixed size and GPU storage.

        :param size: Maximum number of experiences to store in the buffer.
        :param state_dim: Dimension of the state tensor.
        :param device: The device to store the buffer on ("cuda" for GPU or "cpu").
        """
        self.size = size
        self.current_size = 0  # Tracks the current number of stored experiences
        self.index = 0         # Tracks the insertion index for circular overwriting
        self.device = device

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state observed.
        :param done: Whether the episode is done.
        """
        self.states[self.index] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.index] = torch.tensor(action, dtype=torch.int64, device=self.device)
        self.rewards[self.index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.index] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.index] = torch.tensor(done, dtype=torch.bool, device=self.device)

        # Update the index and current size
        self.index = (self.index + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer directly on the GPU.

        :param batch_size: Number of experiences to sample.
        :return: A tuple of sampled tensors (states, actions, rewards, next_states, dones).
        """
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        """
        Get the current size of the buffer.

        :return: The number of experiences currently stored in the buffer.
        """
        return self.current_size


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
        self.nr_of_episodes = params['nr_of_episodes']
        self.batch_size = params['batch_size']
        self.target_update_frequency = params['target_update_frequency']
        self.learning_rate = params['learning_rate']
        self.replay_buffer_length = params['replay_buffer_length']

        self.episode_count = 0
        self.games_moves_count = 0
        self.train_step_count = 0
        self.q_update_count = 0
        self.target_update_count = 0

        wandb.init(config=params)

        self.episode_history = []
        self.state_transitions = []
        self.rows = params['rows']
        (state_size, action_size) = (self.rows ** 2, self.rows ** 2)
        self.device = torch.device(params['device'])
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_length, self.rows ** 2, device=self.device)

        self.board_to_state_translation = {'X': 1, 'O': -1, ' ': 0}
        self.state_to_board_translation = {1: 'X', -1: 'O', 0: ' '}

        if self.debug:
            print(f"Player: {self.player}, opponent: {self.opponent}")

        self.evaluation_data = {'loss': [], 'action_value': [], 'histories' : [], 'rewards': [], 'valid_actions': []}

    def get_action(self, state_transition, game):
        next_board, reward, done = state_transition
        self.evaluation_data['rewards'].append(reward)
        if len(self.episode_history) > 0:
            board, action = self.episode_history[-1]
            self.state_transitions.append((board, action, next_board, reward, done))
            state = self.board_to_state(board)
            if next_board is None:
                next_state = self.board_to_state(['X'] * (self.rows ** 2)) # is not needed
            else:
                next_state = self.board_to_state(next_board)

            self.replay_buffer.add(state, action, reward, next_state, done)
            if len(self.replay_buffer) >= self.batch_size:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].squeeze(1)
                targets = rewards + (~dones) * self.gamma * next_q_values
                
                # print(f"states.shape = {states.shape}")
                # print(f"q_values.shape = {q_values.shape}")
                # print(f"next_q_values.shape = {next_q_values.shape}")
                # print(f"(~dones).shape = {(~dones).shape}")
                # print(f"rewards.shape = {rewards.shape}")
                # print(f"targets.shape = {targets.shape}")

                loss = nn.MSELoss()(q_values, targets)
                self.evaluation_data['loss'].append(loss.item())
                self.evaluation_data['action_value'].append(next_q_values.mean().item())
                # Buffer Wandb Metrics
                if self.train_step_count % 50 == 0:
                    wandb.log({
                        "loss": sum(self.evaluation_data['loss'][-10:]) / 10,
                        "action_value": sum(self.evaluation_data['action_value'][-10:]) / 10
                    })
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_step_count += 1

        if not done:
            board = next_board
            action = self.choose_action(board, epsilon=self.epsilon)
            self.episode_history.append((board, action))
            self.games_moves_count += 1
            return action
        else:
            # Update target network
            if self.episode_count % self.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.target_update_count += 1

            self.episode_count += 1
            self.update_rates(self.episode_count)
            self.evaluation_data['histories'].append(self.episode_history)    
            self.episode_history = []
            return None

    def board_to_state(self, board):
        return np.array([self.board_to_state_translation[cell] for cell in board]).reshape(1, -1)
    
    def state_to_board(self, state):
        flat_state = state.flatten()
        board = [self.state_to_board_translation[cell] for cell in flat_state]
        return board

    def update_rates(self, episode):
        self.epsilon = max(self.params['epsilon_min'], self.params['epsilon_start'] / (1 + episode/self.nr_of_episodes))

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
            state = self.board_to_state(board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = torch.argmax(self.q_network(state_tensor)).item()

            if action in self.get_valid_actions(board):
                self.evaluation_data['valid_actions'].append(1)
                # wandb.log({"valid_actions": 1})
            else:
                self.evaluation_data['valid_actions'].append(0)
                # wandb.log({"valid_actions": 0})

            return action


class DeepQPlayingAgent(Agent):
    def __init__(self, q_network, player='X', switching=False):
        super().__init__(player=player, switching=switching)
        # self.device = torch.device('mps')
        self.device = torch.device('cpu')
        if isinstance(q_network, torch.nn.Module):
            self.q_network = q_network.to(self.device)
        elif isinstance(q_network, str):
            self.q_network = torch.load(q_network).to(self.device)
            self.q_network.eval()
        
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        valid_actions = self.get_valid_actions(board)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze().tolist()
        
        # masked_q_values = self.mask_invalid_actions(q_values, valid_actions)
        # action = valid_actions[np.argmax(masked_q_values)]
        action = np.argmax(q_values)
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