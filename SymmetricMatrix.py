import dill
import numpy as np
from itertools import product
from collections import defaultdict


class SymmetricMatrix:
    def __init__(self, file=None, default_value=None):
        self.default_value = 0.0
        # Matrix storing canonical state-action pairs
        if default_value is None and file is None:
            self.q_matrix = defaultdict(self._initialize_q_matrix)
        elif default_value is None and file is not None:
            with open(file, 'rb') as f:
                self.q_matrix = dill.load(f)

        elif file is None and default_value is not None:
            self.default_value = default_value
            self.q_matrix = defaultdict(self._initialize_q_matrix)

        # Precompute symmetries and their inverses
        self.transformations = [
            lambda x: x,                         # Identity
            lambda x: np.rot90(x, 1),            # Rotate 90°
            lambda x: np.rot90(x, 2),            # Rotate 180°
            lambda x: np.rot90(x, 3),            # Rotate 270°
            lambda x: np.fliplr(x),              # Horizontal reflection
            lambda x: np.flipud(x),              # Vertical reflection
            lambda x: np.transpose(x),           # Diagonal reflection (TL-BR)
        ]

        self.inverse_transformations = [
            lambda x: x,                         # Identity
            lambda x: np.rot90(x, 3),            # Rotate 90° inverse (rotate 270°)
            lambda x: np.rot90(x, 2),            # Rotate 180° inverse
            lambda x: np.rot90(x, 1),            # Rotate 270° inverse (rotate 90°)
            lambda x: np.fliplr(x),              # Horizontal reflection inverse
            lambda x: np.flipud(x),              # Vertical reflection inverse
            lambda x: np.transpose(x),           # Diagonal reflection (TL-BR) inverse
        ]

        self.original_actions = np.array(range(9)).reshape(3, 3)
        for i, transform in enumerate(self.transformations):
            assert self.inverse_transformations[i](transform(self.original_actions)).flatten().tolist() == self.original_actions.flatten().tolist()

        transformed_actions = [transform(self.original_actions).flatten().tolist() for transform in self.transformations]
        for i in range(len(transformed_actions)):
            for j in range(i + 1, len(transformed_actions)):
                assert transformed_actions[i] != transformed_actions[j]

        # Generate all valid Tic-Tac-Toe boards
        self.all_valid_boards = self._generate_all_valid_boards()
        self.all_canonical_boards = set()
        self.get_canonical_boards = {}
        self.get_transform = {}
        self.get_inverse_transform = {}
        self.get_canonical_actions = {}
        self.get_inverse_canonical_actions = {}
        for board in self.all_valid_boards:
            canonical_board, transform_idx = self._get_canonical_representation(board)
            self.all_canonical_boards.add(canonical_board)
            self.get_canonical_boards[board] = canonical_board
            self.get_transform[board] = self.transformations[transform_idx]
            self.get_inverse_transform[board] = self.inverse_transformations[transform_idx]
            self.get_inverse_canonical_actions[board] = self.get_transform[board](self.original_actions).flatten().tolist()
            self.get_canonical_actions[board] = self.get_inverse_transform[board](self.original_actions).flatten().tolist()

        self.all_canonical_boards = list(self.all_canonical_boards)
        self.all_canonical_actions = {}
        for board in self.all_canonical_boards:
            empty_positions = self.get_empty_positions(board)
            self.all_canonical_actions[board] = empty_positions

    def _initialize_q_matrix(self):
        state_dict = defaultdict(lambda: self.default_value)
        return state_dict    

    def _board_to_matrix(self, board):
        """
        Convert a linear board to a 3x3 matrix for easier manipulation
        """
        return np.array(board).reshape(3, 3)

    def _matrix_to_board(self, matrix):
        """
        Convert a 3x3 matrix back to a linear board representation.
        """
        return matrix.flatten().tolist()

    def _generate_all_valid_boards(self):
        symbols = [' ', 'X', 'O']
        all_boards = list(product(symbols, repeat=9))  # Generate all 3^9 combinations
        # print(f"Total number of boards: {len(all_boards)}")
        all_valid_boards = []

        for board in all_boards:
            x_count = board.count('X')
            o_count = board.count('O')
            
            # Valid boards must satisfy these conditions:
            if x_count == o_count or x_count == o_count + 1:
                all_valid_boards.append(board)

        # print(f"Number of valid boards: {len(all_valid_boards)}")    
        return all_valid_boards

    def _generate_symmetries(self, board):
        matrix = self._board_to_matrix(board)
        symmetries = [transform(matrix) for transform in self.transformations]
        return [self._matrix_to_board(sym) for sym in symmetries]

    def _get_canonical_representation(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry), symmetries.index(min_symmetry)

    def get_canonical_action(self, board, action):
            return self.get_canonical_actions[tuple(board)][action]

    def get_canonical_board(self, board):
            return self.get_canonical_boards[tuple(board)]

    def get_inverse_canonical_action(self, board, canonical_action):
        return self.get_inverse_canonical_actions[tuple(board)][canonical_action]
    
    def canonicalize(self, board, action):
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        return canonical_board, canonical_action

    # Generate all empty positions on the board
    def get_empty_positions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def get(self, board=None, action=None):
        """
        Retrieve the value for a state-action pair.
        """
        if board is None and action is None:
            return self.q_matrix
        if action is None and board is not None:
            return self.q_matrix[self.get_canonical_board(board)]
        if action is not None and board is not None:
            canonical_board, canonical_action = self.canonicalize(board, action)
            return self.q_matrix[canonical_board][canonical_action]

    def set(self, board, action, value):
        """
        Set the value for a state-action pair.
        """
        canonical_board, canonical_action = self.canonicalize(board, action)
        self.q_matrix[canonical_board][canonical_action] = value


class QSymmetricMatrix(SymmetricMatrix):
    def best_actions(self, board, player='X'):
        """
        Choose the best action based on Q-values for the current state.
        """
        actions = self.get_empty_positions(board)

        # Retrieve Q-values for all valid actions
        q_values = {action: self.get(board, action) for action in actions}

        # Choose based on player strategy
        if player == 'X':
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
        else:
            min_q = min(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == min_q]

        return best_actions
    
    def best_action(self, board, player='X'):
        """
        Choose the best action based on Q-values for the current state.
        """
        best_actions = self.best_actions(board, player)
        return np.random.choice(best_actions)
    

class TotallySymmetricMatrix:
    def __init__(self, file=None, default_value=None):
        self.default_value = 0.0
        # Matrix storing canonical state-action pairs
        if default_value is None and file is None:
            self.qMatrix = defaultdict(lambda: self.default_value)
        elif default_value is None and file is not None:
            with open(file, 'rb') as f:
                self.qMatrix = dill.load(f)

        elif file is None and default_value is not None:
            self.default_value = default_value
            self.qMatrix = defaultdict(lambda: self.default_value)

        # Precompute symmetries and their inverses
        self.transformations = [
            lambda x: x,                         # Identity
            lambda x: np.rot90(x, 1),            # Rotate 90°
            lambda x: np.rot90(x, 2),            # Rotate 180°
            lambda x: np.rot90(x, 3),            # Rotate 270°
            lambda x: np.fliplr(x),              # Horizontal reflection
            lambda x: np.flipud(x),              # Vertical reflection
            lambda x: np.transpose(x),           # Diagonal reflection (TL-BR)
        ]

        self.inverse_transformations = [
            lambda x: x,                         # Identity
            lambda x: np.rot90(x, 3),            # Rotate 90° inverse (rotate 270°)
            lambda x: np.rot90(x, 2),            # Rotate 180° inverse
            lambda x: np.rot90(x, 1),            # Rotate 270° inverse (rotate 90°)
            lambda x: np.fliplr(x),              # Horizontal reflection inverse
            lambda x: np.flipud(x),              # Vertical reflection inverse
            lambda x: np.transpose(x),           # Diagonal reflection (TL-BR) inverse
        ]

        self.original_actions = np.array(range(9)).reshape(3, 3)
        for i, transform in enumerate(self.transformations):
            assert self.inverse_transformations[i](transform(self.original_actions)).flatten().tolist() == self.original_actions.flatten().tolist()

        transformed_actions = [transform(self.original_actions).flatten().tolist() for transform in self.transformations]
        for i in range(len(transformed_actions)):
            for j in range(i + 1, len(transformed_actions)):
                assert transformed_actions[i] != transformed_actions[j]

        # Generate all valid Tic-Tac-Toe boards
        self.all_valid_boards = self._generate_all_valid_boards()
        self.all_canonical_boards = set()
        self.get_canonical_boards = {}
        self.get_transform = {}
        self.get_inverse_transform = {}
        self.get_canonical_actions = {}
        self.get_inverse_canonical_actions = {}
        for board in self.all_valid_boards:
            canonical_board, transform_idx = self._get_canonical_representation(board)
            self.all_canonical_boards.add(canonical_board)
            self.get_canonical_boards[board] = canonical_board
            self.get_transform[board] = self.transformations[transform_idx]
            self.get_inverse_transform[board] = self.inverse_transformations[transform_idx]
            self.get_inverse_canonical_actions[board] = self.get_transform[board](self.original_actions).flatten().tolist()
            self.get_canonical_actions[board] = self.get_inverse_transform[board](self.original_actions).flatten().tolist()

        self.all_canonical_boards = list(self.all_canonical_boards)
        self.all_canonical_actions = {}
        for board in self.all_canonical_boards:
            empty_positions = self.get_empty_positions(board)
            self.all_canonical_actions[board] = empty_positions

        self.canonical_board_to_next_canonical_board = {}
        self.all_next_canonical_boards = set()
        for canonical_board in self.all_canonical_boards:
            canonical_actions_to_next_canonical_board = {}
            for canonical_action in self.all_canonical_actions[canonical_board]:
                next_board = self._get_next_board(canonical_board, canonical_action)
                next_canonical_board = self.get_canonical_board(next_board)
                self.all_next_canonical_boards.add(next_canonical_board)
                canonical_actions_to_next_canonical_board[canonical_action] = next_canonical_board

            self.canonical_board_to_next_canonical_board[canonical_board] = canonical_actions_to_next_canonical_board


    def _initialize_q_matrix(self):
        state_dict = defaultdict(lambda: self.default_value)
        return state_dict    

    def _board_to_matrix(self, board):
        """
        Convert a linear board to a 3x3 matrix for easier manipulation
        """
        return np.array(board).reshape(3, 3)

    def _matrix_to_board(self, matrix):
        """
        Convert a 3x3 matrix back to a linear board representation.
        """
        return matrix.flatten().tolist()

    def _generate_all_valid_boards(self):
        symbols = [' ', 'X', 'O']
        all_boards = list(product(symbols, repeat=9))  # Generate all 3^9 combinations
        all_valid_boards = []
        for board in all_boards:
            x_count = board.count('X')
            o_count = board.count('O')
            
            # Valid boards must satisfy these conditions:
            if x_count == o_count or x_count == o_count + 1:
                all_valid_boards.append(board)

        return all_valid_boards

    def _generate_symmetries(self, board):
        matrix = self._board_to_matrix(board)
        symmetries = [transform(matrix) for transform in self.transformations]
        return [self._matrix_to_board(sym) for sym in symmetries]
    
    def _get_next_player(self, board):
        x_count = board.count('X')
        o_count = board.count('O')
        return 'X' if x_count == o_count else 'O'

    def _get_next_board(self, board, action):
        new_board = list(board)
        next_player = self._get_next_player(board)
        if next_player == 'X':
            new_board[action] = 'X'
        else:
            new_board[action] = 'O'
        
        return tuple(new_board)

    def _get_canonical_representation(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry), symmetries.index(min_symmetry)

    def get_canonical_action(self, board, action):
            return self.get_canonical_actions[tuple(board)][action]

    def get_canonical_board(self, board):
            return self.get_canonical_boards[tuple(board)]

    def get_inverse_canonical_action(self, board, canonical_action):
        return self.get_inverse_canonical_actions[tuple(board)][canonical_action]
    
    def canonicalize(self, board, action):
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        return canonical_board, canonical_action

    # Generate all empty positions on the board
    def get_empty_positions(self, board):
        return [i for i, cell in enumerate(board) if cell == ' ']

    def get(self, board=None, action=None):
        if board is None and action is None:
            return self.qMatrix
        if action is None and board is not None:
            actions = self.get_empty_positions(board)
            canonical_actions = [self.get_canonical_action(board, action) for action in actions]
            canonical_board = self.get_canonical_board(board)
            return [self.qMatrix[self.canonical_board_to_next_canonical_board[canonical_board][canonical_action]] for canonical_action in canonical_actions]
        if action is not None and board is not None:
            return self.qMatrix[self.canonical_board_to_next_canonical_board[self.get_canonical_board(board)][self.get_canonical_action(board, action)]]

    def set(self, board, action, value):
        """
        Set the value for a state-action pair.
        """
        self.qMatrix[self.canonical_board_to_next_canonical_board[self.get_canonical_board(board)][self.get_canonical_action(board, action)]] = value


class QTotallySymmetricMatrix(TotallySymmetricMatrix):
    def best_actions(self, board, player='X'):
        """
        Choose the best action based on Q-values for the current state.
        """
        actions = self.get_empty_positions(board)

        # Retrieve Q-values for all valid actions
        q_values = {action: self.get(board, action) for action in actions}

        # Choose based on player strategy
        if player == 'X':
            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
        else:
            min_q = min(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == min_q]

        return best_actions
    
    def best_action(self, board, player='X'):
        """
        Choose the best action based on Q-values for the current state.
        """
        best_actions = self.best_actions(board, player)
        return np.random.choice(best_actions)