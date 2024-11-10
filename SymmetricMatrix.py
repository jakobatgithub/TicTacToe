import dill
import numpy as np
from itertools import product
from collections import defaultdict


class LazyComputeDict(dict):
    def __init__(self, compute_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_func = compute_func
    
    def __getitem__(self, key):
        if key not in self:
            # Compute and store the value if it doesn't exist
            self[key] = self.compute_func(key)
        return super().__getitem__(key)


class SymmetricMatrix:
    def __init__(self, file=None, default_value=None, lazy=True):
        self.default_value = 0.0
        if default_value is None and file is None:
            self.qMatrix = defaultdict(self._initialize_q_matrix)
        elif default_value is None and file is not None:
            with open(file, 'rb') as f:
                self.qMatrix = dill.load(f)

        elif file is None and default_value is not None:
            self.default_value = default_value
            self.qMatrix = defaultdict(self._initialize_q_matrix)

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

        if lazy:
            self.dict_canonical_board = LazyComputeDict(self._get_canonical_board)
            self.dict_transform = LazyComputeDict(self._get_canonical_symmetry_transform)
            self.dict_inverse_transform = LazyComputeDict(self._get_inverse_canonical_symmetry_transform)
            self.dict_canonical_actions = LazyComputeDict(lambda board : self.dict_inverse_transform[board](self.original_actions).flatten().tolist())
            self.dict_inverse_canonical_actions = LazyComputeDict(lambda board : self.dict_transform[board](self.original_actions).flatten().tolist())
        else:
            self._generate_all_dicts()


    def _generate_all_dicts(self):
        all_valid_boards = self._generate_all_valid_boards()
        all_canonical_boards = set()
        self.dict_canonical_board = {}
        self.dict_transform = {}
        self.dict_inverse_transform = {}
        self.dict_canonical_actions = {}
        self.dict_inverse_canonical_actions = {}
        for board in all_valid_boards:
            canonical_board, transform_idx = self._get_canonical_representation(board)
            all_canonical_boards.add(canonical_board)
            self.dict_canonical_board[board] = canonical_board
            self.dict_transform[board] = self.transformations[transform_idx]
            self.dict_inverse_transform[board] = self.inverse_transformations[transform_idx]
            self.dict_canonical_actions[board] = self.dict_inverse_transform[board](self.original_actions).flatten().tolist()
            self.dict_inverse_canonical_actions[board] = self.dict_transform[board](self.original_actions).flatten().tolist()

        all_canonical_boards = list(all_canonical_boards)
        self.all_canonical_actions = {}
        for board in all_canonical_boards:
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

    def _get_canonical_board(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry)

    def _get_canonical_symmetry_transform(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        transform_idx = symmetries.index(min_symmetry)
        return self.transformations[transform_idx]

    def _get_inverse_canonical_symmetry_transform(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        transform_idx = symmetries.index(min_symmetry)
        return self.inverse_transformations[transform_idx]

    def _get_canonical_representation(self, board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry), symmetries.index(min_symmetry)

    def get_canonical_action(self, board, action):
            return self.dict_canonical_actions[tuple(board)][action]

    def get_canonical_board(self, board):
            return self.dict_canonical_board[tuple(board)]

    def get_inverse_canonical_action(self, board, canonical_action):
        return self.dict_inverse_canonical_actions[tuple(board)][canonical_action]
    
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
            return self.qMatrix
        if action is None and board is not None:
            return self.qMatrix[self.get_canonical_board(board)]
        if action is not None and board is not None:
            canonical_board, canonical_action = self.canonicalize(board, action)
            return self.qMatrix[canonical_board][canonical_action]

    def set(self, board, action, value):
        """
        Set the value for a state-action pair.
        """
        canonical_board, canonical_action = self.canonicalize(board, action)
        self.qMatrix[canonical_board][canonical_action] = value


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


class TotallySymmetricMatrix(SymmetricMatrix):
    def __init__(self, file=None, default_value=None, lazy=True):
        # Call the parent initializer first
        super().__init__(file=file, default_value=default_value, lazy=lazy)

        # Override q_matrix initialization for TotallySymmetricMatrix
        if file is not None:
            with open(file, 'rb') as f:
                self.qMatrix = dill.load(f)
        else:
            self.qMatrix = defaultdict(lambda: self.default_value)

        # Additional attributes specific to TotallySymmetricMatrix
        if lazy:
            self.canonical_board_to_next_canonical_board = self._create_level2_lazy_dict(self._get_next_canonical_board)
        else:
            self._generate_all_next_canonical_boards()

    def _create_level2_lazy_dict(self, compute_func):
        def level1_compute(outer_key):
            return LazyComputeDict(lambda inner_key: compute_func(outer_key, inner_key))
        
        return LazyComputeDict(level1_compute)

    def _get_next_canonical_board(self, canonical_board, canonical_action):
        next_board = self._get_next_board(canonical_board, canonical_action)
        next_canonical_board = self.get_canonical_board(next_board)
        return next_canonical_board

    def _generate_all_next_canonical_boards(self):
        # Generate canonical board-to-next canonical board mapping
        all_next_canonical_boards = set()
        self.canonical_board_to_next_canonical_board = {}
        for canonical_board in self.all_canonical_boards:
            canonical_actions_to_next_canonical_board = {}
            for canonical_action in self.all_canonical_actions[canonical_board]:
                next_board = self._get_next_board(canonical_board, canonical_action)
                next_canonical_board = self.get_canonical_board(next_board)
                all_next_canonical_boards.add(next_canonical_board)
                canonical_actions_to_next_canonical_board[canonical_action] = next_canonical_board

            self.canonical_board_to_next_canonical_board[canonical_board] = canonical_actions_to_next_canonical_board

    def _get_next_board(self, board, action):
        new_board = list(board)
        next_player = self._get_next_player(board)
        new_board[action] = next_player
        return tuple(new_board)

    def _get_next_player(self, board):
        x_count = board.count('X')
        o_count = board.count('O')
        return 'X' if x_count == o_count else 'O'

    def get(self, board=None, action=None):
        """
        Retrieve the value for a state-action pair.
        """
        if board is None and action is None:
            return self.qMatrix
        canonical_board = self.get_canonical_board(board)
        if action is None:
            actions = self.get_empty_positions(board)
            return [
                self.qMatrix[
                    self.canonical_board_to_next_canonical_board[canonical_board][
                        self.get_canonical_action(board, action)
                    ]
                ]
                for action in actions
            ]
        canonical_action = self.get_canonical_action(board, action)
        return self.qMatrix[
            self.canonical_board_to_next_canonical_board[canonical_board][canonical_action]
        ]

    def set(self, board, action, value):
        """
        Set the value for a state-action pair.
        """
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        self.qMatrix[
            self.canonical_board_to_next_canonical_board[canonical_board][canonical_action]
        ] = value


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