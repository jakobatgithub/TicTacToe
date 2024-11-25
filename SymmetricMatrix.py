from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import Any, Callable, Tuple

import dill  # type: ignore
import numpy as np

from game_types import Action, Player

Board = Tuple[str, ...]


class LazyComputeDict(dict[Any, Any]):
    def __init__(self, compute_func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compute_func = compute_func

    def __getitem__(self, key: Any):
        if key not in self:
            # Compute and store the value if it doesn't exist
            self[key] = self.compute_func(key)
        return super().__getitem__(key)


class BaseMatrix(ABC):
    """
    Abstract base class for Q-value matrices.
    Defines the shared interface and enforces implementation of key methods.
    """

    def __init__(self, file: str | None = None, default_value: float = 0.0) -> None:
        self.default_value = default_value
        if file:
            with open(file, "rb") as f:
                self.qMatrix: dict[Any, Any] = dill.load(f)  # type: ignore
        else:
            self.qMatrix: dict[Any, Any] = defaultdict(self._initialize_q_matrix)

    def _initialize_q_matrix(self) -> dict[Any, Any]:
        """
        Initialize an empty dictionary for storing Q-values for a given state.
        """
        return defaultdict(lambda: self.default_value)

    @abstractmethod
    def get(self, board: Board, action: Action) -> float:
        """
        Get the Q-value for a state-action pair.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def set(self, board: Board, action: Action, value: float) -> None:
        """
        Set the Q-value for a state-action pair.
        Must be implemented by subclasses.
        """
        pass


class Matrix(BaseMatrix):
    def __init__(self, file: str | None = None, default_value: float | None = None) -> None:
        self.default_value = 0.0
        if default_value is None and file is None:
            self.qMatrix: dict[Any, Any] = defaultdict(self._initialize_q_matrix)
        elif default_value is None and file is not None:
            with open(file, "rb") as f:
                self.qMatrix = dill.load(f)  # type: ignore
        elif file is None and default_value is not None:
            self.default_value = default_value
            self.qMatrix = defaultdict(self._initialize_q_matrix)

    def _initialize_q_matrix(self) -> dict[Any, Any]:
        state_dict: dict[Any, Any] = defaultdict(lambda: self.default_value)
        return state_dict

    def get(self, board: Board, action: Action) -> float:
        return self.qMatrix[board][action]

    def set(self, board: Board, action: Action, value: float) -> None:
        self.qMatrix[tuple(board)][action] = value


class SymmetricMatrix(BaseMatrix):
    def __init__(
        self, file: str | None = None, default_value: float | None = None, lazy: bool = True, rows: int = 3
    ) -> None:
        self.default_value = 0.0
        self.rows = rows

        if default_value is None and file is None:
            self.qMatrix: dict[Any, Any] = defaultdict(self._initialize_q_matrix)
        elif default_value is None and file is not None:
            with open(file, "rb") as f:
                self.qMatrix = dill.load(f)  # type: ignore

        elif file is None and default_value is not None:
            self.default_value = default_value
            self.qMatrix = defaultdict(self._initialize_q_matrix)

        # Precompute symmetries and their inverses
        self.transformations: list[Callable[..., Any]] = [
            lambda x: x,  # type: ignore
            lambda x: np.fliplr(x),  # type: ignore
            lambda x: np.flipud(x),  # type: ignore
            lambda x: np.flipud(np.fliplr(x)),  # type: ignore
            lambda x: np.transpose(x),  # type: ignore
            lambda x: np.fliplr(np.transpose(x)),  # type: ignore
            lambda x: np.flipud(np.transpose(x)),  # type: ignore
            lambda x: np.flipud(np.fliplr(np.transpose(x))),  # type: ignore
        ]

        self.inverse_transformations: list[Callable[..., Any]] = [
            lambda x: x,  # type: ignore
            lambda x: np.fliplr(x),  # type: ignore
            lambda x: np.flipud(x),  # type: ignore
            lambda x: np.fliplr(np.flipud(x)),  # type: ignore
            lambda x: np.transpose(x),  # type: ignore
            lambda x: np.transpose(np.fliplr(x)),  # type: ignore
            lambda x: np.transpose(np.flipud(x)),  # type: ignore
            lambda x: np.transpose(np.fliplr(np.flipud(x))),  # type: ignore
        ]

        self.original_actions = np.array(range(self.rows * self.rows)).reshape(self.rows, self.rows)
        for i, transform in enumerate(self.transformations):
            assert (
                self.inverse_transformations[i](transform(self.original_actions)).flatten().tolist()
                == self.original_actions.flatten().tolist()
            )

        transformed_actions = [
            transform(self.original_actions).flatten().tolist() for transform in self.transformations
        ]
        for i in range(len(transformed_actions)):
            for j in range(i + 1, len(transformed_actions)):
                assert transformed_actions[i] != transformed_actions[j]

        if lazy:
            self.dict_canonical_board = LazyComputeDict(self._get_canonical_board)
            self.dict_transform = LazyComputeDict(self._get_canonical_symmetry_transform)
            self.dict_inverse_transform = LazyComputeDict(self._get_inverse_canonical_symmetry_transform)
            self.dict_canonical_actions = LazyComputeDict(
                lambda board: self.dict_inverse_transform[board](self.original_actions).flatten().tolist()  # type: ignore
            )
            self.dict_inverse_canonical_actions = LazyComputeDict(
                lambda board: self.dict_transform[board](self.original_actions).flatten().tolist()  # type: ignore
            )
        else:
            self._generate_all_dicts()

    def _generate_all_dicts(self) -> None:
        all_valid_boards = self._generate_all_valid_boards()
        self.all_canonical_boards: set[Board] = set()
        self.dict_canonical_board: dict[Board, Board] = {}
        self.dict_transform: dict[Board, Callable[..., Any]] = {}
        self.dict_inverse_transform: dict[Board, Callable[..., Any]] = {}
        self.dict_canonical_actions: dict[Board, list[int]] = {}
        self.dict_inverse_canonical_actions: dict[Board, list[int]] = {}
        for board in all_valid_boards:
            canonical_board, transform_idx = self._get_canonical_representation(board)
            self.all_canonical_boards.add(canonical_board)
            self.dict_canonical_board[board] = canonical_board
            self.dict_transform[board] = self.transformations[transform_idx]
            self.dict_inverse_transform[board] = self.inverse_transformations[transform_idx]
            self.dict_canonical_actions[board] = (
                self.dict_inverse_transform[board](self.original_actions).flatten().tolist()
            )
            self.dict_inverse_canonical_actions[board] = (
                self.dict_transform[board](self.original_actions).flatten().tolist()
            )

        # self.all_canonical_boards = list(self.all_canonical_boards)
        self.all_canonical_actions: dict[Board, list[int]] = {}
        for board in self.all_canonical_boards:
            empty_positions = self.get_empty_positions(board)
            self.all_canonical_actions[board] = empty_positions

    def _initialize_q_matrix(self) -> dict[Any, Any]:
        state_dict: dict[Any, Any] = defaultdict(lambda: self.default_value)
        return state_dict

    def _board_to_matrix(self, board: Board) -> np.ndarray[Any, Any]:
        return np.array(board).reshape(self.rows, self.rows)

    def _matrix_to_board(self, matrix: np.ndarray[Any, Any]) -> Board:
        return matrix.flatten().tolist()

    def _generate_all_valid_boards(self) -> list[Board]:
        symbols = [" ", "X", "O"]
        all_boards: list[Board] = list(
            product(symbols, repeat=self.rows * self.rows)
        )  # Generate all 3^(rows^2) combinations
        # print(f"Total number of boards: {len(all_boards)}")
        all_valid_boards: list[Board] = []

        for board in all_boards:
            x_count: int = board.count("X")
            o_count: int = board.count("O")

            # Valid boards must satisfy these conditions:
            if x_count == o_count or x_count == o_count + 1:
                all_valid_boards.append(board)

        # print(f"Number of valid boards: {len(all_valid_boards)}")
        return all_valid_boards

    def _generate_symmetries(self, board: Board) -> list[Board]:
        matrix = self._board_to_matrix(board)
        symmetries = [transform(matrix) for transform in self.transformations]
        return [self._matrix_to_board(sym) for sym in symmetries]

    def _get_canonical_board(self, board: Board):
        symmetries: list[Board] = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry)

    def _get_canonical_symmetry_transform(self, board: Board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        transform_idx = symmetries.index(min_symmetry)
        return self.transformations[transform_idx]

    def _get_inverse_canonical_symmetry_transform(self, board: Board):
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        transform_idx = symmetries.index(min_symmetry)
        return self.inverse_transformations[transform_idx]

    def _get_canonical_representation(self, board: Board) -> tuple[Board, int]:
        symmetries = self._generate_symmetries(board)
        min_symmetry = min(symmetries)
        return tuple(min_symmetry), symmetries.index(min_symmetry)

    def get_canonical_action(self, board: Board, action: int) -> int:
        return self.dict_canonical_actions[tuple(board)][action]

    def get_canonical_board(self, board: Board) -> Board:
        return self.dict_canonical_board[tuple(board)]

    def get_inverse_canonical_action(self, board: Board, canonical_action: int) -> int:
        return self.dict_inverse_canonical_actions[tuple(board)][canonical_action]

    def canonicalize(self, board: Board, action: int) -> tuple[Board, Action]:
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        return canonical_board, canonical_action

    # Generate all empty positions on the board
    def get_empty_positions(self, board: Board):
        return [i for i, cell in enumerate(board) if cell == " "]

    def get(self, board: Board, action: Action) -> float:
        canonical_board, canonical_action = self.canonicalize(board, action)
        return self.qMatrix[canonical_board][canonical_action]

    def set(self, board: Board, action: int, value: float) -> None:
        canonical_board, canonical_action = self.canonicalize(board, action)
        self.qMatrix[canonical_board][canonical_action] = value


class FullySymmetricMatrix(SymmetricMatrix):
    def __init__(
        self, file: str | None = None, default_value: float | None = None, lazy: bool = True, rows: int = 3
    ) -> None:
        # Call the parent initializer first
        super().__init__(file=file, default_value=default_value, lazy=lazy, rows=rows)

        # Override qMatrix initialization for TotallySymmetricMatrix
        if file is not None:
            with open(file, "rb") as f:
                self.qMatrix: dict[Any, Any] = dill.load(f)  # type: ignore
        else:
            self.qMatrix: dict[Any, Any] = defaultdict(lambda: self.default_value)

        # Additional attributes specific to TotallySymmetricMatrix
        if lazy:
            self.canonical_board_to_next_canonical_board = self._create_level2_lazy_dict(self._get_next_canonical_board)
        else:
            self._generate_all_next_canonical_boards()

    def _create_level2_lazy_dict(self, compute_func: Callable[..., Any]):
        def level1_compute(outer_key: Any):
            return LazyComputeDict(lambda inner_key: compute_func(outer_key, inner_key))  # type: ignore

        return LazyComputeDict(level1_compute)

    def _get_next_canonical_board(self, canonical_board: Board, canonical_action: Action) -> Board:
        next_board = self._get_next_board(canonical_board, canonical_action)
        next_canonical_board = self.get_canonical_board(next_board)
        return next_canonical_board

    def _generate_all_next_canonical_boards(self) -> None:
        # Generate canonical board-to-next canonical board mapping
        all_next_canonical_boards: set[Board] = set()
        self.canonical_board_to_next_canonical_board: dict[Board, dict[Action, Board]] = {}
        for canonical_board in self.all_canonical_boards:
            canonical_actions_to_next_canonical_board: dict[Action, Board] = {}
            for canonical_action in self.all_canonical_actions[canonical_board]:
                next_board = self._get_next_board(canonical_board, canonical_action)
                next_canonical_board = self.get_canonical_board(next_board)
                all_next_canonical_boards.add(next_canonical_board)
                canonical_actions_to_next_canonical_board[canonical_action] = next_canonical_board

            self.canonical_board_to_next_canonical_board[canonical_board] = canonical_actions_to_next_canonical_board

    def _get_next_board(self, board: Board, action: Action) -> Board:
        new_board = list(board)
        next_player = self._get_next_player(board)
        new_board[action] = next_player
        return tuple(new_board)

    def _get_next_player(self, board: Board) -> Player:
        x_count = board.count("X")
        o_count = board.count("O")
        return "X" if x_count == o_count else "O"

    def get(self, board: Board, action: Action) -> float:
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        return self.qMatrix[self.canonical_board_to_next_canonical_board[canonical_board][canonical_action]]

    def set(self, board: Board, action: int, value: float) -> None:
        canonical_board = self.get_canonical_board(board)
        canonical_action = self.get_canonical_action(board, action)
        self.qMatrix[self.canonical_board_to_next_canonical_board[canonical_board][canonical_action]] = value
