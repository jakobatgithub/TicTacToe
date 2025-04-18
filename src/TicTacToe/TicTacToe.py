from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from TicTacToe.Agent import Agent, MouseAgent
from TicTacToe.Display import Display, ScreenDisplay
from TicTacToe.game_types import Action, Actions, Board, History, Outcome, Reward, StateTransition


class TwoPlayerBoardGame(ABC):
    def __init__(
        self,
        agent1: Agent,
        agent2: Agent,
        display: Optional[Display] = None,
        waiting_time: float = 1.0,
        rows: int = 3,
        cols: int = 3,
    ) -> None:
        self.display = display
        self._waiting_time = waiting_time
        self._rows = rows
        self._cols = cols
        self._agent1 = agent1
        self._agent2 = agent2
        self._validate_display_and_agents()
        self.board = self.initialize_board(rows, cols)
        self.current_player = self._agent1.player
        self._history: History = []
        self._done: bool = False
        self._invalid_move: bool = False
        if self._agent1.player == self._agent2.player:
            raise ValueError("Players must be different")
        self._initialize()

    def _validate_display_and_agents(self) -> None:
        if isinstance(self._agent1, MouseAgent) or isinstance(self._agent2, MouseAgent):
            if not isinstance(self.display, ScreenDisplay):
                raise ValueError("MouseAgent can only be used with TicTacToeDisplay")

    @abstractmethod
    def _initialize(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def initialize_board(rows: int, cols: int) -> Board:
        pass

    @abstractmethod
    def make_move(self, action: Action) -> bool:
        pass

    @abstractmethod
    def get_board(self) -> Board:
        pass

    @abstractmethod
    def switch_player(self) -> None:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def get_outcome(self) -> Outcome:
        pass

    @abstractmethod
    def display_board(self, board: Board, outcome: Outcome = None) -> None:
        pass

    @abstractmethod
    def get_terminal_rewards(self, outcome: Outcome) -> Tuple[Reward, Reward]:
        pass

    def set_rows_and_cols(self, rows: int, cols: int) -> None:
        self._rows = rows
        self._cols = cols

    @abstractmethod
    def get_valid_actions(self) -> Actions:
        pass

    def play(self) -> Outcome:
        self._initialize()
        while not self.is_game_over():
            self.display_board(self.board)

            if self.current_player == self._agent1.player:
                state_transition: StateTransition = (self.board[:], 0.0, False)
                action = self._agent1.get_action(state_transition, self)
            else:
                state_transition: StateTransition = (self.board[:], 0.0, False)
                action = self._agent2.get_action(state_transition, self)

            self._history.append((self.board[:], action))
            if self.make_move(action):
                self.switch_player()

        outcome = self.get_outcome()
        self.display_board(self.board, outcome)
        rewards = self.get_terminal_rewards(outcome)
        self._agent1.get_action((None, rewards[0], True), self)
        self._agent2.get_action((None, rewards[1], True), self)

        return outcome


class TicTacToe(TwoPlayerBoardGame):
    def __init__(
        self,
        agent1: Agent,
        agent2: Agent,
        params: dict,
        display: Optional[Display] = None,
        waiting_time: float = 1.0
    ) -> None:
        rows = params["rows"]
        cols = params["rows"]
        if rows != cols:
            raise ValueError("Tic Tac Toe board must be square")
        self._win_length = params["win_length"]
        self.periodic = params["periodic"]
        self.rewards = params["rewards"]
        super().__init__(agent1, agent2, display, waiting_time, rows, cols)

    def _initialize(self) -> None:
        self.board = self.initialize_board(self._rows, self._cols)
        self.current_player = "X"
        self._history = []
        self._done = False
        self._invalid_move = False
        if not self.periodic:
            self.win_conditions = self._generate_win_conditions()
        else:
            self.win_conditions = self._generate_periodic_win_conditions()

    @staticmethod
    def initialize_board(rows: int, cols: int) -> Board:
        return [" " for _ in range(rows * cols)]

    def _generate_win_conditions(self) -> List[List[int]]:
        rows, cols, win_length = self._rows, self._cols, self._win_length
        conditions: List[List[int]] = []

        for row in range(rows):
            for col in range(cols - win_length + 1):
                conditions.append([row * cols + col + i for i in range(win_length)])

        for col in range(cols):
            for row in range(rows - win_length + 1):
                conditions.append([col + (row + i) * cols for i in range(win_length)])

        for row in range(rows - win_length + 1):
            for col in range(cols - win_length + 1):
                conditions.append([(row + i) * cols + col + i for i in range(win_length)])

        for row in range(rows - win_length + 1):
            for col in range(win_length - 1, cols):
                conditions.append([(row + i) * cols + col - i for i in range(win_length)])

        return conditions

    def _generate_periodic_win_conditions(self) -> List[List[int]]:
        rows, cols, win_length = self._rows, self._cols, self._win_length
        conditions: List[List[int]] = []

        # Horizontal win conditions with periodic boundary
        for row in range(rows):
            for col in range(cols):
                conditions.append([(row * cols + (col + i) % cols) for i in range(win_length)])

        # Vertical win conditions with periodic boundary
        for col in range(cols):
            for row in range(rows):
                conditions.append([((row + i) % rows * cols + col) for i in range(win_length)])

        # Diagonal (top-left to bottom-right) win conditions with periodic boundary
        for row in range(rows):
            for col in range(cols):
                conditions.append([((row + i) % rows * cols + (col + i) % cols) for i in range(win_length)])

        # Diagonal (top-right to bottom-left) win conditions with periodic boundary
        for row in range(rows):
            for col in range(cols):
                conditions.append([((row + i) % rows * cols + (col - i) % cols) for i in range(win_length)])

        return conditions

    def make_move(self, action: Action) -> bool:
        if action in self.get_valid_actions():
            self.board[action] = self.current_player
            return True
        else:
            self._invalid_move = True
            return False

    def switch_player(self) -> None:
        self.current_player = "O" if self.current_player == "X" else "X"

    def is_game_over(self) -> bool:
        self._done = self.is_won("X") or self.is_won("O") or self.is_draw() or self._invalid_move
        return self._done

    def is_won(self, player: str) -> bool:
        return any(all(self.board[pos] == player for pos in condition) for condition in self.win_conditions)

    def is_draw(self) -> bool:
        return " " not in self.board

    def get_outcome(self) -> Outcome:
        if self.is_won("X") or (self._invalid_move and self.current_player == "O"):
            return "X"
        elif self.is_won("O") or (self._invalid_move and self.current_player == "X"):
            return "O"
        elif self.is_draw():
            return "D"
        else:
            return None

    def display_board(self, board: Board, outcome: Outcome = None) -> None:
        if self.display is not None:
            self.display.update_display(board, outcome)

    def get_terminal_rewards(self, outcome: Outcome) -> Tuple[Reward, Reward]:
        if outcome == "D":
            return self.rewards["D"], self.rewards["D"]
        elif outcome == self._agent1.player:
            return self.rewards["W"], self.rewards["L"]
        elif outcome == self._agent2.player:
            return self.rewards["L"], self.rewards["W"]
        return 0.0, 0.0

    def get_valid_actions(self) -> Actions:
        return [i for i, cell in enumerate(self.board) if cell == " "]

    @staticmethod
    def get_valid_actions_from_board(board: Board) -> Actions:
        return [i for i, cell in enumerate(board) if cell == " "]

    def get_board(self) -> Board:
        return self.board
