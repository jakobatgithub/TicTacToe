from TicTacToe.Agent import Agent, MouseAgent
from TicTacToe.Display import Display, TicTacToeDisplay
from TicTacToe.game_types import Action, Actions, Board, History, Outcome, Player, Reward, StateTransition


class TicTacToe:
    def __init__(
        self,
        agent1: Agent,
        agent2: Agent,
        display: Display | None = None,
        waiting_time: float = 1.0,
        rows: int = 3,
        cols: int = 3,
        win_length: int = 3,
    ) -> None:
        self.display = None
        self._waiting_time = waiting_time
        if cols != rows:
            raise ValueError("Board must be square")

        self._agent1 = agent1
        self._agent2 = agent2
        self._rows = rows
        self._cols = cols
        self._win_length = win_length
        self._generate_win_conditions()
        self._initialize()
        if display is not None:
            self.display = display
            self.start_game()

    def start_game(self) -> None:
        if isinstance(self._agent1, MouseAgent) or isinstance(self._agent2, MouseAgent):
            if not isinstance(self.display, TicTacToeDisplay):
                raise ValueError("Mouse agent can only be used with TicTacToeDisplay")

        if isinstance(self.display, TicTacToeDisplay):
            # For GUI: Schedule game logic
            self.display.after(0, self.play)
            self.display.mainloop()
        else:
            # For Console: Run game logic directly
            self.play()

    @staticmethod
    def initialize_board(rows: int = 3, cols: int = 3) -> Board:
        return [" " for _ in range(rows * cols)]

    def make_move(self, action: Action) -> bool:
        if action in self.get_valid_actions():
            self.board[action] = self.current_player
            return True
        else:
            self._invalid_move = True
            return False

    def switch_player(self) -> None:
        self.current_player = "O" if self.current_player == "X" else "X"

    def _generate_win_conditions(self) -> None:
        rows = self._rows
        cols = self._cols
        win_length = self._win_length
        self.win_conditions: list[list[int]] = []

        for row in range(cols):
            for start_col in range(rows - win_length + 1):
                self.win_conditions.append([start_col + row * rows + offset for offset in range(win_length)])

        for col in range(rows):
            for start_row in range(cols - win_length + 1):
                self.win_conditions.append([col + (start_row + offset) * rows for offset in range(win_length)])

        for row in range(cols - win_length + 1):
            for col in range(rows - win_length + 1):
                self.win_conditions.append([(col + offset) + (row + offset) * rows for offset in range(win_length)])

        for row in range(cols - win_length + 1):
            for col in range(win_length - 1, rows):
                self.win_conditions.append([(col - offset) + (row + offset) * rows for offset in range(win_length)])

    def is_won(self, player: str) -> bool:
        for condition in self.win_conditions:
            if all(self.board[pos] == player for pos in condition):
                return True
        return False

    def is_draw(self) -> bool:
        return " " not in self.board

    def is_game_over(self) -> bool:
        self._done = self.is_won("X") or self.is_won("O") or self.is_draw() or self._invalid_move
        return self._done

    def get_outcome(self) -> Outcome:
        if self.is_won("X") or (self._invalid_move and self.current_player == "O"):
            return "X"
        elif self.is_won("O") or (self._invalid_move and self.current_player == "X"):
            return "O"
        elif self.is_draw():
            return "D"
        else:
            return None

    def _initialize(self) -> None:
        self._done: bool = False
        self._invalid_move: bool = False
        self.board: Board = self.initialize_board(self._rows, self._cols)
        self.current_player: Player = "X"
        self._history: History = []
        if self._agent1.player == self._agent2.player:
            raise ValueError("Players must be different")

    @staticmethod
    def get_valid_actions_from_board(board: Board) -> Actions:
        return [i for i, cell in enumerate(board) if cell == " "]

    def get_valid_actions(self) -> Actions:
        return self.get_valid_actions_from_board(self.board)

    def get_board(self) -> Board:
        return self.board

    def get_history(self) -> History:
        return self._history

    def get_current_player(self) -> Player:
        return self.current_player

    def get_done(self) -> bool:
        return self._done

    def display_board(self, board: Board, outcome: Outcome = None) -> None:
        if self.display is not None:
            self.display.update_display(board, outcome)

    def _get_step_rewards(self):
        return 0.0, 0.0

    def get_terminal_rewards(self, outcome: Outcome) -> tuple[Reward, Reward]:
        if outcome is not None:
            if outcome == "D":
                return 0.0, 0.0
            elif outcome == self._agent1.player:
                return 1.0, -1.0
            elif outcome == self._agent2.player:
                return -1.0, 1.0
        return 0.0, 0.0

    def play(self) -> Outcome:
        self._initialize()
        step_reward1, step_reward2 = 0.0, 0.0
        terminal_reward1, terminal_reward2 = 0.0, 0.0

        while not self.is_game_over():
            self.display_board(self.board)

            if self.current_player == self._agent1.player:
                state_transition1: StateTransition = (self.board[:], step_reward1, False)  # board, reward, done
                action = self._agent1.get_action(state_transition1, self)
                step_reward1 = 0.0
            else:
                state_transition2: StateTransition = (self.board[:], step_reward2, False)  # board, reward, done
                action = self._agent2.get_action(state_transition2, self)
                step_reward2 = 0.0

            self._history.append((self.board[:], action))
            if self.make_move(action):
                step_reward1, step_reward2 = self._get_step_rewards()
                self.switch_player()

        outcome = self.get_outcome()
        self.display_board(self.board, outcome)
        terminal_reward1, terminal_reward2 = self.get_terminal_rewards(outcome)
        state_transition1 = (None, terminal_reward1, True)  # board, reward, done
        state_transition2 = (None, terminal_reward2, True)  # board, reward, done
        self._agent1.get_action(state_transition1, self)
        self._agent2.get_action(state_transition2, self)

        return outcome
