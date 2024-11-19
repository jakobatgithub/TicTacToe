import time

from Display import TicTacToeDisplay

class TicTacToe:
    def __init__(self, agent1, agent2, display=None, waiting_time=1.0, rows=3, cols=3, win_length=3):
        self.display = None
        self._waiting_time = waiting_time
        assert cols == rows
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


    def start_game(self):
        if isinstance(self.display, TicTacToeDisplay):
            # For GUI: Schedule game logic
            self.display.after(0, self.play)
            self.display.mainloop()
        else:
            # For Console: Run game logic directly
            self.play()

    @staticmethod
    def initialize_board(rows=3, cols=3):
        return [' ' for _ in range(rows * cols)]
    
    def _make_move(self, action):
        if action in self.get_valid_actions():
            self._board[action] = self._current_player
            return True
        else:
            self._invalid_move = True
            return False
        
    def _switch_player(self):
        self._current_player = 'O' if self._current_player == 'X' else 'X'

    def _generate_win_conditions(self):
        rows = self._rows
        cols = self._cols
        win_length = self._win_length
        self._win_conditions = []

        for row in range(cols):
            for start_col in range(rows - win_length + 1):
                self._win_conditions.append(
                    [start_col + row * rows + offset for offset in range(win_length)]
                )

        for col in range(rows):
            for start_row in range(cols - win_length + 1):
                self._win_conditions.append(
                    [col + (start_row + offset) * rows for offset in range(win_length)]
                )

        for row in range(cols - win_length + 1):
            for col in range(rows - win_length + 1):
                self._win_conditions.append(
                    [(col + offset) + (row + offset) * rows for offset in range(win_length)]
                )

        for row in range(cols - win_length + 1):
            for col in range(win_length - 1, rows):
                self._win_conditions.append(
                    [(col - offset) + (row + offset) * rows for offset in range(win_length)]
                )

    def _is_won(self, player):
        for condition in self._win_conditions:
            if all(self._board[pos] == player for pos in condition):
                return True
        return False

    def _is_draw(self):
        return ' ' not in self._board

    def _is_game_over(self):
        self._done = self._is_won('X') or self._is_won('O') or self._is_draw() or self._invalid_move
        return self._done
    
    def _get_outcome(self):
        if self._is_won('X') or (self._invalid_move and self._current_player == 'O'):
            return 'X'
        elif self._is_won('O') or (self._invalid_move and self._current_player == 'X'):
            return 'O'
        elif self._is_draw():
            return 'D'
        
        return None

    def _initialize(self):
        self._done = False
        self._invalid_move = False
        self._board = self.initialize_board(self._rows, self._cols)
        self._current_player = 'X'
        self._history = []
        assert self._agent1.player != self._agent2.player

    @staticmethod
    def get_valid_actions_from_board(board):
        return [i for i, cell in enumerate(board) if cell == ' ']
    
    def get_valid_actions(self):
        return self.get_valid_actions_from_board(self._board)

    def get_board(self):
        return self._board
    
    def get_history(self):
        return self._history

    def get_current_player(self):
        return self._current_player
    
    def get_done(self):
        return self._done
    
    def display_board(self, board, outcome=None):
        if self.display is not None:
            self.display.update_display(board, outcome)
        
    def _get_step_rewards(self):
        return 0.0, 0.0
        
    def _get_terminal_rewards(self, outcome):
        if outcome == 'D':
            return 0.0, 0.0
        elif outcome == self._agent1.player:
            return 1.0, -1.0
        elif outcome == self._agent2.player:
            return -1.0, 1.0
        
        return None, None

    def play(self):
        self._initialize()
        step_reward1, step_reward2 = 0.0, 0.0
        terminal_reward1, terminal_reward2 = 0.0, 0.0

        while not self._is_game_over():
            self.display_board(self._board)
            
            if self._current_player == self._agent1.player:
                state_transition1 = (self._board[:], step_reward1, False) # board, reward, done
                action = self._agent1.get_action(state_transition1, self)
                step_reward1 = 0.0
            else:
                state_transition2 = (self._board[:], step_reward2, False) # board, reward, done
                action = self._agent2.get_action(state_transition2, self)
                step_reward2 = 0.0
            
            self._history.append((self._board[:], action))
            if self._make_move(action):
                step_reward1, step_reward2 = self._get_step_rewards()
                self._switch_player()

        outcome = self._get_outcome()
        terminal_reward1, terminal_reward2 = self._get_terminal_rewards(outcome)
        state_transition1 = (None, terminal_reward1, True) # board, reward, done
        state_transition2 = (None, terminal_reward2, True) # board, reward, done
        self._agent1.get_action(state_transition1, self)
        self._agent2.get_action(state_transition2, self)

        self.display_board(self._board, outcome)
        
        return outcome