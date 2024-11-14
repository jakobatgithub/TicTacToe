import time

class TicTacToe:
    def __init__(self, agent1, agent2, display=None, waiting_time=0.25, width=3, height=3, win_length=3):
        self._display = display
        self._waiting_time = waiting_time
        assert height == width
        self._agent1 = agent1
        self._agent2 = agent2
        self._width = width
        self._height = height
        self._win_length = win_length
        self._generate_win_conditions()
        self._initialize()

    def _initialize_board(self):
        return [' ' for _ in range(self._width * self._height)]
    
    def _make_move(self, action):
        if action in self.get_valid_actions():
            self._board[action] = self._current_player
            return True
        else:
            return False
        
    def _switch_player(self):
        self._current_player = 'O' if self._current_player == 'X' else 'X'

    def _generate_win_conditions(self):
        width = self._width
        height = self._height
        win_length = self._win_length
        self._win_conditions = []

        for row in range(height):
            for start_col in range(width - win_length + 1):
                self._win_conditions.append(
                    [start_col + row * width + offset for offset in range(win_length)]
                )

        for col in range(width):
            for start_row in range(height - win_length + 1):
                self._win_conditions.append(
                    [col + (start_row + offset) * width for offset in range(win_length)]
                )

        for row in range(height - win_length + 1):
            for col in range(width - win_length + 1):
                self._win_conditions.append(
                    [(col + offset) + (row + offset) * width for offset in range(win_length)]
                )

        for row in range(height - win_length + 1):
            for col in range(win_length - 1, width):
                self._win_conditions.append(
                    [(col - offset) + (row + offset) * width for offset in range(win_length)]
                )

    def _is_won(self, player):
        for condition in self._win_conditions:
            if all(self._board[pos] == player for pos in condition):
                return True
        return False

    def _is_draw(self):
        return ' ' not in self._board

    def _is_game_over(self):
        self._done = self._is_won('X') or self._is_won('O') or self._is_draw()
        return self._done
    
    def _get_outcome(self):
        if self._is_won('X'):
            return 'X'
        elif self._is_won('O'):
            return 'O'
        elif self._is_draw():
            return 'D'
        
        return None

    def _initialize(self):
        self._done = False
        self._board = self._initialize_board()
        self._current_player = 'X'
        self._history = []
        assert self._agent1.player != self._agent2.player

    def get_valid_actions(self):
        return [i for i, cell in enumerate(self._board) if cell == ' ']

    def get_board(self):
        return self._board
    
    def get_history(self):
        return self._history

    def get_current_player(self):
        return self._current_player
    
    def get_done(self):
        return self._done
    
    def display_board(self):
        from IPython.display import clear_output
        clear_output(wait=True)
        board = self._board
        width = self._width
        height = self._height
        row_divider = "-" * (6 * width - 1)
        
        for row in range(height):
            row_content = " | ".join(
                f" {board[col + row * width]} " for col in range(width)
            )
            print(row_content)
            if row < height - 1:
                print(row_divider)
        print("\n")

    def _get_step_rewards_for_valid_move(self):
        return 0.0, 0.0

    def _get_step_rewards_for_invalid_move(self):
        if self._current_player == self._agent1.player:
            return 0.0, 0.0
        else:
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
            if self._display:
                self.display_board()
                time.sleep(self._waiting_time)
            
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
                step_reward1, step_reward2 = self._get_step_rewards_for_valid_move()
                self._switch_player()
            else:
                print("Invalid move. Try again.")
                step_reward1, step_reward2 = self._get_step_rewards_for_invalid_move()
                continue

        outcome = self._get_outcome()
        terminal_reward1, terminal_reward2 = self._get_terminal_rewards(outcome)
        state_transition1 = (None, terminal_reward1, True) # board, reward, done
        state_transition2 = (None, terminal_reward2, True) # board, reward, done
        self._agent1.get_action(state_transition1, self)
        self._agent2.get_action(state_transition2, self)

        if self._display:
            self.display_board()
            time.sleep(self._waiting_time)
            if outcome == 'X' or outcome == 'O':
                print(f"Player {outcome} wins!")
            else:
                print("It's a draw!")
        
        return outcome


# def play(self):
    #     self._initialize()
    #     while not self._is_game_over():
    #         if self._display:
    #             self.display_board()
    #             time.sleep(self._waiting_time)
            
    #         if self._current_player == self._agent1.player:
    #             action = self._agent1.get_action(self)
    #         else:
    #             action = self._agent2.get_action(self)
            
    #         self._history.append((self._board[:], action))
    #         if self._make_move(action):
    #             self._switch_player()
    #         else:
    #             print("Invalid move. Try again.")
    #             continue

    #     outcome = self._get_outcome()
    #     self._agent1.on_game_end(self, outcome)
    #     self._agent2.on_game_end(self, outcome)

    #     if self._display:
    #         self.display_board()
    #         time.sleep(self._waiting_time)
    #         if outcome == 'X' or outcome == 'O':
    #             print(f"Player {outcome} wins!")
    #         else:
    #             print("It's a draw!")
        
    #     return outcome