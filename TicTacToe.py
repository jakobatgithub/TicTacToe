import time

class TicTacToe:
    def __init__(self, agent1, agent2, display=None, waiting_time=0.25, width=3, height=3, win_length=3):
        self.display = display
        self.waiting_time = waiting_time
        assert height == width
        self.agent1 = agent1
        self.agent2 = agent2
        self.width = width
        self.height = height
        self.win_length = win_length
        self.generate_win_conditions()
        self.initialize()

    # Initialize the Tic-Tac-Toe board
    def initialize_board(self):
        return [' ' for _ in range(self.width * self.height)]
    
    # Generate all empty positions on the board
    def get_valid_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def get_board(self):
        return self.board
    
    def get_history(self):
        return self.history

    def get_current_player(self):
        return self.current_player

    def make_move(self, action):
        if action in self.get_valid_actions():
            self.board[action] = self.current_player
            return True
        else:
            return False
        
    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'        

    
    # Display the board in a width x height format
    def display_board(self):
        from IPython.display import clear_output  # Importing here for standalone function clarity
        clear_output(wait=True)
        board = self.board
        width = self.width
        height = self.height
        row_divider = "-" * (6 * width - 1)  # Dynamic row divider based on board width
        
        for row in range(height):
            row_content = " | ".join(
                f" {board[col + row * width]} " for col in range(width)
            )
            print(row_content)
            if row < height - 1:  # Print dividers only between rows
                print(row_divider)
        print("\n")

    def generate_win_conditions(self):
        width = self.width
        height = self.height
        win_length = self.win_length
        self.win_conditions = []

        # Rows
        for row in range(height):
            for start_col in range(width - win_length + 1):  # Ensure win_length-length sequence fits
                self.win_conditions.append(
                    [start_col + row * width + offset for offset in range(win_length)]
                )

        # Columns
        for col in range(width):
            for start_row in range(height - win_length + 1):  # Ensure win_length-length sequence fits
                self.win_conditions.append(
                    [col + (start_row + offset) * width for offset in range(win_length)]
                )

        # Diagonals (top-left to bottom-right)
        for row in range(height - win_length + 1):
            for col in range(width - win_length + 1):  # Ensure win_length-length sequence fits
                self.win_conditions.append(
                    [(col + offset) + (row + offset) * width for offset in range(win_length)]
                )

        # Diagonals (top-right to bottom-left)
        for row in range(height - win_length + 1):
            for col in range(win_length - 1, width):  # Ensure win_length-length sequence fits
                self.win_conditions.append(
                    [(col - offset) + (row + offset) * width for offset in range(win_length)]
                )


    def is_won(self, player):
        # Check if any win condition is satisfied
        for condition in self.win_conditions:
            if all(self.board[pos] == player for pos in condition):
                return True

        return False


    # Check for a draw (no empty spaces)
    def is_draw(self):
        return ' ' not in self.board

    def is_game_over(self):
        return self.is_won('X') or self.is_won('O') or self.is_draw()
    
    def get_outcome(self):
        if self.is_won('X'):
            return 'X'
        elif self.is_won('O'):
            return 'O'
        elif self.is_draw():
            return 'D'

    def initialize(self):
        self.board = self.initialize_board()
        self.current_player = 'X'
        self.history = []  # To store state-action pairs
        assert self.agent1.player != self.agent2.player

    # Main game loop
    def play(self):
        self.initialize()
        while not self.is_game_over():
            if self.display:
                self.display_board()  # Optional: Display the board after each move
                time.sleep(self.waiting_time)  # Wait a bit before the next move for readability
            if self.current_player == self.agent1.player:
                action = self.agent1.get_action(self)
            else:
                action = self.agent2.get_action(self)
            
            self.history.append((self.board[:], action))
            if self.make_move(action):
                self.switch_player()
            else:
                print("Invalid move. Try again.")
                continue

        outcome = self.get_outcome()
        self.agent1.notify_result(self, outcome)
        self.agent2.notify_result(self, outcome)

        if self.display:
            self.display_board()  # Optional: Display the board after each move
            time.sleep(self.waiting_time)  # Wait a bit before the next move for readability
            if outcome == 'X' or outcome == 'O':
                print(f"Player {outcome} wins!")
            else:
                print("It's a draw!")
        
        return outcome