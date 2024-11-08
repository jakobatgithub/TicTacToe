import time
from IPython.display import clear_output

class TicTacToe:
    def __init__(self, agent_x, agent_o, display=None):
        self.display = display
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.board = self.initialize_board()
        self.current_player = 'X'
        self.history = []  # To store state-action pairs

    # Initialize the Tic-Tac-Toe board
    def initialize_board(self):
        return [' ' for _ in range(9)]
    
    # Generate all empty positions on the board
    def get_valid_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def get_board(self):
        return self.board
    
    def get_history(self):
        return self.history

    def make_move(self, action):
        if action in self.get_valid_actions():
            self.board[action] = self.current_player
            return True
        else:
            return False
        
    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'        

    # Display the board in a 3x3 format
    def display_board(self):
        clear_output(wait=True)
        board = self.board
        # print("\n")
        print(f" {board[0]}  |  {board[1]}  |  {board[2]} ")
        print("----+-----+----")
        print(f" {board[3]}  |  {board[4]}  |  {board[5]} ")
        print("----+-----+----")
        print(f" {board[6]}  |  {board[7]}  |  {board[8]} ")
        print("\n")

    # Check for a winning condition
    def is_won(self, player):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for condition in win_conditions:
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

    # Main game loop
    def play(self):
        self.board = self.initialize_board()
        self.current_player = 'X'
        self.history = []  # To store state-action pairs        

        while not self.is_game_over():
            if self.display:
                self.display_board()  # Optional: Display the board after each move
                time.sleep(0.25)  # Wait a bit before the next move for readability
            if self.current_player == 'X':
                action = self.agent_x.get_action(self)
            else:
                action = self.agent_o.get_action(self)
            
            self.history.append((self.board[:], action))
            if self.make_move(action):
                self.switch_player()
            else:
                print("Invalid move. Try again.")
                continue

        outcome = self.get_outcome()
        self.agent_x.notify_result(self, outcome)
        self.agent_o.notify_result(self, outcome)
        if self.display:
            self.display_board()  # Optional: Display the board after each move
            time.sleep(0.25)  # Wait a bit before the next move for readability
            if outcome == 'X' or outcome == 'O':
                print(f"Player {outcome} wins!")
            else:
                print("It's a draw!")
        
        return outcome