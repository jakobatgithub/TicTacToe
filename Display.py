import time
import tkinter as tk
from IPython.display import clear_output
from abc import ABC, abstractmethod

class Display(ABC):
    @abstractmethod
    def update_display(self, board, outcome=None):
        """Update the display with the given board state."""
        pass

    @abstractmethod
    def set_message(self, message):
        """Display a message."""
        pass


class TicTacToeDisplay(tk.Tk, Display):
    def __init__(self, rows=3, cols=3, waiting_time=0.25):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.waiting_time = waiting_time

        self.title(f"Tic-Tac-Toe {self.rows}x{self.cols}")
        self.labels = []
        self.message_label = tk.Label(self, text="", font=("Arial", 16))
        self.message_label.grid(row=0, column=0, columnspan=self.cols)  # Span the entire top row
        self._init_display()

    def _init_display(self):
        """Initialize the board display as a grid of labels."""
        for i in range(self.rows * self.cols):  # 9 fields for a 3x3 board
            label = tk.Label(self, text=' ', font=('Arial', 24), width=5, height=2, borderwidth=1, relief="solid")
            label.grid(row=(i // self.rows) + 1, column=i % self.cols)  # Offset by +1 to make room for the message label
            self.labels.append(label)

    def update_display(self, board, outcome=None):
        """Update the display with the given board state."""
        for i, value in enumerate(board):
            self.labels[i].config(text=value if value in ['X', 'O'] else ' ')

        if outcome is not None:
            if outcome in ['X', 'O']:
                self.set_message(f"Player {outcome} wins!")
            elif outcome == 'D':
                self.set_message("It's a draw!")
        
        self.update()
        time.sleep(self.waiting_time)
        if outcome is not None:
            self.quit()

    def set_message(self, message):
        """Update the message displayed at the top of the window."""
        self.message_label.config(text=message)


class ConsoleDisplay(Display):
    def __init__(self, rows=3, cols=3, waiting_time=0.25):
        self.rows = rows
        self.cols = cols
        self.waiting_time = waiting_time

    def update_display(self, board, outcome=None):
        """Display the board dynamically in the console."""
        clear_output(wait=True)
        row_divider = "-" * (6 * self.rows - 1)
        
        for row in range(self.cols):
            row_content = " | ".join(
                f" {board[col + row * self.rows]} " for col in range(self.rows)
            )
            print(row_content)
            if row < self.cols - 1:
                print(row_divider)
        
        print("\n")
        if outcome is not None:
            if outcome in ['X', 'O']:
                self.set_message(f"Player {outcome} wins!")
            elif outcome == 'D':
                self.set_message("It's a draw!")
        
        time.sleep(self.waiting_time)

    def set_message(self, message):
        """Set a message for the console display."""
        print(message)