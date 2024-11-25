import time
import tkinter as tk
from abc import ABC, abstractmethod
from typing import Any, Callable

from IPython.display import clear_output

from my_types import Action, Board, Outcome


class Display(ABC):
    @abstractmethod
    def update_display(self, board: Board, outcome: Outcome = None) -> None:
        """Update the display with the given board state."""
        pass

    @abstractmethod
    def set_message(self, message: str) -> None:
        """Display a message."""
        pass


class TicTacToeDisplay(tk.Tk, Display):
    def __init__(self, rows: int = 3, cols: int = 3, waiting_time: float = 0.25) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.waiting_time = waiting_time

        self.title(f"Tic-Tac-Toe {self.rows}x{self.cols}")
        self.labels: list[tk.Label] = []
        self.message_label = tk.Label(self, text="", font=("Arial", 16))
        self.message_label.grid(row=0, column=0, columnspan=self.cols)  # Span the entire top row
        self._init_display()
        self.click_handler: Callable[[Action], None] | None = None  # A callback for handling clicks
        self.action_complete = tk.BooleanVar(value=False)  # Persistent variable to control wait state

    def _init_display(self) -> None:
        """Initialize the board display as a grid of labels."""
        for idx in range(self.rows * self.cols):  # 9 fields for a 3x3 board
            label = tk.Label(self, text=" ", font=("Arial", 24), width=5, height=2, borderwidth=1, relief="solid")
            label.grid(
                row=(idx // self.rows) + 1, column=idx % self.cols
            )  # Offset by +1 to make room for the message label
            label.bind("<Button-1>", lambda event, action=idx: self.handle_click(event, action))
            self.labels.append(label)

    def handle_click(self, event: Any, action: Action) -> None:
        """Handle a mouse click on the board."""
        if self.click_handler:
            self.click_handler(action)
            self.action_complete.set(True)  # Signal that the action is complete

    def bind_click_handler(self, handler: Callable[[Action], None]) -> None:
        """Bind the click handler for mouse input."""
        self.click_handler = handler

    def wait_for_player_action(self) -> None:
        """Wait for the player to perform an action (no-op for GUI)."""
        self.action_complete.set(False)  # Reset the variable before waiting
        self.wait_variable(self.action_complete)  # Suspend until an action occurs

    def update_display(self, board: Board, outcome: Outcome = None) -> None:
        """Update the display with the given board state."""
        for i, value in enumerate(board):
            self.labels[i].config(text=value if value in ["X", "O"] else " ")

        if outcome is not None:
            if outcome in ["X", "O"]:
                self.set_message(f"Player {outcome} wins!")
            elif outcome == "D":
                self.set_message("It's a draw!")

        self.update()
        time.sleep(self.waiting_time)
        if outcome is not None:
            self.quit()

    def set_message(self, message: str) -> None:
        """Update the message displayed at the top of the window."""
        self.message_label.config(text=message)


class ConsoleDisplay(Display):
    def __init__(self, rows: int = 3, cols: int = 3, waiting_time: float = 0.25) -> None:
        self.rows = rows
        self.cols = cols
        self.waiting_time = waiting_time

    def update_display(self, board: Board, outcome: Outcome = None) -> None:
        """Display the board dynamically in the console."""
        clear_output(wait=True)
        row_divider = "-" * (6 * self.rows - 1)

        for row in range(self.cols):
            row_content = " | ".join(f" {board[col + row * self.rows]} " for col in range(self.rows))
            print(row_content)
            if row < self.cols - 1:
                print(row_divider)

        print("\n")
        if outcome is not None:
            if outcome in ["X", "O"]:
                self.set_message(f"Player {outcome} wins!")
            elif outcome == "D":
                self.set_message("It's a draw!")

        time.sleep(self.waiting_time)

    def set_message(self, message: str) -> None:
        """Set a message for the console display."""
        print(message)
