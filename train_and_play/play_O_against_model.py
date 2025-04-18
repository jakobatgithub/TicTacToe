"""
This script allows a human player to play Tic-Tac-Toe as 'O' against a Deep Q-Learning agent playing as 'X'.

Modules:
- TicTacToe.TicTacToe: Defines the game logic.
- TicTacToe.DeepQAgent: Defines the DeepQPlayingAgent class.
- TicTacToe.Agent: Defines the HumanAgent class.
- TicTacToe.Display: Handles the game display.

Usage:
Run the script to start the game. The human player will make moves as 'O', and the agent will respond as 'X'.
"""

from pathlib import Path

from TicTacToe.TicTacToe import TicTacToe
from TicTacToe.DeepQAgent import DeepQPlayingAgent
from TicTacToe.Agent import MouseAgent
from TicTacToe.Display import ScreenDisplay

# Load the model
script_dir = Path(__file__).resolve().parent
relative_folder = (script_dir / '../models/all_models').resolve()
model_path = f"{relative_folder}/q_network_3x3x3_O.pth"

params = {
    "player": "X", # Player symbol for the agent
    "rows": 3,  # Board size (rows x rows)
    "win_length": 3,  # Number of in-a-row needed to win
    "rewards": {
        "W": 1.0,  # Reward for a win
        "L": -1.0,  # Reward for a loss
        "D": 0.5,  # Reward for a draw
    },
}

# Set up the game
agent1 = MouseAgent(player="O")
agent2 = DeepQPlayingAgent(q_network=model_path, params=params)
display = ScreenDisplay(rows=params["rows"], cols=params["rows"], waiting_time=0.5)

game = TicTacToe(agent1, agent2, display=display, params=params)

# Play the game
game.play()