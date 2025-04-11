# Tic Tac Toe

🎮 **Teach the computer to play Tic Tac Toe using reinforcement learning.**

## 🎲 About the Game
Tic Tac Toe is a simple yet strategic game where two players aim to align three of their symbols (X or O) horizontally, vertically, or diagonally on a 3x3 grid.

### 📜 Rules of the Game

1. The game is played on a 3x3 grid.
2. Players take turns placing their symbol (X or O) in an empty cell.
3. X always starts.
4. The first player to align three of their symbols in a row (horizontally, vertically, or diagonally) wins the game.
5. If all nine cells are filled and no player has aligned three symbols, the game ends in a draw.
6. Players cannot place their symbol in a cell that is already occupied.

## ✨ Features

- **Single-Player Mode**: Play against a random player or an AI opponent.
- **Command-Line Interface**: Play directly in the terminal.
- **Input Validation**: Ensures valid and unique moves.
- **Real-Time Board Updates**: Visualizes the game state after every move.
- **Reinforcement Learning AI**: The computer improves its gameplay by learning from self-play.
- **Use [wandb.ai](https://wandb.ai/)**: Online logging of training progress of reinforcement learning.

## 🧠 Reinforcement Learning

The AI uses a reinforcement learning algorithm to optimize its strategy. By playing thousands of games against itself, it learns to make better decisions over time. Key aspects include:

- **State-Action Mapping**: Tracks game states and corresponding actions.
- **Reward System**: Encourages winning moves and penalizes losing ones.
- **Exploration vs. Exploitation**: Balances trying new moves and leveraging learned strategies.

This approach demonstrates the power of machine learning in solving simple yet challenging problems.

## 🔍 Exploring State Space Reduction

A major focus of this project is to explore methods to reduce the state space of the game. This is achieved through:

1. **SymmetricMatrix**: 
   - Located in `src/TicTacToe/SymmetricMatrix.py`.
   - Leverages board symmetries to reduce the number of stored Q-values by identifying equivalent board states.

2. **Equivariant Neural Networks**:
   - Located in `src/TicTacToe/EquivariantNN.py`.
   - Implements neural networks with tied weights and biases based on symmetry patterns, ensuring that the network respects the symmetries of the game.

These techniques significantly reduce computational complexity and memory requirements, making the reinforcement learning process more efficient.

## 📂 File Overview

Here is a list of all files in the `src` folder and their purposes:

- **`TicTacToe/Agent.py`**: Defines the base agent class for the game.
- **`TicTacToe/DeepQAgent.py`**: Implements a deep Q-learning agent.
- **`TicTacToe/Display.py`**: Handles the display of the game board.
- **`TicTacToe/DisplayTest.py`**: Contains tests for the display module.
- **`TicTacToe/EquivariantNN.py`**: Implements equivariant neural networks for symmetry-aware learning.
- **`TicTacToe/Evaluation.py`**: Provides evaluation metrics for agents.
- **`TicTacToe/game_types.py`**: Defines types and constants used in the game.
- **`TicTacToe/QAgent.py`**: Implements a Q-learning agent.
- **`TicTacToe/SymmetricMatrix.py`**: Implements symmetric Q-value matrices to reduce state space.
- **`TicTacToe/TicTacToe.py`**: Contains the main game logic.

## ⚙️ Generalization and Options

While the original game is designed for a 3x3 grid, this implementation allows for generalization by setting various options. Key options include:

- **Grid Size**: Adjust the size of the board (e.g., 4x4, 5x5).
- **Symmetry Handling**: Enable or disable symmetry-based state space reduction.
- **Learning Parameters**: Configure learning rates, exploration rates, and reward structures.
- **Neural Network Architecture**: Customize the architecture of the equivariant neural networks.

These options provide flexibility for experimenting with different configurations and exploring the impact of various parameters on learning performance.

## 🛠️ Installation

1. **Clone the repository and navigate to the project directory**:
   ```bash
   git clone git@github.com:jakobatgithub/TicTacToe.git
   cd TicTacToe
   ```
2. **Create a virtual environment, activate it, and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install .
   ```
3. **Tkinter must be installed**:
   ```bash
   brew install python-tk
   ```
4. **Optional: get an account at [wandb.ai](https://wandb.ai/) and log in**:
   ```bash
   wandb login
   ```
5. **Train models for players 'X' and 'O' by having two computer agents play against each other**:
   ```bash
   python train_and_play/train_model.py
   ```
6. **Play 'X' against the trained model**:
   ```bash
   python train_and_play/play_X_against_model.py
   ```
6. **Play 'O' against the trained model**:
   ```bash
   python train_and_play/play_O_against_model.py
   ```

## 📜 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it under the terms of the license.