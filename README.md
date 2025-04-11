# Tic Tac Toe

Teach the computer to play Tic Tac Toe using reinforcement learning.

## About the Game
Tic Tac Toe is a simple yet strategic game where two players aim to align three of their symbols (X or O) horizontally, vertically, or diagonally on a 3x3 grid.

### Rules of the Game

1. The game is played on a 3x3 grid.
2. Players take turns placing their symbol (X or O) in an empty cell.
3. X always starts.
4. The first player to align three of their symbols in a row (horizontally, vertically, or diagonally) wins the game.
5. If all nine cells are filled and no player has aligned three symbols, the game ends in a draw.
6. Players cannot place their symbol in a cell that is already occupied.

---

## Features

- **Single-Player Mode**: Play against a random player or an AI opponent.
- **Command-Line Interface**: Play directly in the terminal.
- **Input Validation**: Ensures valid and unique moves.
- **Real-Time Board Updates**: Visualizes the game state after every move.
- **Reinforcement Learning AI**: The computer improves its gameplay by learning from self-play.
- **Use [wandb.ai](https://wandb.ai/)**: Online logging of training progress of reinforcement learning.
---

## Reinforcement Learning

The AI uses a reinforcement learning algorithm to optimize its strategy. By playing thousands of games against itself, it learns to make better decisions over time. Key aspects include:

- **State-Action Mapping**: Tracks game states and corresponding actions.
- **Reward System**: Encourages winning moves and penalizes losing ones.
- **Exploration vs. Exploitation**: Balances trying new moves and leveraging learned strategies.

This approach demonstrates the power of machine learning in solving simple yet challenging problems.

---

## Installation

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

---

## How to Play

1. Start the game using the [Installation](#installation) steps.
2. Select a mode: Single-player, Two-player, or AI vs. AI.
3. Follow the prompts to make your moves.
4. The game ends when a player wins or the board is full (draw).

---

## Technologies Used

- **Language**: Python 3.x
- **Libraries**: Standard Python libraries (no external dependencies required).

---

## Contributing

Contributions are welcome! Open an issue or submit a pull request to suggest improvements or report bugs.

### Steps to Contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-branch-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it under the terms of the license.

---

Enjoy exploring reinforcement learning with Tic Tac Toe! 🕹️
