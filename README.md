# Tic Tac Toe

A Python-based implementation of the classic Tic Tac Toe game, enhanced with reinforcement learning. The AI learns to play by competing against itself, making this project a fun way to explore game development and machine learning concepts.

## Table of Contents
- [About the Game](#about-the-game)
- [Features](#features)
- [Reinforcement Learning](#reinforcement-learning)
- [Installation](#installation)
- [How to Play](#how-to-play)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## About the Game

Tic Tac Toe is a simple yet strategic game where two players aim to align three of their symbols (X or O) horizontally, vertically, or diagonally on a 3x3 grid.

---

## Features

- **Single-Player Mode**: Play against a random player or an AI opponent.
- **Reinforcement Learning AI**: The computer improves its gameplay by learning from self-play.
- **Command-Line Interface**: Play directly in the terminal.
- **Input Validation**: Ensures valid and unique moves.
- **Real-Time Board Updates**: Visualizes the game state after every move.

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
4. **Get an account at [wandb.ai](https://wandb.ai/) and log in**:
   ```bash
   wandb login
   ```
5. **Run the Game**:
   ```bash
   python notebooks/TicTacToeDeepQLearning.py
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
