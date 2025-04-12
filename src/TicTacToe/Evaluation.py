from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

import wandb
from TicTacToe.Agent import RandomAgent
from TicTacToe.DeepQAgent import DeepQLearningAgent, DeepQPlayingAgent
from TicTacToe.game_types import Player
from TicTacToe.QAgent import QLearningAgent, QPlayingAgent
from TicTacToe.SymmetricMatrix import FullySymmetricMatrix
from TicTacToe.TicTacToe import TicTacToe


def average_array(array: list[float] | list[int], chunk_size: Optional[int] = None) -> list[float]:
    means: list[float | int] = []
    if chunk_size is None:
        chunk_size = max((len(array) // 100, 1))

    for i in range(0, len(array), chunk_size):
        sublist = array[i : i + chunk_size]
        means.append(sum(sublist) / len(sublist))

    return means


def plot_graphs(loss: list[float], action_value: list[float], rewards: list[float]) -> None:
    # Create a figure with two subplots next to each other
    _, axs = plt.subplots(1, 3, figsize=(12, 3))  # type: ignore

    # Plot the first graph: Mean Loss
    chunk_size = max((len(loss) // 100, 1))
    mean_lossX = average_array(loss)
    steps = [i * chunk_size for i in range(len(mean_lossX))]

    axs[0].plot(steps[:-2], mean_lossX[:-2], label="Mean Loss")
    axs[0].set_title("Mean loss")
    axs[0].set_xlabel("Training steps")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)

    # Plot the second graph: Mean Average Action Value
    chunk_size = max((len(action_value) // 100, 1))
    mean_action_valueX = average_array(action_value)
    steps = [i * chunk_size for i in range(len(mean_lossX))]

    axs[1].plot(steps[:-2], mean_action_valueX[:-2], label="Mean Avg Action Value", color="orange")
    axs[1].set_title("Mean action value")
    axs[1].set_xlabel("Training steps")
    axs[1].set_ylabel("Action value")
    axs[1].grid(True)

    chunk_size = max((len(rewards) // 100, 1))
    mean_rewards = average_array(rewards, chunk_size)
    steps = [i * chunk_size for i in range(len(mean_rewards))]

    axs[2].plot(steps[:-2], mean_rewards[:-2], label="Mean Rewards", color="blue")
    axs[2].set_title("Mean reward per episode")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()  # type: ignore
    plt.show()  # type: ignore


def plot_valid_actions(learning_agent: DeepQLearningAgent) -> None:
    evaluation_data: dict[str, Any] = learning_agent.evaluation_data
    valid_actions: list[int] = evaluation_data["valid_actions"]
    chunk_size = max((len(valid_actions) // 100, 1))
    mean_valid_actions = average_array(valid_actions)
    steps = [i * chunk_size for i in range(len(mean_valid_actions))]
    _, axs = plt.subplots(1, 1, figsize=(4, 3))  # type: ignore

    axs.plot(steps[:-2], mean_valid_actions[:-2], label="Valid actions")  # type: ignore
    axs.set_title("Valid actions")  # type: ignore
    axs.set_xlabel("Training steps")  # type: ignore
    axs.set_ylabel("Fraction of valid actions")  # type: ignore
    axs.grid(True)  # type: ignore

    # Adjust layout for better spacing
    plt.tight_layout()  # type: ignore
    plt.show()  # type: ignore


def plot_evaluation_data(learning_agent: DeepQLearningAgent) -> None:
    evaluation_data = learning_agent.evaluation_data
    loss = evaluation_data["loss"]
    action_value = evaluation_data["action_value"]
    rewards = evaluation_data["rewards"]
    print(f"Number of losses: {len(loss)}")
    print(f"Number of action values: {len(action_value)}")
    print(f"Number of rewards: {len(rewards)}")
    plot_graphs(loss, action_value, rewards)


def extract_values(dictionary: dict[Any, float]) -> list[float]:
    """Extract all values from a potentially nested dictionary."""
    values: list[float] = []
    for _, value in dictionary.items():
        if isinstance(value, dict):  # If the value is a dictionary, recurse
            values.extend(extract_values(value))
        else:
            values.append(value)
    return values


def evaluate_and_plot_Q(learning_agent: QLearningAgent, player: Player) -> None:
    qMatrix = learning_agent.Q.qMatrix
    qValues = extract_values(qMatrix)
    print(qValues)
    print(f"Total number of elements in Q for player {player}: {len(qValues)}")

    mean_q = np.mean(qValues)
    median_q = np.median(qValues)
    std_q = np.std(qValues)
    min_q = np.min(qValues)
    max_q = np.max(qValues)

    print(f"Q-value Statistics for player {player}:")
    print(f"Mean: {mean_q}")
    print(f"Median: {median_q}")
    print(f"Standard Deviation: {std_q}")
    print(f"Minimum: {min_q}")
    print(f"Maximum: {max_q}")

    plt.figure(figsize=(10, 6))  # type: ignore
    plt.hist(qValues, bins=20, edgecolor="black", alpha=0.7)  # type: ignore
    plt.title(f"Histogram of Q-values for player {player}")  # type: ignore
    plt.xlabel("Q-value")  # type: ignore
    plt.ylabel("Frequency")  # type: ignore
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # type: ignore
    plt.show()  # type: ignore


def QAgent_plays_against_RandomAgent(
    Q: FullySymmetricMatrix,
    player: Player,
    nr_of_episodes: int = 5000,
    rows: int = 3,
    cols: int = 3,
    win_length: int = 3,
) -> None:
    playing_agent1 = QPlayingAgent(Q, player=player, switching=False)
    opponent = "O" if player == "X" else "X"
    random_agent1 = RandomAgent(player=opponent, switching=False)
    game = TicTacToe(playing_agent1, random_agent1, display=None, rows=rows, cols=cols, win_length=win_length)
    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

    print("Outcomes during playing:")
    print(
        f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}"
    )


def QAgent_plays_against_QAgent(
    Q1: FullySymmetricMatrix,
    player1: Player,
    Q2: FullySymmetricMatrix,
    player2: Player | None = None,
    nr_of_episodes: int = 5000,
    rows: int = 3,
    cols: int = 3,
    win_length: int = 3,
) -> None:
    playing_agent1 = QPlayingAgent(Q1, player=player1, switching=False)
    if not player2:
        player2 = "O" if player1 == "X" else "X"

    playing_agent2 = QPlayingAgent(Q2, player=player2, switching=False)
    game = TicTacToe(playing_agent1, playing_agent2, display=None, rows=rows, cols=cols, win_length=win_length)
    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

    print("Outcomes during playing:")
    print(
        f"X wins: {outcomes['X']/nr_of_episodes}, O wins: {outcomes['O']/nr_of_episodes}, draws: {outcomes['D']/nr_of_episodes}"
    )


def evaluate_performance(
    learning_agent1: DeepQLearningAgent,
    learning_agent2: DeepQLearningAgent,
    nr_of_episodes: int = 1000,
    rows: int = 3,
    win_length: int = 3,
    wandb_logging: bool = True,
    device: str = "cpu",
) -> None:
    q_network1 = learning_agent1.q_network
    playing_agent1 = DeepQPlayingAgent(q_network1, player="X", switching=False, device=device)
    random_agent2 = RandomAgent(player="O", switching=False)

    game = TicTacToe(playing_agent1, random_agent2, display=None, rows=rows, cols=rows, win_length=win_length)
    nr_of_episodes = nr_of_episodes
    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

    mode = "X_against_random:"
    if wandb_logging:
        wandb.log(
            {
                f"{mode} X wins": outcomes["X"] / nr_of_episodes,
                f"{mode} O wins": outcomes["O"] / nr_of_episodes,
                f"{mode} draws": outcomes["D"] / nr_of_episodes,
            }
        )

    q_network2 = learning_agent2.q_network
    playing_agent2 = DeepQPlayingAgent(q_network2, player="O", switching=False, device=device)
    random_agent1 = RandomAgent(player="X", switching=False)

    game = TicTacToe(random_agent1, playing_agent2, display=None, rows=rows, cols=rows, win_length=win_length)
    nr_of_episodes = nr_of_episodes
    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

    mode = "O_against_random:"
    if wandb_logging:
        wandb.log(
            {
                f"{mode} X wins": outcomes["X"] / nr_of_episodes,
                f"{mode} O wins": outcomes["O"] / nr_of_episodes,
                f"{mode} draws": outcomes["D"] / nr_of_episodes,
            }
        )

    game = TicTacToe(playing_agent1, playing_agent2, display=None, rows=rows, cols=rows, win_length=win_length)
    nr_of_episodes = nr_of_episodes
    outcomes = {"X": 0, "O": 0, "D": 0}
    for _ in range(nr_of_episodes):
        outcome = game.play()
        if outcome is not None:
            outcomes[outcome] += 1

    mode = "X_against_O:"
    if wandb_logging:
        wandb.log(
            {
                f"{mode} X wins": outcomes["X"] / nr_of_episodes,
                f"{mode} O wins": outcomes["O"] / nr_of_episodes,
                f"{mode} draws": outcomes["D"] / nr_of_episodes,
            }
        )
