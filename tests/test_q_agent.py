# type: ignore

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from TicTacToe.QAgent import QLearningAgent, QPlayingAgent


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        # Mock parameters for QLearningAgent
        self.params: dict[str, Any] = {
            "player": "X",
            "switching": False,
            "debug": False,
            "gamma": 0.9,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "alpha_start": 0.1,
            "alpha_min": 0.01,
            "nr_of_episodes": 100,
            "terminal_q_updates": True,
            "Q_initial_value": 0.0,
        }
        self.agent = QLearningAgent(self.params)

    def test_initialization(self):
        self.assertEqual(self.agent.params["player"], "X")
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertIsNotNone(self.agent.Q)

    def test_choose_action_random(self):
        board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        with patch("random.uniform", return_value=0.5):
            with patch("random.choice", return_value=2):
                action = self.agent.choose_action(board, epsilon=1.0)
                self.assertEqual(action, 2)

    def test_choose_action_exploit(self):
        board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        self.agent.Q.set(tuple(board), 2, 1.0)
        self.agent.Q.set(tuple(board), 3, 0.5)
        action = self.agent.choose_action(board, epsilon=0.0)
        self.assertEqual(action, 2)

    def test_q_update(self):
        board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        next_board = ["X", "O", "X", " ", " ", " ", " ", " ", " "]
        action = 2
        reward = 1.0

        self.agent.Q.set(tuple(board), action, 0.0)
        self.agent.Q.set(tuple(next_board), 3, 0.5)

        loss, _ = self.agent.q_update(board, action, next_board, reward)
        self.assertGreater(loss, 0)
        self.assertEqual(self.agent.Q.get(tuple(board), action), 0.1 * (reward + 0.9 * 0.5))

    def test_update_rates(self):
        self.agent.update_rates(50)
        self.assertLess(self.agent.epsilon, 1.0)
        self.assertLess(self.agent.alpha, 0.1)

    def test_get_action_game_in_progress(self):
        state_transition = (["X", "O", " ", " ", " ", " ", " ", " ", " "], 0, False)
        game_mock = MagicMock()
        game_mock.get_board.return_value = ["X", "O", " ", " ", " ", " ", " ", " ", " "]

        with patch.object(self.agent, "choose_action", return_value=2):
            action = self.agent.get_action(state_transition, game_mock)

        self.assertEqual(action, 2)
        self.assertEqual(len(self.agent.episode_history), 1)

    def test_get_action_game_ends(self):
        state_transition = (["X", "O", " ", " ", " ", " ", " ", " ", " "], 1, True)
        game_mock = MagicMock()
        game_mock.get_board.return_value = ["X", "O", " ", " ", " ", " ", " ", " ", " "]

        with patch.object(self.agent, "q_update_backward", return_value=(0, 0)) as q_update_mock:
            action = self.agent.get_action(state_transition, game_mock)

        self.assertEqual(action, -1)
        q_update_mock.assert_called_once()
        self.assertEqual(self.agent.episode_count, 1)

    def test_q_update_backward(self):
        # Mock the history of state-action pairs for the episode
        history = [
            (["X", "O", " ", " ", " ", " ", " ", " ", " "], 2),
            (["X", "O", "X", " ", " ", " ", " ", " ", " "], 3),
            (["X", "O", "X", "O", " ", " ", " ", " ", " "], 4),
        ]
        terminal_reward = 1.0

        # Mock Q.get and Q.set to simulate the Q-value updates
        self.agent.Q = MagicMock()
        self.agent.Q.get.return_value = 0.0
        self.agent.Q.get.side_effect = lambda b, a: 0.0  # All initial Q-values are 0
        self.agent.Q.set = MagicMock()

        avg_loss, action_value = self.agent.q_update_backward(history, terminal_reward)

        # Ensure Q.set was called for each state-action pair
        self.assertEqual(self.agent.Q.set.call_count, len(history))

        # Verify the final update uses the terminal reward
        args = self.agent.Q.set.call_args_list[0][0]
        state, action, new_value = args
        self.assertEqual(state, tuple(history[2][0]))
        self.assertEqual(action, history[2][1])
        self.assertGreaterEqual(new_value, 0)  # New value should not be negative

        args = self.agent.Q.set.call_args_list[1][0]
        state, action, new_value = args
        self.assertEqual(state, tuple(history[1][0]))
        self.assertEqual(action, history[1][1])
        self.assertGreaterEqual(new_value, 0)  # New value should not be negative

        args = self.agent.Q.set.call_args_list[2][0]
        state, action, new_value = args
        self.assertEqual(state, tuple(history[0][0]))
        self.assertEqual(action, history[0][1])
        self.assertGreaterEqual(new_value, 0)  # New value should not be negative

        # Check returned average loss and action value
        self.assertGreaterEqual(avg_loss, 0)
        self.assertGreaterEqual(action_value, 0)


class TestQPlayingAgent(unittest.TestCase):
    def setUp(self):
        # Mock Q matrix for QPlayingAgent
        self.Q = MagicMock()
        self.agent = QPlayingAgent(self.Q, player="O", switching=True)

    def test_initialization(self):
        self.assertEqual(self.agent.player, "O")
        self.assertEqual(self.agent.switching, True)
        self.assertIs(self.agent.Q, self.Q)

    def test_choose_action(self):
        board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        self.Q.get.return_value = 1.0
        self.Q.get.side_effect = lambda b, a: 1.0 if a == 2 else 0.5

        action = self.agent.choose_action(board)
        self.assertEqual(action, 2)

    def test_get_valid_actions(self):
        board = ["X", "O", " ", " ", "X", "O", "X", " ", "O"]
        valid_actions = self.agent.get_valid_actions(board)
        self.assertListEqual(valid_actions, [2, 3, 7])

    def test_get_best_action(self):
        board = ["X", "O", " ", " ", " ", " ", " ", " ", " "]
        self.Q.get.side_effect = lambda b, a: 1.0 if a == 2 else 0.5
        action = self.agent.get_best_action(board, self.Q)
        self.assertEqual(action, 2)

    def test_on_game_end(self):
        self.agent.on_game_end()
        self.assertEqual(self.agent.player, "X")
        self.assertEqual(self.agent.opponent, "O")

    def test_get_action_game_in_progress(self):
        state_transition = (["X", "O", " ", " ", " ", " ", " ", " ", " "], 0, False)
        game_mock = MagicMock()
        game_mock.get_board.return_value = ["X", "O", " ", " ", " ", " ", " ", " ", " "]

        with patch.object(self.agent, "choose_action", return_value=2):
            action = self.agent.get_action(state_transition, game_mock)

        self.assertEqual(action, 2)

    def test_get_action_game_ends(self):
        state_transition = (["X", "O", " ", " ", " ", " ", " ", " ", " "], 1, True)
        game_mock = MagicMock()

        with patch.object(self.agent, "on_game_end") as on_game_end_mock:
            action = self.agent.get_action(state_transition, game_mock)

        self.assertEqual(action, -1)
        on_game_end_mock.assert_called_once()
