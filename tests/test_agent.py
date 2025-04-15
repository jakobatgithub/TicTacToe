# type: ignore

import unittest
from unittest.mock import patch

from TicTacToe.Agent import Agent, HumanAgent, MouseAgent, RandomAgent
from TicTacToe.Display import ScreenDisplay


class TestAgentBase(unittest.TestCase):
    """Tests for the Agent base class and its derived classes."""

    def test_agent_initialization(self) -> None:
        """Test initialization of player and opponent in Agent."""

        class ConcreteAgent(Agent):
            def get_action(self, state_transition, game):
                return None

        agent = ConcreteAgent(player="X", switching=True)
        self.assertEqual(agent.player, "X", "Player should be initialized correctly.")
        self.assertEqual(agent.opponent, "O", "Opponent should be initialized correctly.")
        self.assertTrue(agent.switching, "Switching flag should be set correctly.")

    def test_random_agent_action(self) -> None:
        """Ensure RandomAgent selects valid actions."""

        class MockGame:
            def get_valid_actions(self):
                return [0, 1, 2]

        agent = RandomAgent(player="X")
        game = MockGame()
        action = agent.get_action((None, 0, False), game)
        self.assertIn(action, game.get_valid_actions(), "RandomAgent should select a valid action.")

    @patch("builtins.input", side_effect=["0", "2"])
    def test_human_agent_valid_input(self, mock_input) -> None:
        """Mock user input and verify HumanAgent selects valid actions."""

        class MockGame:
            def get_valid_actions(self):
                return [0, 2]

        agent = HumanAgent(player="X")
        game = MockGame()
        action = agent.get_action((None, 0, False), game)
        self.assertIn(action, game.get_valid_actions(), "HumanAgent should select a valid action.")

    @patch("builtins.input", side_effect=["invalid", "5", "1"])
    def test_human_agent_invalid_input(self, mock_input) -> None:
        """Simulate invalid user inputs and verify HumanAgent handles them."""

        class MockGame:
            def get_valid_actions(self):
                return [1, 2]

        agent = HumanAgent(player="X")
        game = MockGame()
        action = agent.get_action((None, 0, False), game)
        self.assertIn(action, game.get_valid_actions(), "HumanAgent should eventually select a valid action.")

    def test_mouse_agent_action(self) -> None:
        """Simulate GUI clicks to test MouseAgent."""

        class MockGame:
            class MockDisplay(ScreenDisplay):
                def bind_click_handler(self, handler) -> None:
                    self.handler = handler

                def wait_for_player_action(self) -> None:
                    # Simulate a click event
                    self.handler(3)

            display = MockDisplay()

        agent = MouseAgent(player="X")
        game = MockGame()
        game.display = game.MockDisplay()
        action = agent.get_action((None, 0, False), game)
        self.assertEqual(action, 3, "MouseAgent should return the action corresponding to the click.")

    def test_mouse_agent_no_action(self) -> None:
        """Test MouseAgent behavior when no click is registered."""

        class MockGame:
            class MockDisplay:
                def bind_click_handler(self, handler) -> None:
                    # Do nothing: no clicks registered
                    self.handler = handler

                def wait_for_player_action(self) -> None:
                    # Simulate no interaction
                    pass

            display = MockDisplay()

        agent = MouseAgent(player="X")
        game = MockGame()
        game.display = game.MockDisplay()

        # Mock a default fallback action to test graceful failure handling
        agent.selected_action = None
        action = agent.get_action((None, 0, False), game)

        self.assertEqual(action, -1, "MouseAgent should return -1 if no action is selected.")


class MockAgent(Agent):
    def get_action(self, state_transition, game):
        return None

class TestAgent(unittest.TestCase):
    def test_get_opponent(self):
        agent = MockAgent(player="X")
        self.assertEqual(agent.get_opponent("X"), "O")
        self.assertEqual(agent.get_opponent("O"), "X")

    def test_switching_logic(self):
        agent = RandomAgent(player="X", switching=True)
        agent.on_game_end(None)  # Simulate game end
        self.assertEqual(agent.player, "O")
        self.assertEqual(agent.opponent, "X")
