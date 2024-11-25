# type: ignore

import unittest
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from Agent import Agent, HumanAgent, MouseAgent, RandomAgent
from DeepQAgent import DeepQLearningAgent, QNetwork, ReplayBuffer
from Display import ConsoleDisplay, TicTacToeDisplay
from SymmetricMatrix import FullySymmetricMatrix, SymmetricMatrix
from TicTacToe import TicTacToe


class TestTicTacToe(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the common test environment."""
        self.agent1 = RandomAgent(player="X")
        self.agent2 = RandomAgent(player="O")
        self.display = MagicMock(spec=TicTacToeDisplay)
        self.game = TicTacToe(self.agent1, self.agent2)

    def test_initialize_board(self) -> None:
        """Test board initialization."""
        board = TicTacToe.initialize_board(3, 3)
        self.assertEqual(len(board), 9)
        self.assertTrue(all(cell == " " for cell in board))

    def test_valid_actions(self) -> None:
        """Test valid actions are calculated correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        valid_actions = TicTacToe.get_valid_actions_from_board(board)
        self.assertEqual(valid_actions, [2, 3, 6, 7, 8])

    def test_win_conditions(self) -> None:
        """Test win condition generation."""
        game = TicTacToe(self.agent1, self.agent2, rows=3, cols=3, win_length=3)
        self.assertIn([0, 1, 2], game.win_conditions)  # Horizontal win
        self.assertIn([0, 3, 6], game.win_conditions)  # Vertical win
        self.assertIn([0, 4, 8], game.win_conditions)  # Diagonal win

    def test_is_won(self) -> None:
        """Test win condition detection."""
        self.game.board = ["X", "X", "X", "O", " ", "O", " ", " ", " "]
        self.assertTrue(self.game.is_won("X"))
        self.assertFalse(self.game.is_won("O"))

    def test_is_draw(self) -> None:
        """Test draw detection."""
        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game.is_draw())

    def test_make_move(self) -> None:
        """Test making valid and invalid moves."""
        self.game.board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game.current_player = "O"
        self.assertTrue(self.game.make_move(2))  # Valid move
        self.assertFalse(self.game.make_move(0))  # Invalid move

    def test_switch_player(self) -> None:
        """Test player switching logic."""
        self.game.current_player = "X"
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "O")
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "X")

    def test_game_over(self) -> None:
        """Test game-over detection."""
        self.game.board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertTrue(self.game.is_game_over())  # Win detected

        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game.is_game_over())  # Draw detected

    def test_get_outcome(self) -> None:
        """Test outcome determination."""
        self.game.board = ["X", "X", "X", "O", "O", " ", " ", " ", " "]
        self.assertEqual(self.game.get_outcome(), "X")

        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertEqual(self.game.get_outcome(), "D")

    def test_display_board(self) -> None:
        """Test board display updates."""
        board = ["X", "O", " ", "X", "O", "X", " ", " ", " "]
        self.game.display = self.display
        self.game.display_board(board)
        self.display.update_display.assert_called_with(board, None)

    def test_terminal_rewards(self) -> None:
        """Test terminal rewards calculation."""
        rewards = self.game.get_terminal_rewards("X")
        self.assertEqual(rewards, (1.0, -1.0))

        rewards = self.game.get_terminal_rewards("O")
        self.assertEqual(rewards, (-1.0, 1.0))

        rewards = self.game.get_terminal_rewards("D")
        self.assertEqual(rewards, (0.0, 0.0))

    def test_agent_interaction(self) -> None:
        """Test interaction with agents."""
        self.agent1.get_action = MagicMock(return_value=0)
        self.agent2.get_action = MagicMock(return_value=1)

        outcome = self.game.play()
        self.agent1.get_action.assert_called()
        self.agent2.get_action.assert_called()
        self.assertIn(outcome, ["X", "O", "D"])

    def test_mouse_agent_requires_display(self) -> None:
        """Test MouseAgent raises an error without a display."""
        agent1 = MouseAgent(player="X")
        display = ConsoleDisplay(rows=3, cols=3, waiting_time=0.25)
        with self.assertRaises(ValueError):
            TicTacToe(agent1, self.agent2, display=display)

    def test_non_quadratic_board(self) -> None:
        """Test non-quadratic board raises an error."""
        with self.assertRaises(ValueError):
            TicTacToe(self.agent1, self.agent2, rows=3, cols=4)


class TestDisplay(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the common test environment."""
        self.rows = 3
        self.cols = 3
        self.board = ["X", "O", " ", " ", "X", "O", " ", " ", "X"]
        self.empty_board = [" " for _ in range(self.rows * self.cols)]
        self.outcome_x: Literal["X", "O", "D"] | None = "X"
        self.outcome_draw: Literal["X", "O", "D"] | None = "D"

    @patch("time.sleep", return_value=None)  # To skip delays during tests
    def test_console_display_outcome(self, _) -> None:
        """Test the ConsoleDisplay handles outcomes correctly."""
        display = ConsoleDisplay(rows=self.rows, cols=self.cols)
        with patch("builtins.print") as print_mock:
            display.update_display(self.board, outcome=self.outcome_x)
            print_mock.assert_any_call("Player X wins!")

            display.update_display(self.board, outcome=self.outcome_draw)
            print_mock.assert_any_call("It's a draw!")

    def test_console_display_message(self) -> None:
        """Test ConsoleDisplay message setting."""
        display = ConsoleDisplay(rows=self.rows, cols=self.cols)
        with patch("builtins.print") as print_mock:
            display.set_message("Player X's turn")
            print_mock.assert_called_with("Player X's turn")

    def test_tkinter_display_initialization(self) -> None:
        """Test TicTacToeDisplay initializes labels correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        self.assertEqual(len(display.labels), self.rows * self.cols)
        for label in display.labels:
            self.assertEqual(label.cget("text"), " ")

    def test_tkinter_display_update(self) -> None:
        """Test TicTacToeDisplay updates board correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        for label in display.labels:
            label.config = MagicMock()

        display.update_display(self.board)
        for i, label in enumerate(display.labels):
            expected_text = self.board[i] if self.board[i] in ["X", "O"] else " "
            label.config.assert_called_with(text=expected_text)

    def test_tkinter_display_outcome(self) -> None:
        """Test TicTacToeDisplay handles outcomes correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.set_message = MagicMock()

        display.update_display(self.board, outcome=self.outcome_x)
        display.set_message.assert_called_with("Player X wins!")

        display.update_display(self.board, outcome=self.outcome_draw)
        display.set_message.assert_called_with("It's a draw!")

    def test_tkinter_display_message(self) -> None:
        """Test TicTacToeDisplay sets messages correctly."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.message_label.config = MagicMock()

        display.set_message("Player X's turn")
        display.message_label.config.assert_called_with(text="Player X's turn")

    def test_tkinter_wait_for_action(self) -> None:
        """Test TicTacToeDisplay waits for player action."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.wait_variable = MagicMock()

        display.wait_for_player_action()
        display.wait_variable.assert_called_with(display.action_complete)

    def test_tkinter_click_handler(self) -> None:
        """Test TicTacToeDisplay click handler functionality."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        mock_handler = MagicMock()
        display.bind_click_handler(mock_handler)

        event_mock = MagicMock()
        display.handle_click(event_mock, 5)
        mock_handler.assert_called_with(5)

    def test_tkinter_display_quit_on_outcome(self) -> None:
        """Test TicTacToeDisplay calls quit after an outcome."""
        display = TicTacToeDisplay(rows=self.rows, cols=self.cols)
        display.quit = MagicMock()

        display.update_display(self.board, outcome=self.outcome_x)
        display.quit.assert_called_once()


class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer class."""

    def test_initialization(self) -> None:
        """Test ReplayBuffer initialization with proper size and device allocation."""
        buffer = ReplayBuffer(size=10, state_dim=4, device="cpu")
        self.assertEqual(buffer.size, 10, "Buffer size should be correctly initialized.")
        self.assertEqual(buffer.device, "cpu", "Buffer device should be correctly initialized.")
        self.assertEqual(len(buffer), 0, "Buffer should initially have zero stored experiences.")
        self.assertTrue(torch.is_tensor(buffer.states), "States should be stored as a torch tensor.")

    def test_add_experience(self) -> None:
        """Ensure experiences are added correctly, including overwriting behavior."""
        buffer = ReplayBuffer(size=3, state_dim=4, device="cpu")
        for i in range(5):  # Add more experiences than the buffer size
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        self.assertEqual(len(buffer), 3, "Buffer size should not exceed its capacity.")
        self.assertTrue(
            torch.equal(buffer.states[0], torch.tensor([3, 4, 5, 6], dtype=torch.float32)),
            "Oldest experiences should be overwritten in circular fashion.",
        )
        self.assertTrue(
            torch.equal(buffer.states[1], torch.tensor([4, 5, 6, 7], dtype=torch.float32)),
            "Oldest experiences should be overwritten in circular fashion.",
        )
        self.assertFalse(buffer.dones[0], "Stored 'done' value should match the input.")

    def test_sample_experiences(self) -> None:
        """Check that the sampling returns correct shapes and values."""
        buffer = ReplayBuffer(size=5, state_dim=4, device="cpu")
        for i in range(5):
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )

        batch_size = 3
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        self.assertEqual(states.shape, (batch_size, 4), "Sampled states should have correct shape.")
        self.assertEqual(actions.shape, (batch_size,), "Sampled actions should have correct shape.")
        self.assertEqual(rewards.shape, (batch_size,), "Sampled rewards should have correct shape.")
        self.assertEqual(next_states.shape, (batch_size, 4), "Sampled next states should have correct shape.")
        self.assertEqual(dones.shape, (batch_size,), "Sampled dones should have correct shape.")

    def test_buffer_length(self) -> None:
        """Verify __len__ returns the correct number of stored experiences."""
        buffer = ReplayBuffer(size=5, state_dim=4, device="cpu")
        self.assertEqual(len(buffer), 0, "Initial buffer length should be zero.")
        for i in range(7):  # Add more experiences than buffer size
            buffer.add(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=i % 2 == 0,
            )
        self.assertEqual(len(buffer), 5, "Buffer length should match its capacity after being filled.")


class TestQNetwork(unittest.TestCase):
    """Tests for the QNetwork class."""

    def test_initialization(self) -> None:
        """Test QNetwork initialization with correct input and output dimensions."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)

        self.assertEqual(model.fc[0].in_features, input_dim, "Input dimension of the first layer is incorrect.")
        self.assertEqual(model.fc[-1].out_features, output_dim, "Output dimension of the last layer is incorrect.")

    def test_forward_pass(self) -> None:
        """Ensure the forward pass produces output of the correct shape."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))  # Batch of 5 inputs
        output = model(test_input)

        self.assertEqual(output.shape, (5, output_dim), "Output shape is incorrect.")

    def test_gradient_flow(self) -> None:
        """Confirm that gradients flow correctly during backpropagation."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))
        target = torch.randn((5, output_dim))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")

    def test_parameter_count(self) -> None:
        """Verify that the number of trainable parameters is as expected."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        expected_params = (input_dim * 128 + 128) + (128 * 64 + 64) + (64 * output_dim + output_dim)
        self.assertEqual(total_params, expected_params, "The number of trainable parameters is incorrect.")

    def test_reproducibility(self) -> None:
        """Check that the network produces consistent outputs in evaluation mode."""
        input_dim, output_dim = 4, 2
        model = QNetwork(input_dim=input_dim, output_dim=output_dim)
        test_input = torch.randn((5, input_dim))

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)

        self.assertTrue(torch.equal(output1, output2), "Outputs should be consistent in evaluation mode.")


class TestDeepQLearningAgent(unittest.TestCase):
    """Tests for the DeepQLearningAgent class."""

    def setUp(self) -> None:
        """Set up common parameters and objects for the tests."""
        self.params: dict[str, Any] = {
            "player": "X",
            "switching": False,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "nr_of_episodes": 100,
            "batch_size": 32,
            "target_update_frequency": 10,
            "learning_rate": 0.001,
            "replay_buffer_length": 100,
            "wandb_logging_frequency": 5,
            "debug": False,
            "rows": 3,
            "device": "cpu",
        }
        self.agent = DeepQLearningAgent(self.params)

    def test_initialization(self) -> None:
        """Check initialization of networks, replay buffer, and parameters."""
        self.assertIsInstance(self.agent.q_network, QNetwork, "Q-network should be an instance of QNetwork.")
        self.assertIsInstance(self.agent.target_network, QNetwork, "Target network should be an instance of QNetwork.")
        self.assertIsInstance(
            self.agent.replay_buffer, ReplayBuffer, "Replay buffer should be an instance of ReplayBuffer."
        )
        self.assertEqual(self.agent.epsilon, self.params["epsilon_start"], "Epsilon should match the starting value.")
        self.assertEqual(self.agent.gamma, self.params["gamma"], "Gamma should be correctly initialized.")

    def test_action_selection_exploration(self) -> None:
        """Test action selection in exploration mode (random)."""
        self.agent.epsilon = 1.0  # Force exploration
        board = [" " for _ in range(self.params["rows"] ** 2)]
        action = self.agent.choose_action(board, epsilon=self.agent.epsilon)
        self.assertIn(action, self.agent.get_valid_actions(board), "Action should be valid for the given board.")

    def test_action_selection_exploitation(self) -> None:
        """Test action selection in exploitation mode (greedy)."""
        self.agent.epsilon = 0.0  # Force exploitation
        board = [" " for _ in range(self.params["rows"] ** 2)]
        # Mock the Q-network output to test greedy selection
        self.agent.q_network = MagicMock()
        self.agent.q_network.return_value = torch.tensor([0.1, 0.9, -0.2] + [-float("inf")] * 6)  # Mock 3 valid actions
        action = self.agent.choose_action(board, epsilon=self.agent.epsilon)
        self.assertEqual(action, 1, "Agent should select the action with the highest Q-value.")

    def test_add_experience_to_replay_buffer(self) -> None:
        """Ensure experiences are added to the replay buffer correctly."""
        board = [" " for _ in range(self.params["rows"] ** 2)]
        next_board = board[:]
        next_board[0] = "X"
        self.agent.episode_history = [(board, 0)]  # Simulate a previous move
        self.agent.get_action((next_board, 1.0, False), game=None)  # Add experience
        self.assertEqual(len(self.agent.replay_buffer), 1, "Replay buffer should contain one experience.")

    def test_training_step_updates_network(self) -> None:
        """Validate that the training step updates the Q-network weights."""
        # Fill the episode history
        board = [" " for _ in range(self.params["rows"] ** 2)]
        next_board = board[:]
        next_board[0] = "X"
        self.agent.episode_history = [(board, 0)]  # Simulate a previous move
        # Fill the replay buffer
        for _ in range(self.agent.batch_size):
            self.agent.replay_buffer.add(
                state=np.random.rand(self.params["rows"] ** 2),
                action=0,
                reward=1.0,
                next_state=np.random.rand(self.params["rows"] ** 2),
                done=False,
            )

        initial_weights = [param.clone() for param in self.agent.q_network.parameters()]
        self.agent.get_action((None, 0, True), None)  # Trigger training
        updated_weights = [param for param in self.agent.q_network.parameters()]

        for init, updated in zip(initial_weights, updated_weights):
            self.assertFalse(torch.equal(init, updated), "Weights should update after training.")

    def test_target_network_update(self) -> None:
        """Check that the target network updates at specified intervals."""
        # Mock Q-network to track its state_dict updates
        self.agent.q_network.load_state_dict = MagicMock()
        self.agent.target_network.load_state_dict = MagicMock()

        for i in range(1, 21):  # Simulate 20 episodes
            self.agent.episode_count = i
            if i % self.agent.target_update_frequency == 0:
                self.agent.get_action((None, 0, True), None)  # Simulate an action
                self.agent.target_network.load_state_dict.assert_called()
            else:
                self.agent.target_network.load_state_dict.assert_not_called()
            self.agent.target_network.load_state_dict.reset_mock()


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
            class MockDisplay(TicTacToeDisplay):
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


class TestSymmetricMatrix(unittest.TestCase):
    """Tests for the SymmetricMatrix class and its subclass TotallySymmetricMatrix."""

    def setUp(self) -> None:
        """Set up reusable objects for tests."""
        self.default_value = 0.0
        self.rows = 3
        self.matrix = SymmetricMatrix(default_value=self.default_value, rows=self.rows)
        self.totally_symmetric_matrix = FullySymmetricMatrix(default_value=self.default_value, rows=self.rows)

    def test_initialization(self) -> None:
        """Test initialization of SymmetricMatrix."""
        self.assertEqual(
            self.matrix.default_value, self.default_value, "Default value should be correctly initialized."
        )
        self.assertEqual(self.matrix.rows, self.rows, "Matrix size (rows) should be correctly initialized.")

    def test_canonical_board(self) -> None:
        """Validate that boards are transformed to canonical forms correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        canonical_board = self.matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent.",
        )

    def test_canonical_action(self) -> None:
        """Ensure actions are canonicalized consistently."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2  # Assume an action in the original board
        canonical_action = self.matrix.get_canonical_action(board, action)
        inverse_action = self.matrix.get_inverse_canonical_action(board, canonical_action)
        self.assertEqual(action, inverse_action, "Canonical action should map back to the original action.")

    def test_value_storage_and_retrieval(self) -> None:
        """Verify that values are stored and retrieved correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        value = 10.0

        self.matrix.set(board, action, value)
        retrieved_value = self.matrix.get(board, action)
        self.assertEqual(retrieved_value, value, "Stored value should match the retrieved value.")

        # Check symmetry handling
        symmetries = self.matrix._generate_symmetries(board)
        canonical_action = self.matrix.get_canonical_action(board, action)
        for symmetry in symmetries:
            symmetry_action = self.matrix.get_inverse_canonical_action(symmetry, canonical_action)
            self.assertEqual(
                self.matrix.get(symmetry, symmetry_action),
                value,
                "Value should be consistent across symmetric representations of the board.",
            )

    def test_empty_positions(self) -> None:
        """Ensure that empty positions on the board are identified correctly."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        expected_empty_positions = [2, 3, 6, 7, 8]
        empty_positions = self.matrix.get_empty_positions(board)
        self.assertListEqual(
            empty_positions,
            expected_empty_positions,
            "Empty positions should match the expected positions.",
        )

    def test_totally_symmetric_matrix_behavior(self) -> None:
        """Ensure TotallySymmetricMatrix behaves as expected."""
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        value = 15.0

        self.totally_symmetric_matrix.set(board, action, value)
        retrieved_value = self.totally_symmetric_matrix.get(board, action)
        self.assertEqual(
            retrieved_value, value, "Stored value should match the retrieved value in TotallySymmetricMatrix."
        )

        # Check that symmetries in TotallySymmetricMatrix are handled as expected
        canonical_board = self.totally_symmetric_matrix.get_canonical_board(board)
        self.assertEqual(
            canonical_board,
            self.totally_symmetric_matrix.get_canonical_board(canonical_board),
            "Canonical board transformation should be idempotent in TotallySymmetricMatrix.",
        )

    def test_totally_symmetric_matrix_behavior_2(self) -> None:
        """Ensure that different actions leading to equal next boards yield the same Q-value."""
        board_1 = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action_1 = 2
        # next_board_1 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_1 = 15.0

        self.totally_symmetric_matrix.set(board_1, action_1, value_1)

        board_2 = [" ", "O", "X", " ", "X", "O", " ", " ", " "]
        action_2 = 0
        # next_board_2 = ["X", "O", "X", " ", "X", "O", " ", " ", " "]
        value_2 = self.totally_symmetric_matrix.get(board_2, action_2)
        self.assertEqual(value_1, value_2, "Different actions leading to equal next boards yield the same Q-value.")


class TestIntegration(unittest.TestCase):
    """Integration tests for the Tic-Tac-Toe project."""

    def test_random_agent_vs_random_agent(self) -> None:
        """Simulate a game between two RandomAgents."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2)

        outcome = game.play()
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be a win, loss, or draw.")

    def test_deep_q_agent_training(self) -> None:
        """Simulate training of a DeepQLearningAgent during gameplay."""
        params: dict[str, Any] = {
            "player": "X",
            "switching": False,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.1,
            "nr_of_episodes": 10,
            "batch_size": 32,
            "target_update_frequency": 2,
            "learning_rate": 0.001,
            "replay_buffer_length": 100,
            "wandb_logging_frequency": 5,
            "debug": False,
            "rows": 3,
            "device": "cpu",
        }
        agent1 = DeepQLearningAgent(params)
        agent2 = RandomAgent(player="O")
        game = TicTacToe(agent1, agent2)

        # Simulate multiple episodes to test training
        for episode in range(10):
            outcome = game.play()
            self.assertIn(outcome, ["X", "O", "D"], f"Game outcome in episode {episode} should be valid.")

        self.assertGreater(len(agent1.replay_buffer), 0, "Replay buffer should contain experiences after training.")

    def test_console_display_updates(self) -> None:
        """Ensure the ConsoleDisplay updates correctly during a game."""
        agent1 = RandomAgent(player="X")
        agent2 = RandomAgent(player="O")
        display = ConsoleDisplay(rows=3, cols=3)
        game = TicTacToe(agent1, agent2, display=display)

        display.update_display = MagicMock()  # Mock the display update method
        game.play()

        # Verify the display was updated at least once per move
        self.assertGreaterEqual(display.update_display.call_count, 1, "Display should update during the game.")

    def test_symmetric_matrix_usage(self) -> None:
        """Verify that SymmetricMatrix is used correctly in a game."""
        matrix = SymmetricMatrix(default_value=0.0, rows=3)
        board = ["X", "O", " ", " ", "X", "O", " ", " ", " "]
        action = 2
        matrix.set(board, action, 5.0)

        # Simulate a game where matrix is used for Q-value updates
        self.assertEqual(matrix.get(board, action), 5.0, "SymmetricMatrix should retrieve stored values correctly.")

    def test_human_agent_interaction(self) -> None:
        """Mock HumanAgent to simulate user input during a game."""
        agent1 = RandomAgent(player="X")
        agent2 = HumanAgent(player="O")
        game = TicTacToe(agent1, agent2)

        # Mock HumanAgent's input to simulate user actions
        agent2.get_action = MagicMock(return_value=0)
        outcome = game.play()

        self.assertTrue(agent2.get_action.called, "HumanAgent should be called for actions during the game.")
        self.assertIn(outcome, ["X", "O", "D"], "Game outcome should be valid.")


if __name__ == "__main__":
    unittest.main()
