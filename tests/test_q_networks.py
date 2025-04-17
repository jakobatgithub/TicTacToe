# type: ignore

import unittest

import torch
import torch.nn as nn

from TicTacToe.QNetworks import QNetwork, CNNQNetwork, FullyConvQNetwork


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

        # n1, n2 = 128, 64
        n1, n2 = 49, 49
        expected_params = (input_dim * n1 + n1) + (n1 * n2 + n2) + (n2 * output_dim + output_dim)
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


class TestCNNQNetwork(unittest.TestCase):
    """Tests for the CNNQNetwork class."""

    def test_initialization(self):
        """Test CNNQNetwork initialization with correct input and output dimensions."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, rows=grid_size, output_dim=output_dim)

        self.assertEqual(
            model.conv_layers[0].in_channels, input_dim, "Input channels of the first convolutional layer are incorrect."
        )
        # Find the last Conv2d layer in the conv_layers sequence
        last_conv_layer = [layer for layer in model.conv_layers if isinstance(layer, nn.Conv2d)][-1]
        self.assertEqual(
            last_conv_layer.out_channels, 64, "Output channels of the last convolutional layer are incorrect."
        )
        self.assertEqual(
            model.fc_layers[-1].out_features, output_dim, "Output dimension of the final fully connected layer is incorrect."
        )

    def test_forward_pass(self):
        """Ensure the forward pass produces output of the correct shape."""
        batch_size, input_dim, rows = 5, 1, 3
        model = CNNQNetwork(input_dim=input_dim, rows=rows, output_dim=rows * rows)
        test_input = torch.randn((batch_size, rows * rows))  # Batch of 5 inputs
        output = model(test_input)

        self.assertEqual(output.shape, (batch_size, rows * rows), "Output shape is incorrect.")

    def test_gradient_flow(self):
        """Confirm that gradients flow correctly during backpropagation."""
        batch_size, input_dim, rows, output_dim = 5, 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, rows=rows, output_dim=output_dim)
        test_input = torch.randn((batch_size, rows * rows))
        target = torch.randn((batch_size, output_dim))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)
        loss.backward()

        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")

    def test_parameter_count(self):
        """Verify that the number of trainable parameters is as expected."""
        input_dim, grid_size, output_dim = 1, 3, 9
        model = CNNQNetwork(input_dim=input_dim, rows=grid_size, output_dim=output_dim)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate expected parameters
        conv1_params = (input_dim * 32 * 3 * 3) + 32
        conv2_params = (32 * 64 * 3 * 3) + 64
        conv3_params = (64 * 64 * 3 * 3) + 64
        fc1_params = (64 * grid_size * grid_size * 128) + 128
        fc2_params = (128 * output_dim) + output_dim
        expected_params = conv1_params + conv2_params + conv3_params + fc1_params + fc2_params

        self.assertEqual(total_params, expected_params, "The number of trainable parameters is incorrect.")


class TestFullyConvQNetwork(unittest.TestCase):
    """Tests for the FullyConvQNetwork class."""

    def test_output_dimensions(self):
        """Test that the output dimension is correct."""
        batch_size, rows = 1, 5  # Example grid size
        model = FullyConvQNetwork()

        # Create a test input tensor
        test_input = torch.zeros((batch_size, rows * rows))
        test_input[0, 4] = 1.0  # Set a single active cell in the center
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, rows * rows), "Output shape is incorrect.")

    def test_shift_invariance(self):
        """Test that shifting the input and then applying conv_layers yields the same result as applying conv_layers and then shifting the output."""
        batch_size, rows = 1, 5  # Example grid size
        model = FullyConvQNetwork()

        # Create a test input tensor
        test_input = torch.zeros((batch_size, rows * rows)).view(-1, 1, rows, rows)
        test_input[0, 0, 2, 2] = 1.0  # Set a single active cell in the center

        # Define a shift (x, y)
        shift_x, shift_y = 1, 2

        # Shift the input tensor
        shifted_input = torch.roll(test_input, shifts=(shift_x, shift_y), dims=(2, 3))

        # Apply conv_layers to both the original and shifted inputs
        original_output = model.conv_layers(test_input)
        shifted_output = model.conv_layers(shifted_input)

        # Shift the original output by the same (x, y) vector
        shifted_original_output = torch.roll(original_output, shifts=(shift_x, shift_y), dims=(2, 3))

        # Compare the results
        self.assertTrue(
            torch.allclose(shifted_output, shifted_original_output, atol=1e-7),
            "Shifting the input and then applying conv_layers should yield the same result as applying conv_layers and then shifting the output.",
        )

    def test_random_input(self):
        """Test that the network produces consistent outputs for the same random input."""
        batch_size, rows = 1, 5  # Example grid size
        model = FullyConvQNetwork()

        # Create a random input tensor
        test_input = torch.randn((batch_size, 1, rows, rows))

        # Apply the network twice
        output1 = model(test_input)
        output2 = model(test_input)

        # Assert that the outputs are identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5), "Outputs should be consistent for the same input.")

    def test_gradient_flow(self):
        """Test that gradients flow correctly during backpropagation."""
        batch_size, rows = 1, 5  # Example grid size
        model = FullyConvQNetwork()

        # Create a random input tensor
        test_input = torch.randn((batch_size, rows * rows), requires_grad=True)
        target = torch.randn((batch_size, rows * rows))

        # Define a loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Perform a forward pass
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, target)

        # Perform a backward pass
        loss.backward()

        # Check that gradients exist for all parameters
        for param in model.parameters():
            self.assertIsNotNone(param.grad, "Parameter does not have gradients.")
            self.assertTrue((param.grad != 0).any(), "Parameter gradients should not be zero.")

    def test_parameter_counts(self):
        """Test that the base and head have the expected number of parameters."""
        model = FullyConvQNetwork()

        base_params = count_trainable_params(model.base)
        head_params = count_trainable_params(model.head)
        total_params = count_trainable_params(model)

        self.assertEqual(
            base_params + head_params,
            total_params,
            "Total parameter count should equal the sum of base and head parameters."
        )

        print("\n")
        print("Number of parameters in the model FullyConvQNetwork:")
        print(f"Base parameters: {base_params}")
        print(f"Head parameters: {head_params}")
        print(f"Total parameters: {total_params}")

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)