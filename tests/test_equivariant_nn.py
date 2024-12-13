# type: ignore

import unittest
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from TicTacToe.EquivariantNN import (
    EquivariantLayer,
    EquivariantNN,
    get_bias_pattern,
    get_number_of_unique_elements,
    get_weight_pattern,
    permutation_matrix,
)


class TestGetWeightPattern(unittest.TestCase):
    def setUp(self) -> None:
        # Define transformation matrices
        B0 = [[1, 0], [0, 1]]
        B1 = [[-1, 0], [0, -1]]
        B2 = [[-1, 0], [0, 1]]
        B3 = [[1, 0], [0, -1]]
        B4 = [[0, 1], [1, 0]]
        B5 = [[0, -1], [1, 0]]
        B6 = [[0, 1], [-1, 0]]
        B7 = [[0, -1], [-1, 0]]
        Bs = [B0, B1, B2, B3, B4, B5, B6, B7]
        # Bs = [B0, B2]
        self.groupMatrices = [np.array(B) for B in Bs]

    def test_get_weight_pattern(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.int64)
        self.assertEqual(get_number_of_unique_elements(weight_pattern), 15)

    def test_get_weight_pattern_large_m(self) -> None:
        n, m = 1, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.int64)
        self.assertEqual(get_number_of_unique_elements(weight_pattern), 66)

    def test_get_weight_pattern_large_n(self) -> None:
        n, m = 3, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.int64)
        self.assertEqual(get_number_of_unique_elements(weight_pattern), 66)

    def test_get_weight_pattern_large_n_large_m(self) -> None:
        n, m = 3, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.int64)
        self.assertEqual(get_number_of_unique_elements(weight_pattern), 325)

    def test_get_weight_pattern_very_large_n_large_m(self) -> None:
        n, m = 5, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.int64)
        self.assertEqual(get_number_of_unique_elements(weight_pattern), 780)

    def test_get_bias_pattern(self) -> None:
        m = 1
        output_dim = (2 * m + 1) ** 2

        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        self.assertEqual(bias_pattern.shape, (output_dim,))
        self.assertEqual(bias_pattern.dtype, torch.int64)

        unique_elements = set(bias_pattern.detach().numpy().flatten())
        self.assertEqual(len(unique_elements), 3)
        self.assertNotIn(0, unique_elements)


class TestEquivariantLayer(unittest.TestCase):
    def setUp(self) -> None:
        # Define transformation matrices
        B0 = [[1, 0], [0, 1]]
        B1 = [[-1, 0], [0, -1]]
        B2 = [[-1, 0], [0, 1]]
        B3 = [[1, 0], [0, -1]]
        B4 = [[0, 1], [1, 0]]
        B5 = [[0, -1], [1, 0]]
        B6 = [[0, 1], [-1, 0]]
        B7 = [[0, -1], [-1, 0]]
        Bs = [B0, B1, B2, B3, B4, B5, B6, B7]
        # Bs = [B0, B2]
        self.groupMatrices = [np.array(B) for B in Bs]

        self.transformations: list[Any] = [
            lambda x: x,
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.flipud(np.fliplr(x)),
            lambda x: np.transpose(x),
            lambda x: np.fliplr(np.transpose(x)),
            lambda x: np.flipud(np.transpose(x)),
            lambda x: np.flipud(np.fliplr(np.transpose(x))),
        ]

        # Example patterns for testing
        self.weight_pattern = torch.tensor([[1, 2], [3, 4]])
        self.bias_pattern = torch.tensor([1, 2])
        self.input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Initialize the layer
        self.layer = EquivariantLayer(weight_pattern=self.weight_pattern, bias_pattern=self.bias_pattern)

    def test_initialization(self):
        # Check that the number of parameters matches the unique non-zero entries
        self.assertEqual(len(self.layer.weight_params), 4)
        self.assertEqual(len(self.layer.bias_params), 2)

    def test_forward_shape(self):
        # Perform a forward pass and check the output shape
        output = self.layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_forward_computation(self):
        # Manually construct expected output for simple weights and biases
        with torch.no_grad():
            self.layer.weight_params[0].fill_(1.0)
            self.layer.weight_params[1].fill_(0.0)
            self.layer.weight_params[2].fill_(0.0)
            self.layer.weight_params[3].fill_(0.0)
            self.layer.bias_params[0].fill_(1.0)
            self.layer.bias_params[1].fill_(2.0)

        self.input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        output = self.layer(self.input_tensor)
        expected_output = torch.tensor([[1.0 * 1.0 + 1.0, 2.0], [1.0 * 3.0 + 1.0, 2.0]])
        self.assertTrue(torch.allclose(output, expected_output))

    def test_EquivariantLayer(self) -> None:
        n, m = 1, 1
        input_dim, _ = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        # Initialize the custom layer
        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        layer = EquivariantLayer(weight_pattern=weight_pattern, bias_pattern=bias_pattern)

        layer_weight_pattern = list(layer.weight_pattern.detach().numpy().flatten().astype(np.int64))
        list_weight_pattern = list(weight_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_weight_pattern, list_weight_pattern)

        layer_bias_pattern = list(layer.bias_pattern.detach().numpy().flatten().astype(np.int64))
        list_bias_pattern = list(bias_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_bias_pattern, list_bias_pattern)

        # Test the layer with input data
        x = torch.randn(2, input_dim)  # Batch size 2, input_dim
        output = layer(x)

        for transform in self.transformations:
            P = torch.tensor(
                permutation_matrix(
                    transform(np.arange((2 * n + 1) ** 2, dtype=np.int64).reshape(2 * n + 1, 2 * n + 1)).flatten()
                ),
                dtype=torch.float32,
            )
            self.assertAlmostEqual(torch.linalg.norm(layer(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=5)

    # def test_EquivariantLayer_masked(self) -> None:
    #     n, m = 1, 1
    #     input_dim, _ = (2 * n + 1) ** 2, (2 * m + 1) ** 2

    #     # Initialize the custom layer
    #     weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
    #     weight_pattern[4, :] = 0
    #     print(weight_pattern)
    #     bias_pattern = get_bias_pattern(self.groupMatrices, m)
    #     bias_pattern[:] = 0
    #     layer = EquivariantLayer(weight_pattern=weight_pattern, bias_pattern=bias_pattern)

    #     # print(f"Number of trainable weight parameters: {len(layer.weight_params)}")
    #     # print(f"Number of trainable bias parameters: {len(layer.bias_params)}")
    #     # print(f"weight_idx_mask = {layer.weight_idx_mask}")

    #     layer_weight_pattern = list(layer.weight_pattern.detach().numpy().flatten().astype(np.int64))
    #     list_weight_pattern = list(weight_pattern.detach().numpy().flatten().astype(np.int64))
    #     self.assertListEqual(layer_weight_pattern, list_weight_pattern)

    #     layer_bias_pattern = list(layer.bias_pattern.detach().numpy().flatten().astype(np.int64))
    #     list_bias_pattern = list(bias_pattern.detach().numpy().flatten().astype(np.int64))
    #     self.assertListEqual(layer_bias_pattern, list_bias_pattern)

    #     x = torch.ones(1, input_dim)  # Batch size 2, input_dim
    #     output = layer(x)

    #     self.assertEqual(output[4, 4], 0.0)

    #     x = torch.zeros(1, input_dim)  # Batch size 2, input_dim
    #     output = layer(x)
    #     self.assertTrue(all([out == 0.0 for out in output[0]]))

    #     # Test the layer with input data
    #     x = torch.randn(20, input_dim)  # Batch size 2, input_dim
    #     output = layer(x)
    #     for transform in self.transformations:
    #         P = torch.tensor(
    #             permutation_matrix(
    #                 transform(np.arange((2 * n + 1) ** 2, dtype=np.int64).reshape(2 * n + 1, 2 * n + 1)).flatten()
    #             ),
    #             dtype=torch.float32,
    #         )
    #         self.assertAlmostEqual(torch.linalg.norm(layer(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=5)

    #     weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
    #     bias_pattern = get_bias_pattern(self.groupMatrices, m)
    #     bias_pattern[4] = 0
    #     layer = EquivariantLayer(weight_pattern=weight_pattern, bias_pattern=bias_pattern)

    #     layer_weight_pattern = list(layer.weight_pattern.detach().numpy().flatten().astype(np.int64))
    #     list_weight_pattern = list(weight_pattern.detach().numpy().flatten().astype(np.int64))
    #     self.assertListEqual(layer_weight_pattern, list_weight_pattern)

    #     layer_bias_pattern = list(layer.bias_pattern.detach().numpy().flatten().astype(np.int64))
    #     list_bias_pattern = list(bias_pattern.detach().numpy().flatten().astype(np.int64))
    #     self.assertListEqual(layer_bias_pattern, list_bias_pattern)

    #     x = torch.zeros(1, input_dim)  # Batch size 2, input_dim
    #     output = layer(x)
    #     self.assertTrue(output[0, 4] == 0.0)


class TestEquivariantNN2(unittest.TestCase):
    def setUp(self) -> None:
        # Define transformation matrices
        B0 = [[1, 0], [0, 1]]
        B1 = [[-1, 0], [0, -1]]
        B2 = [[-1, 0], [0, 1]]
        B3 = [[1, 0], [0, -1]]
        B4 = [[0, 1], [1, 0]]
        B5 = [[0, -1], [1, 0]]
        B6 = [[0, 1], [-1, 0]]
        B7 = [[0, -1], [-1, 0]]
        Bs = [B0, B1, B2, B3, B4, B5, B6, B7]
        self.groupMatrices = [np.array(B) for B in Bs]

        self.transformations: list[Any] = [
            lambda x: x,
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.flipud(np.fliplr(x)),
            lambda x: np.transpose(x),
            lambda x: np.fliplr(np.transpose(x)),
            lambda x: np.flipud(np.transpose(x)),
            lambda x: np.flipud(np.fliplr(np.transpose(x))),
        ]

    def test_EquivariantNN(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        equivariant_nn = EquivariantNN(self.groupMatrices)
        x = torch.zeros(2, input_dim)
        output = equivariant_nn(x)
        self.assertEqual(output.shape, (2, output_dim))

        for transform in self.transformations:
            P = torch.tensor(
                permutation_matrix(
                    transform(np.arange((2 * n + 1) ** 2, dtype=np.int64).reshape(2 * n + 1, 2 * n + 1)).flatten()
                ),
                dtype=torch.float32,
            )
            self.assertAlmostEqual(
                torch.linalg.norm(equivariant_nn(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=3
            )


class TestEquivariantNN(unittest.TestCase):
    def setUp(self):
        # Example groupMatrices and parameters
        self.groupMatrices = [np.eye(2)]  # Simplified group matrices for testing
        self.ms = (1, 2, 2, 1)
        self.input_tensor = torch.randn(5, (2 * 1 + 1) ** 2)

        self.nn = EquivariantNN(self.groupMatrices, self.ms)

    def test_initialization(self):
        # Check the number of layers in the sequential module
        self.assertEqual(len(self.nn.fc_equivariant), 5)  # 3 layers + 2 ReLU

    def test_forward_shape(self):
        # Perform a forward pass and check the output shape
        output = self.nn(self.input_tensor)
        self.assertEqual(output.shape, (5, (2 * 1 + 1) ** 2))

    def test_forward_execution(self):
        # Ensure no errors during forward pass
        try:
            self.nn(self.input_tensor)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")


class TestEquivariantNNGradients(unittest.TestCase):
    def setUp(self):
        # Example groupMatrices and parameters
        self.groupMatrices = [np.eye(2)]  # Simplified group matrices for testing
        self.ms = (1, 2, 2, 1)
        self.input_tensor = torch.randn(2, (2 * 1 + 1) ** 2, requires_grad=True)
        self.nn = EquivariantNN(self.groupMatrices, self.ms)

    def test_gradients_exist(self):
        # Perform a forward pass
        output = self.nn(self.input_tensor)
        loss = output.sum()  # Example loss function
        loss.backward()

        # Check that gradients exist for all parameters in the network
        for param in self.nn.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_gradients_propagate_correctly(self):
        # Perform a forward pass
        output = self.nn(self.input_tensor)
        loss = (output**2).sum()  # Example loss function
        loss.backward()

        # Ensure gradients flow back to input_tensor
        self.assertIsNotNone(self.input_tensor.grad)
        self.assertTrue(torch.isfinite(self.input_tensor.grad).all())


class TestEquivariantNNTraining(unittest.TestCase):
    def setUp(self):
        # Example groupMatrices and parameters
        B0 = [[1, 0], [0, 1]]
        B1 = [[-1, 0], [0, -1]]
        B2 = [[-1, 0], [0, 1]]
        B3 = [[1, 0], [0, -1]]
        B4 = [[0, 1], [1, 0]]
        B5 = [[0, -1], [1, 0]]
        B6 = [[0, 1], [-1, 0]]
        B7 = [[0, -1], [-1, 0]]
        Bs = [B0, B1, B2, B3, B4, B5, B6, B7]
        self.groupMatrices = [np.array(B) for B in Bs]
        self.ms = (1, 2, 2, 1)
        self.input_dim = (2 * 1 + 1) ** 2
        self.output_dim = self.input_dim
        # Initialize the model
        self.nn = EquivariantNN(self.groupMatrices, self.ms)

        # Synthetic dataset
        self.x_train = torch.randn(10, self.input_dim)
        self.y_train = torch.randn(10, self.output_dim)

    def test_training_loop(self):
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.nn.parameters(), lr=0.00025)

        initial_weights = [param.clone().detach() for param in self.nn.parameters()]

        # Track loss to ensure it decreases
        initial_loss = None
        for epoch in range(15):  # Short training loop
            optimizer.zero_grad()
            outputs = self.nn(self.x_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()

            # Print loss for debugging
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            if initial_loss is None:
                initial_loss = loss.item()
            else:
                # Assert that the loss decreases (at least once)
                self.assertLess(loss.item(), initial_loss)
                initial_loss = loss.item()

        updated_weights = [param.clone().detach() for param in self.nn.parameters()]

        # Compare initial and updated weights
        for initial, updated in zip(initial_weights, updated_weights):
            assert not torch.equal(initial, updated), "Training step did not update the Q-network weights."

    def test_gradients_exist_during_training(self):
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.nn.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = self.nn(self.x_train)
        loss = criterion(outputs, self.y_train)
        loss.backward()

        # Check that gradients exist for all parameters
        for param in self.nn.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
