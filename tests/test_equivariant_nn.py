# type: ignore

import unittest
from typing import Any

import numpy as np
import torch

from TicTacToe.EquivariantNN import (
    EquivariantLayer,
    EquivariantNN,
    get_bias_pattern,
    get_weight_pattern,
    permutation_matrix,
)


class TestEquivariantLayer(unittest.TestCase):
    """Tests for the EquivariantLayer class."""

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
        self.groupMatrices = [B0, B1, B2, B3, B4, B5, B6, B7]
        # self.groupMatrices = [B0, B2]

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

    def test_get_weight_pattern(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.float32)

        unique_elements = set(weight_pattern.detach().numpy().flatten())
        self.assertNotIn(0, unique_elements)
        self.assertEqual(len(unique_elements), 15)

    def test_get_weight_pattern_large_m(self) -> None:
        n, m = 1, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.float32)

        unique_elements = set(weight_pattern.detach().numpy().flatten())
        self.assertNotIn(0, unique_elements)
        self.assertEqual(len(unique_elements), 66)

    def test_get_weight_pattern_large_n(self) -> None:
        n, m = 3, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.float32)

        unique_elements = set(weight_pattern.detach().numpy().flatten())

        self.assertEqual(len(unique_elements), 66)
        self.assertNotIn(0, unique_elements)

    def test_get_weight_pattern_large_n_large_m(self) -> None:
        n, m = 3, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.float32)

        unique_elements = set(weight_pattern.detach().numpy().flatten())
        self.assertEqual(len(unique_elements), 325)
        self.assertNotIn(0, unique_elements)

    def test_get_weight_pattern_very_large_n_large_m(self) -> None:
        n, m = 5, 3
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        self.assertEqual(weight_pattern.shape, (input_dim, output_dim))
        self.assertEqual(weight_pattern.dtype, torch.float32)

        unique_elements = set(weight_pattern.detach().numpy().flatten())
        self.assertEqual(len(unique_elements), 780)
        self.assertNotIn(0, unique_elements)

    def test_get_bias_pattern(self) -> None:
        m = 1
        output_dim = (2 * m + 1) ** 2

        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        self.assertEqual(bias_pattern.shape, (output_dim,))
        self.assertEqual(bias_pattern.dtype, torch.float32)

        unique_elements = set(bias_pattern.detach().numpy().flatten())
        self.assertEqual(len(unique_elements), 3)
        self.assertNotIn(0, unique_elements)

    def test_EquivariantLayer(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        # Initialize the custom layer
        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        layer = EquivariantLayer(
            input_dim=input_dim, output_dim=output_dim, weight_pattern=weight_pattern, bias_pattern=bias_pattern
        )

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

    def test_EquivariantLayer_masked(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        # Initialize the custom layer
        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        weight_pattern[:, 4] = 0
        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        bias_pattern[:] = 0
        layer = EquivariantLayer(
            input_dim=input_dim, output_dim=output_dim, weight_pattern=weight_pattern, bias_pattern=bias_pattern
        )

        layer_weight_pattern = list(layer.weight_pattern.detach().numpy().flatten().astype(np.int64))
        list_weight_pattern = list(weight_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_weight_pattern, list_weight_pattern)

        layer_bias_pattern = list(layer.bias_pattern.detach().numpy().flatten().astype(np.int64))
        list_bias_pattern = list(bias_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_bias_pattern, list_bias_pattern)

        x = torch.ones(1, input_dim)  # Batch size 2, input_dim
        output = layer(x)
        self.assertEqual(output[0, 4], 0.0)

        x = torch.zeros(1, input_dim)  # Batch size 2, input_dim
        output = layer(x)
        self.assertTrue(all([out == 0.0 for out in output[0]]))

        # Test the layer with input data
        x = torch.randn(20, input_dim)  # Batch size 2, input_dim
        output = layer(x)
        for transform in self.transformations:
            P = torch.tensor(
                permutation_matrix(
                    transform(np.arange((2 * n + 1) ** 2, dtype=np.int64).reshape(2 * n + 1, 2 * n + 1)).flatten()
                ),
                dtype=torch.float32,
            )
            self.assertAlmostEqual(torch.linalg.norm(layer(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=5)

        weight_pattern = get_weight_pattern(self.groupMatrices, n, m)
        bias_pattern = get_bias_pattern(self.groupMatrices, m)
        bias_pattern[4] = 0
        layer = EquivariantLayer(
            input_dim=input_dim, output_dim=output_dim, weight_pattern=weight_pattern, bias_pattern=bias_pattern
        )

        layer_weight_pattern = list(layer.weight_pattern.detach().numpy().flatten().astype(np.int64))
        list_weight_pattern = list(weight_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_weight_pattern, list_weight_pattern)

        layer_bias_pattern = list(layer.bias_pattern.detach().numpy().flatten().astype(np.int64))
        list_bias_pattern = list(bias_pattern.detach().numpy().flatten().astype(np.int64))
        self.assertListEqual(layer_bias_pattern, list_bias_pattern)

        x = torch.zeros(1, input_dim)  # Batch size 2, input_dim
        output = layer(x)
        self.assertTrue(output[0, 4] == 0.0)

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
                torch.linalg.norm(equivariant_nn(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=4
            )
