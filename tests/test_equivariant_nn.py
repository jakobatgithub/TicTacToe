# type: ignore

import unittest
from typing import Any

import numpy as np
import torch

from TicTacToe.EquivariantNN import EquivariantLayer, get_mask_tied_groups, get_matrix_pattern, permutation_matrix


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

    def test_get_matrix_pattern(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2

        matrix_pattern = get_matrix_pattern(self.groupMatrices, n, m)

        self.assertEqual(matrix_pattern.shape, (input_dim, output_dim))
        self.assertEqual(matrix_pattern.dtype, np.int64)

        unique_elements = set(matrix_pattern.flatten())

        self.assertEqual(len(unique_elements), 15)
        self.assertNotIn(0, unique_elements)

    def test_get_mask_tied_groups(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2
        mask, tied_groups = get_mask_tied_groups(self.groupMatrices, n, m)

        self.assertEqual(mask.shape, (input_dim, output_dim))
        self.assertEqual(mask.dtype, second=torch.int64)

        unique_elements = set(list(mask.detach().numpy().flatten()))

        self.assertEqual(len(unique_elements), 1)
        self.assertIn(1, unique_elements)

        self.assertEqual(len(tied_groups), 15)
        for group in tied_groups:
            for i, j in group:
                self.assertEqual(mask[i, j], 1)

        matrix_pattern = get_matrix_pattern(self.groupMatrices, n, m)
        for group in tied_groups:
            (i0, j0) = group[0]
            first = matrix_pattern[i0, j0]
            self.assertTrue(all(first == matrix_pattern[i, j] for i, j in group))

    def test_EquivariantLayer(self) -> None:
        n, m = 1, 1
        input_dim, output_dim = (2 * n + 1) ** 2, (2 * m + 1) ** 2
        mask, tied_groups = get_mask_tied_groups(self.groupMatrices, n, m)

        # Initialize the custom layer
        layer = EquivariantLayer(input_dim=input_dim, output_dim=output_dim, mask=mask, tied_groups=tied_groups)
        weight_matrix = list(layer.weight_matrix.detach().numpy().flatten().astype(np.int64))
        matrix_pattern = list(get_matrix_pattern(self.groupMatrices, n, m).flatten())
        self.assertListEqual(weight_matrix, matrix_pattern)

        # Test the layer with input data
        x = torch.randn(2, input_dim)  # Batch size 2000, input_dim
        output = layer(x)

        for transform in self.transformations:
            P = torch.tensor(
                permutation_matrix(
                    transform(np.arange((2 * n + 1) ** 2, dtype=np.int64).reshape(2 * n + 1, 2 * n + 1)).flatten()
                ),
                dtype=torch.float32,
            )
            self.assertAlmostEqual(torch.linalg.norm(layer(x @ P.T) - output @ P.T).detach().numpy(), 0.0, places=5)
