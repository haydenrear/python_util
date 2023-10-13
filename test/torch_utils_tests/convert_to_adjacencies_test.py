from unittest import TestCase

import torch

from python_util.torch_utils.torch_geometric_utils import top_k_adjacency_matrix, convert_to_adjacencies


class TestTopKAdjacency(TestCase):

    def test_create_adjacency_matrix(self):
        # Create a test model instance

        # Example input data dict with probabilities for each node
        input_data = {
            'node1': torch.tensor([0.2, 0.3, 0.5]),
            'node2': torch.tensor([0.4, 0.2, 0.4]),
            'node3': torch.tensor([0.1, 0.2, 0.7]),
            'node4': torch.tensor([0.8, 0.1, 0.1]),
            'node5': torch.tensor([0.3, 0.5, 0.2]),
            'node6': torch.tensor([0.6, 0.3, 0.1]),
            'node7': torch.tensor([1.0, 0.3, 0.1]),
            'node8': torch.tensor([1.0, 0.3, 0.1]),
            'node9': torch.tensor([1.0, 0.3, 0.1]),
        }

        # Expected adjacency matrix
        expected_adjacency_matrix = torch.tensor(
            [[0., 0., 1., 0., 0.],
             [1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0.],
             [1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0.],
             [0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 1.],
             [0., 0., 0., 1., 0.]]
        ).float()



        # Compute the adjacency matrix
        adjacency_matrix = top_k_adjacency_matrix(5, input_data)
        print(adjacency_matrix)

        # Check if the computed adjacency matrix matches the expected matrix
        assert torch.allclose(adjacency_matrix, expected_adjacency_matrix), "Test failed: Incorrect adjacency matrix"

        out = convert_to_adjacencies(adjacency_matrix)
        assert out == {2: [0, 2], 0: [1, 3], 1: [4, 5], 3: [6, 8], 4: [7]}

