import unittest

import torch

from python_util.graph_util.graph_utils import add_connections
from python_util.torch_utils.torch_geometric_util import create_star_edge_index, create_fully_connected_edge_list


class GraphTests(unittest.TestCase):
    def test_create_star_edge_index(self):
        edge_index = create_star_edge_index(3)
        print(edge_index)
        assert_index = torch.Tensor([[1, 2, 0, 0],
                                     [0, 0, 1, 2]])
        assert edge_index[0].numpy().tolist() == assert_index[0].numpy().tolist()
        assert edge_index[1].numpy().tolist() == assert_index[1].numpy().tolist()

        edge_index = create_star_edge_index(3, True)
        assert_index = torch.Tensor([[1, 2, 0, 0, 0, 1, 2],
                                     [0, 0, 1, 2, 0, 1, 2]])
        assert edge_index[0].numpy().tolist() == assert_index[0].numpy().tolist()
        assert edge_index[1].numpy().tolist() == assert_index[1].numpy().tolist()

    def test_create_fully_connected_adjacency(self):
        edge_index = create_fully_connected_edge_list(3)
        assert_index = torch.Tensor([[0, 0, 1, 1, 2, 2],
                                     [1, 2, 0, 2, 0, 1]])

        assert edge_index[0].numpy().tolist() == assert_index[0].numpy().tolist()
        assert edge_index[1].numpy().tolist() == assert_index[1].numpy().tolist()

    def test_add_edges(self):
        edgelist = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=torch.long)
        min_edges = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Add connections
        add_connections(edgelist.to(device), min_edges, device)


if __name__ == '__main__':
    unittest.main()
