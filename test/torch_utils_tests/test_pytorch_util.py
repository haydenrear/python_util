import unittest

import torch

from python_util.torch_utils.pytorch_util import insert_at_indices, drop_indices, insert_at_indices_no_proj, \
    split_tensor, \
    reconstruct_tensor, copy_tensor_to


class PytorchUtilTest(unittest.TestCase):

    def test_copy(self):
        first = torch.tensor([10, 20])
        second = torch.tensor([20, 20])
        copy_tensor_to(first, second)
        assert torch.allclose(first, second)
        first = torch.tensor([10, 20], dtype=torch.float)
        first.requires_grad_(True)
        second = torch.tensor([20, 20], dtype=torch.int)
        copy_tensor_to(second, first)
        assert torch.allclose(first, second)
        assert second.requires_grad

    def test_split(self):
        # Test 1: tensor of size 10, split size 2
        tensor = torch.arange(10)
        embedding_size = torch.Size([2])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        assert len(split_dict) == len(index_dict) == 5  # We should have 5 splits
        for i, (start, end) in index_dict.items():
            assert torch.all(split_dict[i] == tensor[start:end])  # The values should match the original tensor

        # Test 2: tensor of size 100, split size 20
        tensor = torch.arange(100)
        embedding_size = torch.Size([20])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        assert len(split_dict) == len(index_dict) == 5  # We should have 5 splits
        for i, (start, end) in index_dict.items():
            assert torch.all(split_dict[i] == tensor[start:end])  # The values should match the original tensor

        # Test 3: tensor of size 10, split size 3
        tensor = torch.arange(10)
        embedding_size = torch.Size([3])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        assert len(split_dict) == 4  # We should have 4 splits
        assert len(index_dict) == 4  # We should have 4 splits
        for i, (start, end) in index_dict.items():
            assert torch.all(split_dict[i] == tensor[start:end])  # The values should match the original tensor

        tensor = torch.rand([30, 768])
        split_dict, index_dict = split_tensor(tensor, 10)
        print(split_dict)

    def test_reconstruct_tensor(self):
        # Test 1: tensor of size 10, split size 2
        tensor = torch.arange(10)
        embedding_size = torch.Size([2])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        reconstructed_tensor = reconstruct_tensor(split_dict, index_dict)
        assert torch.all(tensor == reconstructed_tensor)

        # Test 2: tensor of size 100, split size 20
        tensor = torch.arange(100)
        embedding_size = torch.Size([20])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        reconstructed_tensor = reconstruct_tensor(split_dict, index_dict)
        assert torch.all(tensor == reconstructed_tensor)

        # Test 3: tensor of size 10, split size 3
        tensor = torch.arange(10)
        embedding_size = torch.Size([3])
        split_dict, index_dict = split_tensor(tensor, embedding_size[0])
        reconstructed_tensor = reconstruct_tensor(split_dict, index_dict)
        assert torch.all(tensor == reconstructed_tensor)

    def test_insert_at_indices(self):
        first = torch.rand([100, 100])
        indices = torch.tensor([[2, 2]])
        out_first = insert_at_indices(first, indices, 1000)
        assert out_first[2][2] == 1000
        assert out_first.shape[0] == 101

        first = torch.rand([5, 5, 5])
        indices = torch.tensor([[0], [1]])
        out = insert_at_indices(first, indices, 1000)
        assert out[0][0][0] == 1000
        assert out[3][3][3] != 1000
        assert out.shape[0] == 7

        out_first = drop_indices(out_first, torch.tensor([[0], [1]]))
        assert out_first.shape[0] == 99

        out = drop_indices(out, torch.tensor([[0, 0, 0], [0, 0, 1]]))
        assert out.shape[0] == 5

        out_again = insert_at_indices(torch.rand(100), torch.tensor([[0], [1]]), 1000)
        assert out_again[0] == 1000
        assert out_again[1] == 1000
        assert out_again[2] != 1000

        out_again = insert_at_indices(torch.rand(100), torch.tensor([[0], [1]]), torch.tensor([1000.0, 2000.0]))
        assert out_again[0] == 1000
        assert out_again[1] == 2000
        assert out_again[2] != 1000

        out_again = insert_at_indices(torch.rand([100, 100]),
                                      torch.tensor([[0], [1]]),
                                      torch.tensor([
                                          [i * 1000 for i in range(100)],
                                          [i * 1000 for i in range(100)]
                                      ], dtype=torch.float))
        for i in range(100):
            assert out_again[0][i] == 1000 * i
        assert out_again[20][0] != 1000

    def test_no_proj_3d(self):
        out_again = insert_at_indices_no_proj(torch.rand([4, 4]),
                                              torch.tensor([[[0, 0], [0, 1], [0, 2], [0, 3]]]),
                                              torch.tensor([
                                                  [i * 1000 for i in range(4)],
                                                  [i * 1000 for i in range(4)]
                                              ]))
        for i in range(4):
            assert out_again[0][i] == 1000 * i
        assert out_again[1][0] != 1000

        values_tensor = torch.ones([2, 2, 2])
        out_again = insert_at_indices_no_proj(torch.rand([2, 2, 2]),
                                              torch.tensor([
                                                  [
                                                      [
                                                          [0, 0],
                                                          [0, 1]
                                                      ],
                                                      [
                                                          [0, 0],
                                                          [0, 1]
                                                      ]
                                                  ]
                                              ]),
                                              values_tensor)
        assert out_again[0][0][0] == 1
        assert out_again[0][0][1] == 1
        assert out_again[1][0][0] != 1
