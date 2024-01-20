import unittest

import torch

from python_util.torch_utils.complex_torch import init_complex_weights, complex_attn_mask, complex_boltzmann_prob


class TestComplex(unittest.TestCase):
    def test_init_complex(self):
        out = init_complex_weights((20, 1024), 10)
        out = torch.tensor(out, dtype=torch.complex64)
        print(out)

    def test_complex_attn(self):
        out = complex_attn_mask(torch.ones([100, 100]).tril())
        print(out)
        # torch.view_as_complex(torch.ones(100, 100).unsqueeze(2).expand(100, 100, 2))
        ones = torch.ones([100, 100, 2])
        ones = torch.view_as_complex(ones)
        print(ones)
        # print(torch.matmul(out, ones))

    def test_complex_boltzmann(self):
        out = complex_boltzmann_prob(torch.rand([1, 1, 512, 512], dtype=torch.complex64), 3)
        print(torch.matmul(out, torch.rand([1, 1, 512, 512], dtype=torch.complex64)))


if __name__ == '__main__':
    unittest.main()
