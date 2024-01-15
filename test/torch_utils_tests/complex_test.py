import unittest

import torch

from python_util.torch_utils.complex_torch import init_complex_weights


class TestComplex(unittest.TestCase):
    def test_init_complex(self):
        out = init_complex_weights((20, 1024), 10)
        out = torch.tensor(out, dtype=torch.complex64)
        print(out)


if __name__ == '__main__':
    unittest.main()
