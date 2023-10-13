import unittest

import torch

from python_util.torch_utils.running_average import running_average


class TestRollingAvg(unittest.TestCase):
    def test_pt_rolling_avg(self):
        out = running_average(10)
        out(torch.ones(100))
        next = out(torch.full([100], 2.0))
        assert all(next == 1.5 for next in next)


if __name__ == '__main__':
    unittest.main()
