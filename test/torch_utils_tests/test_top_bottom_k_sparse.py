import torch

from python_util.torch_utils.top_bottom_k_sparse import get_top_k_sparse_coo


def test_top_k_sparse_coo():
    out = torch.rand([100, 100])
    out = get_top_k_sparse_coo(out, 10)
    assert len(out.coalesce().values()) == 10
    assert [i for i in out.shape] == [100, 100]

    out = torch.rand([100])
    out = get_top_k_sparse_coo(out, 10)
    assert len(out.coalesce().values()) == 10
    assert [i for i in out.shape] == [100]
