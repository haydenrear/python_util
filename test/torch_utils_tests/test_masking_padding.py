from unittest import TestCase

from numpy.testing import assert_allclose
import numpy
import torch

from python_util.torch_utils.agg_fn import MeanAgg
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.masking import create_attn_mask_2d, create_attn_mask_3d, create_key_padding_mask, \
    make_causal_attn_mask_2d, \
    make_causal_attn_mask_3d, translate_huggingface_batched_attn_mask_3d, create_key_padding_mask_from_attn_mask, \
    merge_masks, merge_masks_or_threshold, apply_mask_as_value, merge_mask_src_tgt
from python_util.torch_utils.padding import pad_collapse_states_2d, pad_collapse_states_3d, pad_add_end_to_match, \
    fill_with_value_to_match
from python_util.torch_utils.split_sequences import split_sequence_get_attn_mask, split_sequence, \
    split_sequence_get_key_padding_mask

import test


class TestMaskingPadding(TestCase):

    def test_create_key_padding_mask_from_attn_mask(self):
        mask = torch.ones((20, 10, 5))
        to_add = torch.zeros((20, 10, 5))
        concatted_mask = torch.concat([mask, to_add], dim=2)
        out = create_key_padding_mask_from_attn_mask(concatted_mask, 10)
        assert all([torch.allclose(torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                dtype=torch.float), out_value)
                    for out_value in out])

    def test_apply_masking(self):
        to_mask = torch.ones((20, 30, 768))
        triangle_mask = torch.tril(torch.ones([30, 20]), diagonal=0)
        out_value = apply_mask_as_value(to_mask, triangle_mask, filter_value=0.0, to_set_value=0.0)
        assert all([torch.allclose(out_value[:, :, i], triangle_mask.T) for i in range(out_value.shape[2])])


    # def test_create_key_padding(self):
    #     out = create_key_padding_mask_from_attn_mask(torch.triu(torch.ones(20, 20, 20), diagonal=0), 5)
    #     assert torch.allclose(torch.tensor(out.shape), torch.tensor([5, 20]))

    def test_get_split_key_padding_masks(self):
        triu = torch.tril(torch.ones([20, 30]), diagonal=0)
        out = split_sequence_get_key_padding_mask(torch.rand([30, 20, 768]),
                                                  10,
                                                  triu)
        assert len(out) == 3


    def test_make_causal_mask_2d(self):
        out = make_causal_attn_mask_2d(torch.ones((20, 20)))
        assert torch.allclose(out, torch.ones((20, 20, 20)).tril())
        out = make_causal_attn_mask_2d(torch.ones((10, 20)))
        assert torch.allclose(out, torch.ones((10, 20, 20)).tril())
        zeros = torch.zeros((10, 20, 20))
        assert torch.allclose(out, torch.ones((10, 20, 20)).tril())

    def test_make_causal_mask_3d_(self):
        out = torch.rand([10, 20, 20])
        out[out > 0.2] = 1.0
        out[out < 0.2] = 0.0
        print(out)
        out = make_causal_attn_mask_3d(out)
        print(out)

    def test_merge_masks(self):
        attn = torch.nn.MultiheadAttention(768, 4)
        two = torch.ones((10, 32))
        rand = torch.rand((32, 10, 768))
        out = merge_masks(attn, rand, two)
        LoggerFacade.debug(f'{out.shape} is output size')
        out = torch.allclose(torch.tensor(out.shape), torch.tensor([40, 32, 32]))
        assert out
        two = torch.ones((10, 32, 32))
        rand = torch.rand((32, 10, 768))
        out = merge_masks(attn, rand, two)
        LoggerFacade.debug(f'{out.shape} is output size')
        out = torch.allclose(torch.tensor(out.shape), torch.tensor([40, 32, 32]))
        assert out


    def test_merge_mask(self):
        # Assume you have source and target key padding masks with the same batch size
        batch_size = 3
        src_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]]) # shape: (batch_size, src_length)
        tgt_mask = torch.tensor([[1, 1], [1, 0], [1, 1]]) # shape: (batch_size, tgt_length)

        # Get the number of attention heads
        n_heads = 4 # Change this to the actual number of attention heads in your model

        # Extend the masks along the head dimension
        src_mask = src_mask.unsqueeze(1).expand(-1, n_heads, -1)
        tgt_mask = tgt_mask.unsqueeze(1).expand(-1, n_heads, -1)

        # Reshape the masks to have a shape of [batch_size * n_heads, ...]
        src_mask = src_mask.reshape(batch_size * n_heads, -1)
        tgt_mask = tgt_mask.reshape(batch_size * n_heads, -1)

        # Use broadcasting to combine the masks
        attn_mask = src_mask.unsqueeze(-1) * tgt_mask.unsqueeze(-2)
        print(attn_mask.shape)

    def test_get(self):
        out_vals = []
        for i in range(20, 30):
            out_vals.append(torch.ones((20, i)))
        maxes = [max([out_val.shape[j] for out_val in out_vals]) for j in range(len(out_vals[0].shape))]
        print(maxes)

    def test_merge_mask_src_tgt(self):
        attn = torch.nn.MultiheadAttention(768, 4)
        out = merge_masks(attn, torch.rand([17, 5, 768]), torch.ones([5, 17, 20]))
        assert torch.allclose(torch.tensor(out.shape), torch.tensor([20, 17, 20]))

    # def test_padding(self):
    #     pad = torch.ones((10, 10, 10))
    #     to_pad = torch.ones((5, 5, 5))
    #     out = pad_add_end_to_match(pad, to_pad)
    #     assert torch.allclose(torch.tensor(pad.shape), torch.tensor(out.shape))
    #     assert torch.allclose(out[5:, 5:, 5:], torch.zeros((5,5,5)))
    #     assert torch.allclose(out[:5, :5, :5], torch.ones((5,5,5)))

    def test_merge_masking_or_threshold(self):
        first = torch.zeros([30, 20])
        second = torch.ones([30, 20])
        third = torch.triu(torch.ones([30, 20]), diagonal=0)
        out = merge_masks_or_threshold([first, second, third], 0.32)
        assert torch.allclose(out, torch.triu(torch.ones([30, 20]), diagonal=0))
        first = torch.zeros([30, 20])
        second = torch.ones([30, 17])
        third = torch.triu(torch.ones([30, 20]), diagonal=0)
        out = merge_masks_or_threshold([first, second, third], 0.32)
        assert out is not None

    def test_pad_fill(self):
        ones = torch.ones([20, 20, 768], dtype=torch.float)
        out = fill_with_value_to_match([30, 30, 790], ones, 1.0)
        assert torch.allclose(torch.tensor(out.shape), torch.tensor([30, 30, 790]))


    def test_make_causal_mask_3d(self):
        out = make_causal_attn_mask_3d(torch.ones((20, 20, 20)))
        assert torch.allclose(out, torch.tril(torch.ones((20, 20, 20), dtype=torch.float)))

    def test_masks(self):
        max_seq_length = 10
        batch_size = 5
        n_heads = 2

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, 32))

        # Test the create_key_padding_mask function
        key_padding_mask = create_key_padding_mask(max_seq_length, batch_size, decoder_states)
        assert key_padding_mask.shape == (batch_size, max_seq_length)

        max_seq_length = 10
        batch_size = 5
        n_heads = 2
        hidden_dim = 32

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, hidden_dim))

        # Test the create_key_padding_mask function without passing a key_padding_mask
        key_padding_mask = create_key_padding_mask(max_seq_length, batch_size, decoder_states)
        assert key_padding_mask.shape == (batch_size, max_seq_length)

        # Test the create_key_padding_mask function with predefined key_padding_mask
        predefined_key_padding_mask = torch.ones((batch_size, max_seq_length - 2), dtype=torch.bool)
        key_padding_mask = create_key_padding_mask(max_seq_length, batch_size, decoder_states,
                                                   predefined_key_padding_mask)
        assert key_padding_mask.shape == (batch_size, max_seq_length)

    def test_attn_mask_2d(self):
        max_seq_length = 10
        batch_size = 5
        n_heads = 2

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, 32))

        # Test the create_attn_mask_2d function
        attn_mask_2d = create_attn_mask_2d(decoder_states, max_seq_length)
        assert attn_mask_2d.shape == (max_seq_length, max_seq_length)

        max_seq_length = 10
        batch_size = 5
        n_heads = 2
        hidden_dim = 32

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, hidden_dim))

        # Test the create_attn_mask_2d function without passing an attn_mask
        attn_mask_2d = create_attn_mask_2d(decoder_states, max_seq_length)
        assert attn_mask_2d.shape == (max_seq_length, max_seq_length)

        # Test the create_attn_mask_2d function with predefined attn_mask
        predefined_attn_mask_2d = torch.ones((max_seq_length - 2, max_seq_length - 2), dtype=torch.bool)
        attn_mask_2d = create_attn_mask_2d(decoder_states, max_seq_length, predefined_attn_mask_2d)
        assert attn_mask_2d.shape == (max_seq_length, max_seq_length)

    def test_attn_mask_3d(self):
        max_seq_length = 10
        batch_size = 5
        n_heads = 2
        hidden_dim = 32

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, hidden_dim))

        # Test the create_attn_mask_3d function without passing an attn_mask
        attn_mask_3d = create_attn_mask_3d(max_seq_length, batch_size, n_heads, decoder_states.shape[0])
        assert attn_mask_3d.shape == (batch_size * n_heads, max_seq_length, max_seq_length)

        # Test the create_attn_mask_3d function with predefined attn_mask
        predefined_attn_mask_3d = torch.ones((batch_size * n_heads, max_seq_length - 2, max_seq_length - 2),
                                             dtype=torch.bool)
        attn_mask_3d = create_attn_mask_3d(max_seq_length, batch_size, n_heads, decoder_states.shape[0], predefined_attn_mask_3d)
        assert attn_mask_3d.shape == (batch_size * n_heads, max_seq_length, max_seq_length)

        max_seq_length = 10
        batch_size = 5
        n_heads = 2

        # Create a dummy tensor for decoder_states
        decoder_states = torch.randn((max_seq_length, batch_size, 32))

        # Test the create_attn_mask_3d function
        attn_mask_3d = create_attn_mask_3d(max_seq_length, batch_size, n_heads, decoder_states.shape[0])
        assert attn_mask_3d.shape == (batch_size * n_heads, max_seq_length, max_seq_length)

    # def test_translate_hf(self):
    #     diag = torch.triu(torch.ones(20, 30, dtype=torch.float)).bool()
    #     out = translate_huggingface_batched_attn_mask_3d(diag)
    #     attn_mask_created = create_attn_mask_3d(30, 20, 4, torch.rand((30, 20)), out.shape[0])
    #     print(attn_mask_created)

    # def test_split_sequences_extra_pad_seq_state(self):
    #     values = torch.rand(124, 30, 768)
    #     out_values = split_sequence_get_attn_mask(values, 20, 4, torch.ones((120, 124, 124)))
    #     found = split_sequence(values, 20)
    #     out_value = out_values[len(found) - 1]
    #     assert torch.allclose(torch.tensor(out_value[0].shape), torch.tensor([20, 30, 768]))
    #     assert torch.allclose(torch.tensor(out_value[1].shape), torch.tensor((120, 20, 20)))

    def test_pad_collapse_states_2d(self):
        to_pad = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        size_sequence_state = 2
        aggregation = MeanAgg()

        padded = pad_collapse_states_2d(size_sequence_state, to_pad, aggregation)

        assert padded.shape == (size_sequence_state, to_pad.shape[1])
        print(padded)
        assert_allclose(padded, torch.tensor([[2., 3.], [5., 6.]]))

    def test_pad_collapse_states_3d(self):
        to_pad = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]])
        size_sequence_state = 2
        aggregation = MeanAgg()

        padded = pad_collapse_states_3d(size_sequence_state, to_pad, aggregation)

        assert padded.shape == (size_sequence_state, to_pad.shape[1], to_pad.shape[2])
        print(padded)
        assert_allclose(padded, torch.tensor([[[9., 10.],
                                               [11., 12.]],
                                              [[5., 6.],
                                               [7., 8.]]]))

    def test_pad_collapse_states_2d_size(self):
        to_pad = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        aggregation = MeanAgg()

        padded = pad_collapse_states_2d(2, to_pad, aggregation)

        assert padded.shape == (2, to_pad.shape[1])

        padded = pad_collapse_states_2d(4, to_pad, aggregation)

        assert padded.shape == (4, to_pad.shape[1])

    def test_pad_collapse_states_3d_size(self):
        to_pad = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]])
        aggregation = MeanAgg()

        padded = pad_collapse_states_3d(2, to_pad, aggregation)

        assert padded.shape == (2, to_pad.shape[1], to_pad.shape[2])
        padded = pad_collapse_states_3d(4, to_pad, aggregation)

        assert padded.shape == (4, to_pad.shape[1], to_pad.shape[2])


    def test_create_2d_causal_mask(self):
        seq_len = 5
        attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        expected_mask = torch.tensor([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True]
        ])

        mask = make_causal_attn_mask_2d(attn_mask)
        print(mask)
        self.assertTrue(torch.all(mask.eq(expected_mask)))

    def test_causal_mask_3d(self):
        b, seq_len = 2, 3
        attn_mask = torch.ones(b, seq_len, seq_len).bool()
        expected_mask = torch.tensor([
            [
                [True, False, False],
                [True, True, False],
                [True, True, True]
            ],
            [
                [True, False, False],
                [True, True, False],
                [True, True, True]
            ]
        ])

        mask = make_causal_attn_mask_3d(attn_mask)
        print(mask)
        self.assertTrue(torch.all(mask.eq(expected_mask)))