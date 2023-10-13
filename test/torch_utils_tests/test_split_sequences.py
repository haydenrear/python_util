from unittest import TestCase
import torch

from python_util.torch_utils.split_sequences import split_sequence, get_split_attn_masks, get_key_padding_mask, \
    split_sequence_get_attn_mask, split_sequence_get_key_padding_mask


class TestSplitSequence(TestCase):
    def test_split_sequence(self):
        seq = torch.arange(32).reshape(8, 4)
        split_seq = split_sequence(seq, 3)
        assert len(split_seq) == 3
        assert split_seq[0].shape == (3, 4)
        assert split_seq[1].shape == (3, 4)
        assert split_seq[2].shape == (3, 4)
        assert torch.all(split_seq[2][:2, :] == seq[6:, :])

    def test_get_split_attn_masks(self):
        seq = torch.arange(32).reshape(8, 4)
        attn_mask = torch.ones((8, 8))
        split_attn_mask = get_split_attn_masks(seq, 3, 2, attn_mask)
        assert len(split_attn_mask) == 3
        assert split_attn_mask[0].shape == (3, 3)
        assert split_attn_mask[1].shape == (3, 3)
        assert split_attn_mask[2].shape == (3, 3)
        assert torch.all(split_attn_mask[2][:2, :2] == attn_mask[6:, 6:])

    def test_get_key_padding_mask(self):
        seq = torch.arange(32).reshape(8, 4)
        key_padding_mask = torch.ones((1, 8))
        split_key_padding_mask = get_key_padding_mask(seq, 3, key_padding_mask=key_padding_mask)
        assert len(split_key_padding_mask) == 3
        assert split_key_padding_mask[0].shape == (1, 3)
        assert split_key_padding_mask[1].shape == (1, 3)
        assert split_key_padding_mask[2].shape == (1, 3)
        assert torch.all(split_key_padding_mask[2][:, :2] == key_padding_mask[:, 6:])

    def test_split_sequence_get_attn_mask(self):
        seq = torch.arange(32).reshape(8, 4)
        attn_mask = torch.ones((8, 8))
        splits = split_sequence_get_attn_mask(seq, 3, 2, attn_mask)
        assert len(splits) == 3
        assert splits[0][0].shape == (3, 4)
        assert splits[0][1].shape == (3, 3)
        assert torch.all(splits[2][0][:2, :] == seq[6:, :])
        assert torch.all(splits[2][1][:2, :2] == attn_mask[6:, 6:])

    def test_split_sequence_get_key_padding_mask(self):
        seq = torch.arange(32).reshape(8, 4)
        key_padding_mask = torch.ones((1, 8))
        splits = split_sequence_get_key_padding_mask(seq, 3, key_padding_mask=key_padding_mask)
        assert len(splits) == 3
        assert splits[0][0].shape == (3, 4)
        assert splits[0][1].shape == (1, 3)
        assert torch.all(splits[2][0][:2, :] == seq[6:, :])
        assert torch.all(splits[2][1][:, :2] == key_padding_mask[:, 6:])

    def test_split_sequence_get_attn_mask_2d(self):
        decoder_states = torch.randn((10, 10, 512))
        n_sequence_state = 5
        n_attn_heads = 8
        attn_mask = torch.randn((10, 10))

        result = split_sequence_get_attn_mask(decoder_states, n_sequence_state, n_attn_heads, attn_mask)

        assert len(result) == 2, f"Expected length 2, got {len(result)}"
        assert result[0][0].shape == (5, 10, 512), f"Expected shape (5, 32, 512), got {result[0][0].shape}"
        assert result[0][1] is not None, "Expected a tensor, got None"

    def test_split_sequence_get_attn_mask_3d(self):
        decoder_states = torch.randn((10, 10, 512))
        n_sequence_state = 5
        n_attn_heads = 8
        attn_mask = torch.randn((80, 10, 10))

        result = split_sequence_get_attn_mask(decoder_states, n_sequence_state, n_attn_heads, attn_mask)

        assert len(result) == 2, f"Expected length 2, got {len(result)}"
        assert result[0][0].shape == (5, 10, 512), f"Expected shape (5, 32, 512), got {result[0][0].shape}"
        assert result[0][1] is not None, "Expected a tensor, got None"

        decoder_states = torch.randn((10, 10, 512))
        n_sequence_state = 5
        n_attn_heads = 8
        attn_mask = torch.randn((80, 10, 10))

        result = split_sequence_get_attn_mask(decoder_states, n_sequence_state, n_attn_heads, attn_mask)

        assert len(result) == 2, f"Expected length 2, got {len(result)}"
        assert result[0][0].shape == (5, 10, 512), f"Expected shape (5, 32, 512), got {result[0][0].shape}"
        assert result[0][1] is not None, "Expected a tensor, got None"

    def test_split_sequence_get_key_padding_mask_2(self):
        decoder_states = torch.randn((10, 32, 512))
        n_sequence_state = 5
        key_padding_mask = torch.randn((32, 10))

        result = split_sequence_get_key_padding_mask(decoder_states, n_sequence_state, key_padding_mask)

        assert len(result) == 2, f"Expected length 2, got {len(result)}"
        assert result[0][0].shape == (5, 32, 512), f"Expected shape (5, 32, 512), got {result[0][0].shape}"
        assert result[0][1] is not None, "Expected a tensor, got None"
