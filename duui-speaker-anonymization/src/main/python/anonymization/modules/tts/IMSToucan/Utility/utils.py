"""Padding and masking helpers used by the inference runtime."""

import torch


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """Return a boolean mask whose true entries are padding."""
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    batch_size = len(lengths)
    max_length = int(max(lengths)) if xs is None else xs.size(length_dim)

    seq_range = torch.arange(0, max_length, dtype=torch.int64, device=device)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_length)
    mask = seq_range >= seq_range.new(lengths).unsqueeze(-1)

    if xs is not None:
        assert xs.size(0) == batch_size, (xs.size(0), batch_size)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        index = tuple(
            slice(None) if i in (0, length_dim) else None
            for i in range(xs.dim())
        )
        mask = mask[index].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """Return a boolean mask whose true entries are not padding."""
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


def pad_list(tensors, pad_value):
    """Pad tensors along their first dimension to a common length."""
    batch_size = len(tensors)
    max_length = max(tensor.size(0) for tensor in tensors)
    padded = tensors[0].new(
        batch_size, max_length, *tensors[0].size()[1:]
    ).fill_(pad_value)

    for index, tensor in enumerate(tensors):
        padded[index, : tensor.size(0)] = tensor
    return padded
