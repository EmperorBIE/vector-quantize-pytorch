'''
TODO:
    1. 原代码中的 ResidualVQ 的 init 没有选择是用euler还是cos方式的参数 use_cosine_sim, SelfResidualVQ 中是否要加上?
    2. 返回的 quantized 的维度是否要转换 n 1 c w -> n c w
'''

import random
from math import ceil
from functools import partial
from itertools import zip_longest

import torch
from torch import nn
import torch.nn.functional as F
# from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

import pdb

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class SelfResidualVQ(nn.Module):
    """ Quantizes the residuals using the same codebook iteratively """
    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim = None,
        shared_codebook = False,
        heads = 1,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        assert heads == 1, 'self residual vq is not compatible with multi-headed codes'
        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layer = VectorQuantize(dim=codebook_dim, codebook_dim=codebook_dim, accept_image_fmap=accept_image_fmap, **kwargs)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    def forward(
        self,
        x,
        indices=None,
        return_all_codes=False,
        sample_codebook_temp=None,
        freeze_codebook=False,
        mask=None,
    ):
        residual = self.project_in(x)
        all_indices = []
        commit_losses = []
        quantized = []

        for _ in range(self.num_quantizers):
            quantized_output, indices, commit_loss = self.layer(
                residual,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=freeze_codebook,
                mask=mask
            )
            residual = residual - quantized_output

            all_indices.append(indices)
            commit_losses.append(commit_loss)
            quantized.append(quantized_output)

        # pdb.set_trace()

        quantized = torch.stack(quantized)
        commit_losses = torch.stack(commit_losses)
        all_indices = torch.stack(all_indices)

        # pdb.set_trace()

        quantized = self.project_out(quantized)
        return quantized, all_indices, commit_losses

if __name__ == '__main__':
    self_residual_vq = SelfResidualVQ(
        dim = 256,
        num_quantizers = 8,      # specify number of quantizers
        codebook_size = 1024,    # codebook size
    )

    x = torch.randn(1, 1024, 256)

    # pdb.set_trace()
    quantized, indices, commit_loss = self_residual_vq(x)

    print(quantized.shape, indices.shape, commit_loss.shape)