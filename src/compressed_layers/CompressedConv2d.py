# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compression.coding import decode, get_num_centroids
from ..compression.kmeans import kmeans
from ..compression.kmeans_sr import src
from .AbstractCompressedLayer import AbstractCompressedLayer


class CompressedConv2d(AbstractCompressedLayer):
    """Compressed representation of a Conv2d layer"""

    def __init__(
        self,
        codes_matrix: torch.Tensor,
        codebook: torch.Tensor,
        kernel_height: int,
        kernel_width: int,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super(CompressedConv2d, self).__init__()

        self.initialize_codes(codes_matrix, codebook)

        self.kernel_height = kernel_height
        self.kernel_width = kernel_width

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        self.codebook = nn.Parameter(codebook)

    def _get_uncompressed_weight(self):
        decoded_weights = decode(self.codes_matrix, self.codebook).float()
        c_out = decoded_weights.size(0)
        return decoded_weights.reshape(c_out, -1, self.kernel_height, self.kernel_width)

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self._get_uncompressed_weight(),
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=self.groups,
        )

    @staticmethod
    def get_codes_and_codebook(
        uncompressed_layer: torch.nn.Conv2d,
        k: int,
        kmeans_n_iters: int,
        kmeans_fn: Callable,
        large_subvectors: bool,
        pw_subvector_size: int,
        name: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtain the codes and codebook to build a compressed conv layer. See `from_uncompressed` for doc details."""

        assert kmeans_fn in [kmeans, src]

        weight = uncompressed_layer.weight.detach()

        c_out, c_in, kernel_width, kernel_height = weight.size()
        reshaped_weight = weight.reshape(c_out, -1).detach()

        # For 1x1 convs, this is always 1
        subvector_size = kernel_height * kernel_width
        is_pointwise_convolution = subvector_size == 1

        # Determine subvector_size
        if is_pointwise_convolution:
            subvector_size = pw_subvector_size
        if large_subvectors and not is_pointwise_convolution:
            subvector_size *= 2

        assert (c_in * kernel_height * kernel_width) % subvector_size == 0

        num_blocks_per_row = (c_in * kernel_height * kernel_width) // subvector_size
        num_centroids = get_num_centroids(num_blocks_per_row, c_out, k)

        # Reshape and quantize
        training_set = reshaped_weight.reshape(-1, subvector_size)
        codebook, codes = kmeans_fn(training_set, k=num_centroids, n_iters=kmeans_n_iters)
        codes_matrix = codes.view(-1, num_blocks_per_row)

        # Log quantization error
        decoded_weights = decode(codes_matrix, codebook)
        error = (decoded_weights - reshaped_weight).pow(2).sum() / (num_blocks_per_row * c_out)
        AbstractCompressedLayer.log_quantization_error(name, kmeans_n_iters, error, codebook, codes_matrix)

        return codes_matrix, codebook

    @staticmethod
    def from_uncompressed(
        uncompressed_layer: torch.nn.Conv2d,
        k: int,
        kmeans_n_iters: int,
        kmeans_fn: Callable,
        large_subvectors: bool,
        pw_subvector_size: int,
        name: str = "",
    ) -> "CompressedConv2d":
        """Given an uncompressed layer, initialize the compressed equivalent according to the specified parameters.

        Parameters:
            uncompressed_layer: Conv2d layer to compress
            k: Size of the codebook
            k_means_n_iters: Number of iterations of k means
            kmeans_fn: k means type (kmeans or src)
            large_subvectors: Large or small block sizes for convolutions
            pw_subvector_size: Block size for point-wise convolutions
            name: Name of the layer to print alongside mean-squared error
        Returns:
            compressed_layer: Initialized compressed layer
        """

        codes_matrix, codebook = CompressedConv2d.get_codes_and_codebook(
            uncompressed_layer, k, kmeans_n_iters, kmeans_fn, large_subvectors, pw_subvector_size, name
        )
        c_out, c_in, kernel_width, kernel_height = uncompressed_layer.weight.size()

        return CompressedConv2d(
            codes_matrix,
            codebook,
            kernel_height,
            kernel_width,
            uncompressed_layer.bias,
            uncompressed_layer.stride,
            uncompressed_layer.padding,
            uncompressed_layer.dilation,
            uncompressed_layer.groups,
        )
