# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..compression.coding import decode
from .AbstractCompressedLayer import AbstractCompressedLayer
from .CompressedConv2d import CompressedConv2d


class CompressedConvTranspose2d(AbstractCompressedLayer):
    """Compressed representation of a ConvTranspose2d layer"""

    def __init__(
        self,
        codes_matrix: torch.Tensor,
        codebook: torch.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_height: int,
        kernel_width: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: Optional[torch.Tensor] = None,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ):
        super(CompressedConvTranspose2d, self).__init__()

        self.initialize_codes(codes_matrix, codebook)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self.codebook = nn.Parameter(codebook)

    def _get_uncompressed_weight(self):
        decoded_weights = decode(self.codes_matrix, self.codebook).float()
        c_out = decoded_weights.size(0)
        return decoded_weights.reshape(c_out, -1, self.kernel_height, self.kernel_width)

    def get_default_output_size(self, input: torch.Tensor, dimension: str):
        """Gets the default height/width of the output tensor for this layer"""
        if dimension == "height":
            shape_idx, idx, kernel_size = 2, 0, self.kernel_height
        elif dimension == "width":
            shape_idx, idx, kernel_size = 3, 1, self.kernel_width
        else:
            raise ValueError(f"dimension must be either height or width, but was {dimension}")
        return (
            (input.shape[shape_idx] - 1) * self.stride[idx]
            - 2 * self.padding[idx]
            + self.dilation[idx] * (kernel_size - 1)
            + self.output_padding[idx]
            + 1
        )

    def forward(self, input, output_size=None):
        if len(input) == 0:
            # NOTE We sometimes hit this case, eg, right after quantizing the full network, when the network produces no
            # detection candidates. Since F.conv_transpose2d crashes if we pass it an empty input, so we are handling it
            # separately here.

            # Inferring the output size from the documentation: https://pytorch.org/docs/stable/nn.html#convtranspose2d
            h_out = self.get_default_output_size(input, "height")
            w_out = self.get_default_output_size(input, "width")
            return torch.zeros((0, self.out_channels, h_out, w_out), dtype=input.dtype, device=input.device)

        return F.conv_transpose2d(
            input=input,
            weight=self._get_uncompressed_weight(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )

    @staticmethod
    def from_uncompressed(
        uncompressed_layer: torch.nn.ConvTranspose2d,
        k: int,
        kmeans_n_iters: int,
        kmeans_fn: Callable,
        large_subvectors: bool,
        pw_subvector_size: int,
        name: str = "",
    ) -> "CompressedConvTranspose2d":
        """Given an uncompressed layer, initialize the compressed equivalent according to the specified parameters

        Parameters:
            uncompressed_layer: ConvTranspose2d layer to compress
            k: Size of the codebook
            kmeans_n_iters: Number of iterations of k means
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

        _, _, kernel_width, kernel_height = uncompressed_layer.weight.shape

        return CompressedConvTranspose2d(
            codes_matrix,
            codebook,
            in_channels=uncompressed_layer.in_channels,
            out_channels=uncompressed_layer.out_channels,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            stride=uncompressed_layer.stride,
            padding=uncompressed_layer.padding,
            output_padding=uncompressed_layer.output_padding,
            groups=uncompressed_layer.groups,
            bias=uncompressed_layer.bias,
            dilation=uncompressed_layer.dilation,
        )
