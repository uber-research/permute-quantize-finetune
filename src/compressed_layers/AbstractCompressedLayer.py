# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractCompressedLayer(ABC, nn.Module):
    """Abstract superclass for quantized layers"""

    def __init__(self):
        super(AbstractCompressedLayer, self).__init__()

    def initialize_codes(self, codes_matrix: nn.Module, codebook: nn.Module) -> None:
        """Save the codes using the smalles amount of memory possible"""
        if codebook.size(0) <= (1 << 8):
            self.codes_matrix = nn.Parameter(codes_matrix.byte(), requires_grad=False)
        elif codebook.size(0) <= (1 << 16):
            self.codes_matrix = nn.Parameter(codes_matrix.short(), requires_grad=False)
        else:
            self.codes_matrix = nn.Parameter(codes_matrix.int(), requires_grad=False)

    @staticmethod
    def log_quantization_error(
        name: str, k_means_n_iters: int, error: float, codebook: nn.Module, codes_matrix: nn.Module
    ) -> None:
        """Log the quantization error of the codes matrix"""
        logging.info(
            "{} compression: {}; mse: {:2e}; codebook size: {}; code size: {}".format(
                name,
                k_means_n_iters,
                error,
                " x ".join(map(str, codebook.size())),
                " x ".join(map(str, codes_matrix.size())),
            )
        )

    @abstractmethod
    def _get_uncompressed_weight(self):
        """Uses codes and codebooks to reconstruct an uncompressed weights matrix"""
        pass

    @abstractmethod
    def forward(self):
        """Does a forward pass as a torch.nn.Module"""
        pass

    @staticmethod
    @abstractmethod
    def from_uncompressed():
        """Creates a compressed layer starting from an uncompressed one"""
        pass
