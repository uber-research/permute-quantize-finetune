# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Union

import torch

Tensors = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor]]


def to_cuda(inputs: Tensors) -> Tensors:
    """Recursively moves inputs to the GPU

    Parameters:
        inputs: Tensors to move to the GPU
    Returns:
        output: Tensors moved to the GPU
    """

    def _to_cuda(inputs):
        if type(inputs) == torch.Tensor:
            return inputs.cuda(non_blocking=True)
        elif type(inputs) == list:
            return [_to_cuda(inp) for inp in inputs]
        elif type(inputs) == dict:
            return {k: _to_cuda(v) for k, v in inputs.items()}

    output = _to_cuda(inputs)
    torch.cuda.synchronize()
    return output


def to_cpu(inputs: Tensors) -> Tensors:
    """Recursively moves inputs to the CPU

    Parameters:
        inputs: Tensors to move to the CPU
    Returns:
        output: Tensors moved to the CPU
    """
    if type(inputs) == torch.Tensor:
        return inputs.cpu()
    elif type(inputs) == list:
        return [to_cpu(inp) for inp in inputs]
    elif type(inputs) == dict:
        return {k: to_cpu(v) for k, v in inputs.items()}
