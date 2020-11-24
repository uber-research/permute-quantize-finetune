# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Dict

import torch

# fmt: off
try:
    import horovod.torch as hvd
    HAVE_HOROVOD = True
except ImportError:
    HAVE_HOROVOD = False
# fmt: on


def _compress_state_dict(original_state_dict: Dict, float16_codebooks: bool = True) -> Dict:
    """Given an uncompressed state dict, compresses it by
    1. Expressing BatchNorms with just two vectors, and
    2. Converting codebooks to float16
    These two transformations result in smaller models, and are necessary to match the
    compression ratios of the Bit Goes Down paper.

    Parameters:
        original_state_dict: Uncompressed state dict for the model
        float16_codebooks: Whether we should use 16 or 32 bit floats for the codebooks
    Returns:
        compressed_dict: Dict with the same structure as `original_state_dict`, but compressed
    """
    compressed_dict = {}

    batchnorm_layers = [k.strip(".running_mean") for k in original_state_dict.keys() if "running_mean" in k]

    for key, value in original_state_dict.items():
        if "running_var" in key:
            continue

        if "running_mean" in key:
            # Compress batchnorm to two vectors instead of four
            weight_param_name = key.replace("running_mean", "weight")
            bias_param_name = key.replace("running_mean", "bias")
            running_var_param_name = key.replace("running_mean", "running_var")

            original_weight = original_state_dict[weight_param_name]
            original_bias = original_state_dict[bias_param_name]
            running_mean = value
            running_var = original_state_dict[running_var_param_name]

            EPSILON = 1e-8  # Needed for numerical stability

            adjusted_weight = original_weight / (torch.sqrt(running_var) + EPSILON)
            adjusted_bias = original_bias - (original_weight * running_mean / (torch.sqrt(running_var) + EPSILON))

            compressed_dict[weight_param_name] = adjusted_weight
            compressed_dict[bias_param_name] = adjusted_bias

        elif "codebook" in key and float16_codebooks:
            compressed_dict[key] = value.half()

        else:
            # Copy other values verbatim, unless they are a batchnorm layer
            if not any(key.startswith(bnorm_layer) for bnorm_layer in batchnorm_layers):
                compressed_dict[key] = value

    return compressed_dict


def save_state_dict_compressed(model: torch.nn.Module, file_path: str, float16_codebooks: bool = True) -> None:
    """Saves a compressed dict for the model at the given file path

    Parameters:
        model: Model whose weights we wish to save
        file_path: Destination file for the weights
        float16_codebooks: Whether we should use 16 or 32 bit floats while saving
            codebooks
    """
    if (HAVE_HOROVOD and hvd.rank() == 0) or (not HAVE_HOROVOD):
        compressed_dict = _compress_state_dict(model.state_dict(), float16_codebooks)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(compressed_dict, file_path)


def save_state_dict(model: torch.nn.Module, file_path: str) -> None:
    """Saves an uncompressed state dict for the given model at the file path

    Parameters:
        model: Model whose state we wish to save
        file_path: Destination file for the state
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), file_path)


def _load_compressed_dict(model: torch.nn.Module, state_dict: Dict) -> None:
    """Loads a compressed state dict using the batchnorm trick. This assumes that the running mean and variance are not
    provided in the state dictionary, but instead encoded in the BatchNorm's weight and bias. Therefore, the running
    mean and variance are set to zero and one respectively.

    Parameters:
        model: Network for which we are loading the compressed dict
        state_dict: State dictionary for the network
    """
    all_keys = set(model.state_dict().keys())
    given_keys = set(state_dict.keys())

    # Verify that the only missing keys are the batchnorm keys
    BNORM_ONLY_KEYS = ["running_mean", "running_var", "num_batches_tracked"]

    assert given_keys.issubset(all_keys)
    assert all(any(batchnorm_key in k for batchnorm_key in BNORM_ONLY_KEYS) for k in (all_keys - given_keys))

    # We need strict=False for the missing batchnorm mean and variance
    model.load_state_dict(state_dict, strict=False)

    def is_batchnorm_layer(mod: torch.nn.Module) -> bool:
        return hasattr(module, "running_mean")

    # Manually set the batchnorm mean and variance to one and zero
    for module in model.modules():
        if is_batchnorm_layer(module):
            module.running_mean.fill_(0)
            module.running_var.fill_(1)


def load_state_dict(model: torch.nn.Module, file_path: str) -> torch.nn.Module:
    """Load a state dict from a given file path into the model.
    If the state dict is compressed, the batchnorm trick will be applied while loading. Otherwise, normal load

    Parameters:
        model: Network whose state dict we should load
        file_path: Path to the file with the saved weights
    Return:
        model: The model with the weights loaded into it
    """
    state_dict = torch.load(file_path)
    is_compressed = all("running_mean" not in k for k in state_dict.keys())
    if is_compressed:
        logging.info("Loading compressed dict from: {}".format(file_path))
        _load_compressed_dict(model, state_dict)
    else:
        # normal state dict
        logging.info("Loading uncompressed dict from: {}".format(file_path))
        model.load_state_dict(state_dict)

    return model
