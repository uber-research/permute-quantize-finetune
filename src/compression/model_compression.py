# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, List, NewType, Optional, Set, Union

import torch

from ..compressed_layers.CompressedConv2d import CompressedConv2d
from ..compressed_layers.CompressedConvTranspose2d import CompressedConvTranspose2d
from ..compressed_layers.CompressedLinear import CompressedLinear
from ..compression.coding import get_kmeans_fn

RecursiveReplaceFn = NewType("RecursieReplaceFn", Callable[[torch.nn.Module, torch.nn.Module, int, str, str], bool])


def _replace_child(model: torch.nn.Module, child_name: str, compressed_child_model: torch.nn.Module, idx: int) -> None:
    """Replaces a given module into `model` with another module `compressed_child_model`

    Parameters:
        model: Model where we are replacing elements
        child_name: The key of `compressed_child_model` in the parent `model`. Used if `model` is a torch.nn.ModuleDict
        compressed_child_model: Child module to replace into `model`
        idx: The index of `compressed_child_model` in the parent `model` Used if `model` is a torch.nn.Sequential
    """
    if isinstance(model, torch.nn.Sequential):
        # Add back in the correct position
        model[idx] = compressed_child_model
    elif isinstance(model, torch.nn.ModuleDict):
        model[child_name] = compressed_child_model
    else:
        model.add_module(child_name, compressed_child_model)


def prefix_name_lambda(prefix: str) -> Callable[[str], str]:
    """Returns a function that preprends `prefix.` to its arguments.

    Parameters:
        prefix: The prefix that the return function will prepend to its inputs
    Returns:
        A function that takes as input a string and prepends `prefix.` to it
    """
    return lambda name: (prefix + "." + name) if prefix else name


@torch.no_grad()
def apply_recursively_to_model(fn: RecursiveReplaceFn, model: torch.nn.Module, prefix: str = "") -> None:
    """Recursively apply fn on all modules in models

    Parameters:
        fn: The callback function, it is given the parents, the children, the index of the children,
            the name of the children, and the prefixed name of the children
            It must return a boolean to determine whether we should stop recursing the branch
        model: The model we want to recursively apply fn to
        prefix: String to build the full name of the model's children (eg `layer1` in `layer1.conv1`)
    """
    get_prefixed_name = prefix_name_lambda(prefix)

    for idx, named_child in enumerate(model.named_children()):

        child_name, child = named_child
        child_prefixed_name = get_prefixed_name(child_name)

        if fn(model, child, idx, child_name, child_prefixed_name):
            continue
        else:
            apply_recursively_to_model(fn, child, child_prefixed_name)


def compress_model(
    model: torch.nn.Module,
    ignored_modules: Union[List[str], Set[str]],
    k: int,
    k_means_n_iters: int,
    k_means_type: str,
    fc_subvector_size: int,
    pw_subvector_size: int,
    large_subvectors: bool,
    layer_specs: Optional[Dict] = None,
) -> torch.nn.Module:
    """
    Given a neural network, modify it to its compressed representation with hard codes
      - Linear is replaced with compressed_layers.CompressedLinear
      - Conv2d is replaced with compressed_layers.CompressedConv2d
      - ConvTranspose2d is replaced with compressed_layers.CompressedConvTranspose2d

    Parameters:
        model: Network to compress. This will be modified in-place
        ignored_modules: List or set of submodules that should not be compressed
        k: Number of centroids to use for each compressed codebook
        k_means_n_iters: Number of iterations of k means to run on each compressed module
            during initialization
        k_means_type: k means type (kmeans, src)
        fc_subvector_size: Subvector size to use for linear layers
        pw_subvector_size: Subvector size for point-wise convolutions
        large_subvectors: Kernel size of K^2 of 2K^2 for conv layers
        layer_specs: Dict with different configurations for individual layers
    Returns:
        The passed model, which is now compressed
    """
    if layer_specs is None:
        layer_specs = {}

    def _compress_and_replace_layer(
        parent: torch.nn.Module, child: torch.nn.Module, idx: int, name: str, prefixed_child_name: str
    ) -> bool:
        """Compresses the `child` layer and replaces the uncompressed version into `parent`"""

        assert isinstance(parent, torch.nn.Module)
        assert isinstance(child, torch.nn.Module)

        if prefixed_child_name in ignored_modules:
            return True

        child_layer_specs = layer_specs.get(prefixed_child_name, {})

        _k = child_layer_specs.get("k", k)
        _kmeans_n_iters = child_layer_specs.get("kmeans_n_iters", k_means_n_iters)
        _kmeans_fn = get_kmeans_fn(child_layer_specs.get("kmeans_type", k_means_type))
        _fc_subvector_size = child_layer_specs.get("subvector_size", fc_subvector_size)
        _large_subvectors = child_layer_specs.get("large_subvectors", large_subvectors)
        _pw_subvector_size = child_layer_specs.get("subvector_size", pw_subvector_size)

        if isinstance(child, torch.nn.Conv2d):
            compressed_child = CompressedConv2d.from_uncompressed(
                child, _k, _kmeans_n_iters, _kmeans_fn, _large_subvectors, _pw_subvector_size, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        elif isinstance(child, torch.nn.ConvTranspose2d):
            compressed_child = CompressedConvTranspose2d.from_uncompressed(
                child, _k, _kmeans_n_iters, _kmeans_fn, _large_subvectors, _pw_subvector_size, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        elif isinstance(child, torch.nn.Linear):
            compressed_child = CompressedLinear.from_uncompressed(
                child, _k, _kmeans_n_iters, _kmeans_fn, _fc_subvector_size, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        else:
            return False

    apply_recursively_to_model(_compress_and_replace_layer, model)
    return model
