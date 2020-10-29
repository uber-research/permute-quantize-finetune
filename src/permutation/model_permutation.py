# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for permuting deep networks without changing their mappings, such that they are easier to vector-compress"""

import logging
import random
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from ..compression.model_compression import apply_recursively_to_model
from .optimization import (
    get_cov_det,
    get_random_permutation,
    optimize_permutation_by_greedy_search,
    optimize_permutation_by_stochastic_local_search,
)

YAMLParserResult = NewType("YAMLParserResult", Iterable[Iterable[Dict[str, List]]])


class PermutationOptimizationMethod(Enum):
    RANDOM = 1
    GREEDY = 2
    STOCHASTIC = 3


def _get_subvector_size(layer: torch.nn.Module, layer_specs: Dict[str, Any]) -> int:
    """Return the appropriate subvector size for a layer"""

    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
        _, c_in, kernel_width, kernel_height = layer.weight.size()
        subvector_size = kernel_width * kernel_height
        if subvector_size == 1:
            subvector_size = layer_specs["pw_subvector_size"]
        elif layer_specs["large_subvectors"]:
            subvector_size *= 2

        assert (c_in * kernel_width * kernel_height) % subvector_size == 0
        return subvector_size

    elif isinstance(layer, torch.nn.Linear):
        return layer_specs["fc_subvector_size"]

    else:
        raise ValueError(f"Got unsupported layer type `{type(layer)}`")


def _collect_layers(model: nn.Module) -> Dict:
    """Gets a dictionary with all the layers of a network. We prefer this over `model.named_parameters()`, as here
    we can also get batchnorm layers

    Parameters:
        model: A network whose layers we want to get
    Returns:
        layers: A dictionary where the keys are the prefixed names of each layer, and the value is said layer
    """
    layers = {}

    def _collect_layer(parent: nn.Module, layer: nn.Module, idx: int, name: str, prefixed_name: str) -> bool:
        """Adds layers to a dictionary. Returns True if recursion should stop, and False if it should continue"""
        if (
            isinstance(layer, nn.Conv2d)
            or isinstance(layer, nn.ConvTranspose2d)
            or isinstance(layer, nn.BatchNorm2d)
            or isinstance(layer, FrozenBatchNorm2d)
            or isinstance(layer, nn.Linear)
        ):
            layers[prefixed_name] = layer
            return True
        return False

    apply_recursively_to_model(_collect_layer, model)
    return layers


def _is_optimizable(layer: nn.Module, subvector_size: int) -> bool:
    """Decide whether a permutation can be optimized on a given layer and vector size"""

    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        _, _, height, width = layer.weight.shape
        is_pointwise = (height == 1) and (width == 1)

        if not is_pointwise and subvector_size == height * width:  # 3x3 conv with single block cannot be optimized
            return False

    return True


def get_permutation(
    name: str,
    weight: torch.Tensor,
    subvector_size: int,
    optimization_methods: List[PermutationOptimizationMethod],
    sls_iterations: int,
) -> List[int]:
    """Obtains permutation that minimizes the determinant of the covariance of `weight`."""

    if weight.ndim == 2:
        # This is a linear layer. We can add two dummy dimensions and treat is as a 1x1 conv for permutation purposes
        weight = weight[:, :, None, None]

    c_out, c_in, h, w = weight.shape

    if h != w:
        raise ValueError("Currently, convolutions with different height and width are not supported.")

    # if h not in [1, 2, 3]:
    #     raise ValueError("Currently, only convolutions of size 1x1, 2x2 or 3x3 are supported.")

    d = c_in * h * w
    if (d % subvector_size) != 0:
        raise ValueError(
            f"Weight with shape {weight.shape} cannot be evenly divided into subvectors of size {subvector_size}"
        )

    assert len(optimization_methods) > 0

    permutation = None

    # If asked for random permutation, just return that
    for method in optimization_methods:

        if method == PermutationOptimizationMethod.RANDOM:
            permutation = get_random_permutation(weight)

        elif method == PermutationOptimizationMethod.GREEDY:
            permutation = optimize_permutation_by_greedy_search(weight, subvector_size)

        elif method == PermutationOptimizationMethod.STOCHASTIC:
            permutation = optimize_permutation_by_stochastic_local_search(
                name, weight, subvector_size, sls_iterations, permutation
            )

        else:
            raise ValueError(f"Permutation method {method} unknown.")

    assert len(permutation) == c_in, "{} != {}".format(len(permutation), c_in)
    return permutation


def _permute_group(
    parent_dict: Dict[str, nn.Module],
    children_dict: Dict[str, Tuple[nn.Module, int]],
    sls_iterations: Optional[int] = 10_000,
    do_random_permutation: Optional[bool] = False,
) -> None:
    """Permute a group of parent and children layers, such that the children are easier to vector-compress

    Parameters:
        parent_dict: The key is the parent name, and the value is the parent layer
        children_dict: The key is the child name, and the value is a tuple with the child layer and its subvector size
        sls_iterations: Number of iterations for stochastic local search
        do_random_permutation: Whether to permute with random permutations, instead of optimized ones (used for testing)
    """

    logging.info(f"Optimizing permutation for {children_dict.keys()} with {len(parent_dict)} parents")

    # Find the first child in the list that is optimizable
    for child_name, (child, sub_size, child_permutation_specs) in children_dict.items():
        if _is_optimizable(child, sub_size):
            child_weight = child.weight.detach()
            prev_cov_det = get_cov_det(child_weight.reshape(-1, sub_size))
            logging.info(f"Optimizing permutation for {child_name}")

            if child_permutation_specs["reshape"] is not None:
                child_weight = child_weight.reshape(child_permutation_specs["reshape"])

            break
    else:
        # As a reminder, the else clause of a for loop in python is only reached if the for loop finished and it never
        # breaks. In this case, this means that none of the children layers are optimizable (eg, because they are all
        # 3x3 convolutions, and the block size is 9).
        if do_random_permutation:
            # This will return a permutation for a random child
            random_child = random.choice(list(children_dict))

            child_name = random_child
            child, sub_size, _ = children_dict[child_name]
            child_weight = child.weight.detach()
            prev_cov_det = get_cov_det(child_weight.reshape(-1, sub_size))
        else:
            logging.info("None of the layers are optimizable. Skipping.")
            return

    permutation = get_permutation(
        child_name, child_weight, sub_size, child_permutation_specs["optimization"], sls_iterations
    )

    new_cov_det = get_cov_det(child_weight[:, permutation].detach().reshape(-1, sub_size))

    if not do_random_permutation and new_cov_det > prev_cov_det:
        # Make sure the permutation did not make things worse. This is rare but may happen if eg there are no SLS
        # iterations, since the greedy method can result in somewhat higher determinant of the covariance.
        logging.warning(f"new covdet is higher than previous one: {new_cov_det} > {prev_cov_det}. Skipping")
        return

    logging.info(f"{child_name}: prev covdet {prev_cov_det:2e}, new covdet: {new_cov_det:2e}")

    # Apply permutation to all children
    for child_name, (child, sub_size, child_permutation_specs) in children_dict.items():

        if child_permutation_specs["reshape"] is not None:
            reshape_dim = child_permutation_specs["reshape"]
            reshaped_weight = child.weight.reshape(reshape_dim)[:, permutation]
            child.weight = torch.nn.Parameter(reshaped_weight.reshape([reshape_dim[0], -1]))

        elif (
            isinstance(child, nn.Conv2d) or isinstance(child, torch.nn.ConvTranspose2d) or isinstance(child, nn.Linear)
        ):
            assert len(permutation) == child.weight.shape[1], "{} != {}".format(len(permutation), child.weight.shape[1])
            child.weight = torch.nn.Parameter(child.weight[:, permutation])

        else:
            raise ValueError(f"Child layer permutation not supported: {type(child)}")

    # Apply the same permutation to parents
    for parent_name, parent in parent_dict.items():

        assert len(permutation) == len(parent.weight), "{} != {}".format(len(permutation), len(parent.weight))

        if (
            isinstance(parent, torch.nn.Conv2d)
            or isinstance(parent, torch.nn.ConvTranspose2d)
            or isinstance(parent, nn.Linear)
        ):
            parent.weight = torch.nn.Parameter(parent.weight[permutation])
            if parent.bias is not None:
                parent.bias = torch.nn.Parameter(parent.bias[permutation])

        elif isinstance(parent, torch.nn.BatchNorm2d) or isinstance(parent, FrozenBatchNorm2d):
            parent.weight = torch.nn.Parameter(parent.weight[permutation])
            parent.bias = torch.nn.Parameter(parent.bias[permutation])
            parent.running_mean = parent.running_mean[permutation]
            parent.running_var = parent.running_var[permutation]

        else:
            raise ValueError(f"Parent layer not supported: {type(parent)}")


def permute_model(
    model: torch.nn.Module,
    fc_subvector_size: int,
    pw_subvector_size: int,
    large_subvectors: bool,
    permutation_groups: YAMLParserResult = None,
    layer_specs: Optional[Dict] = None,
    permutation_specs: Optional[Dict] = None,
    sls_iterations: Optional[int] = 10_000,
    do_random_permutation: Optional[bool] = False,
) -> None:
    """Find the permutations of a model, such that the model is easier to vector-compress. The model is permuted in
    place, and this should not affect its outputs as long as the `permutation_groups` describe parent-child
    relationships in the network

    Parameters:
        model: The model we want to permute
        fc_subvector_size: Subvector size for fully connected layers
        pw_subvector_size: Subvect size for pointwise convolutions
        large_subvectors: Whether to use larger codeword sizes (and thus, higher compression)
        permutation_groups: Groups of parent-child layers that must share the same permutation
        layer_specs: Compression specs for specific layers that override default values
        permutation_specs: Permutation specs to override how we treat the permutation of specific layers
        sls_iterations: Number of iterations for stochastic local search
        do_random_permutation: Whether to return a random permutation instead of an optimized one. Used to test that
                               the permutation groups do not change the output of a network
    """

    if layer_specs is None:
        layer_specs = {}

    if permutation_specs is None:
        permutation_specs = {}

    name_to_layer_dict = _collect_layers(model)

    def _layer_specs(prefixed_name: str) -> Dict[str, Any]:
        child_layer_specs = layer_specs.get(prefixed_name, {})
        return {
            "pw_subvector_size": child_layer_specs.get("subvector_size", pw_subvector_size),
            "fc_subvector_size": child_layer_specs.get("subvector_size", fc_subvector_size),
            "large_subvectors": child_layer_specs.get("large_subvectors", large_subvectors),
        }

    def _permutation_specs(prefixed_name: str) -> Dict[str, Any]:
        """Returns a dictionary that specifies how permutations should be handled for a specific layer.

        The fields include "reshape", indicating that a layer should be permuted following a certain shape,
        and "optimization", a list of methods used to optimize the permutation.
        """
        child_permutation_specs = permutation_specs.get(prefixed_name, {})

        # By default, we do greedy optimization followed by stochastic local search
        optimization_methods = [PermutationOptimizationMethod.GREEDY, PermutationOptimizationMethod.STOCHASTIC]

        overwritten_child_permutation_specs = {}
        overwritten_child_permutation_specs["reshape"] = child_permutation_specs.get("reshape", None)

        if child_permutation_specs.get("optimization", None) is None:
            overwritten_child_permutation_specs["optimization"] = optimization_methods
        else:
            overwritten_child_permutation_specs["optimization"] = [
                PermutationOptimizationMethod[x] for x in child_permutation_specs.get("optimization", None)
            ]

        # do_random_permutation means that we randomly permute _all_ the layers
        if do_random_permutation:
            overwritten_child_permutation_specs["optimization"] = [PermutationOptimizationMethod.RANDOM]

        return overwritten_child_permutation_specs

    # Optimize a permutation for each group
    for group in permutation_groups:

        assert len(group) == 2  # A group is a list of two dictionaries: parents and children
        parent_names, children_names = group[0]["parents"], group[1]["children"]

        # Check that all the passsed layers are actually in the network
        for parent in parent_names:
            assert parent in name_to_layer_dict
        for child in children_names:
            assert child in name_to_layer_dict

        parent_dict = OrderedDict({pn: name_to_layer_dict[pn] for pn in parent_names})

        children_dict = OrderedDict(
            {
                child_name: tuple(
                    (
                        name_to_layer_dict[child_name],
                        _get_subvector_size(name_to_layer_dict[child_name], _layer_specs(child_name)),
                        _permutation_specs(child_name),
                    )
                )
                for child_name in children_names
            }
        )

        _permute_group(parent_dict, children_dict, sls_iterations, do_random_permutation=do_random_permutation)
