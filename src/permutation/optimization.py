# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations of different methods used to optimized a layer permutation"""

import copy
import itertools
import logging
import time
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm


def get_cov_det(x: torch.Tensor) -> float:
    """Compute the determinant of the covariance of `x`. This is the objective we minimize to find good permutations.

    Parameters:
        x: Matrix whose determinant of covariance we are computing
    Returns:
        The determinant of the covariance of `x`
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    cov = np.cov(x, rowvar=False)
    return np.linalg.det(cov)


def interleave_lists(lists: List[List]) -> List:
    """Given lists of equal length, interleave their values into a single big list

    Parameters:
        lists: A list of lists with values to interleave
    Returns:
        A list with the values in lists, but interleaved
    Example:
        >>> interleave_lists([1, 2, 3], [4, 5, 6])
        [1, 4, 2, 5, 3, 6]
    """
    lens = list(map(len, lists))
    assert all([lens[0] == x for x in lens]), "All the lists must have equal length, but lengths are {lens}"

    return list(itertools.chain(*zip(*lists)))


def get_random_permutation(weight: torch.Tensor):
    """No optimization, just return a random permutation. Useful for testing and debugging"""
    c_out, c_in, h, w = weight.shape
    return list(np.random.permutation(c_in))


def optimize_permutation_by_greedy_search(weight: torch.Tensor, subvector_size: int) -> List[int]:
    """Greedily computes a permutation that minimizes the determinant of the covariance of `weight`

    Parameters:
        weight: 4-dimensional tensor, the weight of a convolutional layer
        subvector_size: The size of vectors to split the weight matrix in
    Returns:
        permutation: List to organize input dimensions (ie, a permutation)
    """

    start_timestamp = time.time()

    c_out, c_in, h, w = weight.shape
    is_pointwise = (h == 1) and (w == 1)

    n_buckets = subvector_size // (h * w)
    max_entries_per_bucket = c_in // n_buckets

    init_cov_det = get_cov_det(weight.reshape(-1, subvector_size))

    buckets = {i: {"bucket": list(), "variance": 1.} for i in range(n_buckets)}

    # Compute variance per dimension
    if is_pointwise:
        variances = torch.var(weight.reshape(c_out, -1), dim=0).cpu()
    else:
        # or get determinant of covariance for larger groups
        variances = torch.Tensor([get_cov_det(weight[:, i, :, :].reshape(-1, h * w)) for i in range(c_in)])

    sorted_variances, indices = torch.sort(variances, descending=True)

    full_buckets = []
    for i, (variance, index) in enumerate(zip(sorted_variances, indices)):

        # Find the bucket with the least variance and add to it
        bucket_indices, bucket_variances = [], []
        for j, bucket_dict in buckets.items():
            bucket_indices.append(j)
            bucket_variances.append(bucket_dict["variance"])

        bucket_to_add = bucket_indices[np.argmin(bucket_variances)]

        # Update variance and bucket
        buckets[bucket_to_add]["bucket"].append(index.item())
        buckets[bucket_to_add]["variance"] = torch.var(weight[:, buckets[bucket_to_add]["bucket"]])

        if len(buckets[bucket_to_add]["bucket"]) >= max_entries_per_bucket:
            # Remove a bucket when it is full
            full_buckets.append(buckets.pop(bucket_to_add)["bucket"])

    # Interleave the buckets so they land on the same dimension after reshaping
    permutation = interleave_lists(full_buckets)

    end_timestamp = time.time()
    elapsed_s = end_timestamp - start_timestamp

    final_cov_det = get_cov_det(weight[:, permutation].reshape(-1, subvector_size))
    logging.info(f"Greedy: {init_cov_det:2e} -> {final_cov_det:2e}. Done in {(elapsed_s):.2f} seconds")

    return permutation


def optimize_permutation_by_stochastic_local_search(
    name: str, weight: torch.Tensor, subvector_size: int, n_iter: int, permutation: Optional[List] = None
) -> List[int]:
    """Uses stochastic local search to find a permutation that minimizes the determinant of the covariance of `weight`

    Parameters:
        name: The name of the layer whose permuation is being optimized. Only used for display purposes
        weight: 4-dimensional weight matrix whose determinant covariance we are minimizing
        subvector_size: Length of the vectors that `weight` will be quantized in
        n_iter: Number of iterations for the optimization
        permutation: Initial permutation for `weight`, usually obtained by a greedy method. Defaults to the None, in
                     which case we assign it the identity permutation
    Returns:
        permutation: List to organize input dimensions (ie, a permutation)
    """

    start_timestamp = time.time()
    weight = weight.cpu().numpy()

    c_out, c_in, h, w = weight.shape

    if permutation is None:
        permutation = list(np.arange(c_in))

    assert len(permutation) == c_in

    if n_iter == 0:
        return permutation

    # Initialization
    best_permutation = copy.copy(permutation)
    best_cov_det = get_cov_det(weight[:, best_permutation].reshape(-1, subvector_size))
    init_cov_det = best_cov_det

    # Obtain random pairs for search
    random_pairs = np.random.randint(c_in, size=(n_iter, 2))

    # Iterated thorugh the random pairs. Evaluate if they result in a lower covdet, and if so keep the solution
    with tqdm(random_pairs) as progress_bar:
        for i, j in random_pairs:
            candidate_permutation = copy.copy(best_permutation)
            candidate_permutation[i], candidate_permutation[j] = candidate_permutation[j], candidate_permutation[i]

            new_cov_det = get_cov_det(weight[:, candidate_permutation].reshape(-1, subvector_size))

            # Acceptance criterion
            if new_cov_det < best_cov_det:
                progress_bar.set_description(f"{name} {new_cov_det:2e}")
                best_permutation = candidate_permutation
                best_cov_det = new_cov_det
            progress_bar.update()

    end_timestamp = time.time()
    elapsed_s = end_timestamp - start_timestamp
    logging.info(f"SLS   : {init_cov_det:2e} -> {best_cov_det:2e}. Done in {(elapsed_s):.2f} seconds")
    return best_permutation
