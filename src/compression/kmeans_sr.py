# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Stochastic relaxations of k-means as discussed in [0]

[0]: Zeger, Kenneth, Jacques Vaisey, and Allen Gersho.
"Globally optimal vector quantizer design by stochastic relaxation."
IEEE Transactions on Signal Processing 40, no. 2 (1992): 310-322.
"""

from typing import Tuple

import torch

from .kmeans import assign_clusters, get_initial_codebook, slow_update_codebook, update_codebook


def get_noise_coefficient(iteration: int, max_iterations: int, p: float = 0.5) -> float:
    """Temperature decay schedule. This controls how much noise we add to the dataset during annealed noisy k-means
    (aka src). This is a function of the iteration number because we go from high to low noise

    Parameters:
        iteration: Current iteration
        max_iteration: Total number of iterations
        p: Power that controls decay speed
    """
    return (1 - (iteration / max_iterations)) ** p


def _add_noise(X: torch.Tensor, X_stdev: torch.Tensor, i: int, n_iters: int) -> torch.Tensor:
    """Adds noise to the training set"""
    noise = torch.randn_like(X) * X_stdev
    scheduled_training_set_noise = get_noise_coefficient(i, n_iters) * noise
    noisy_X = X + scheduled_training_set_noise
    return noisy_X


@torch.no_grad()
def src(
    training_set: torch.Tensor,
    k: int,
    n_iters: int,
    slow_cb_update: bool = False,
    resolve_empty_clusters: bool = False,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stochastic relaxation of k-means with decreasing noise in the process -- adding noise to training set

    Parameters:
        training_set: n x d matrix of training examples
        k: Number of centroids
        n_iters: Number of iterations
        resolve_empty_clusters: If k-means produces empty centroids, take care of them. Otherwise the codes may use
                                fewer than k centres in the codebook
        epsilon: Noise to add to initial codebook
    Returns:
        codebook: n-by-k matrix with the learned codebook
        codes: n-long vector with the learned codes
    """
    codebook = get_initial_codebook(training_set, k, epsilon)

    # Compute std of training_set
    training_set_std = training_set.std(dim=0)

    for i in range(n_iters + 1):
        codes = assign_clusters(training_set, codebook, resolve_empty_clusters)
        noisy_training_set = _add_noise(training_set, training_set_std, i, n_iters)
        if slow_cb_update:
            slow_update_codebook(codes, noisy_training_set, codebook)
        else:
            update_codebook(codes, noisy_training_set, codebook)

    return codebook, codes
