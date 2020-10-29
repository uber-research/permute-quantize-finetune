# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch


def get_initial_codebook(training_set: torch.Tensor, k: int, epsilon: float = 1e-5) -> torch.Tensor:
    """Creates the initial codebook by sampling k points from the dataset and adding noise to them

    Parameters:
        training_set: n-by-d matrix of training examples
        k: Number of centroids
    Return:
        codebook: k-by-d matrix with the initial codebook for k-means
    """
    n, d = training_set.shape

    noise = torch.empty(k, training_set.size(1)).normal_(0, epsilon).to(training_set.device)
    initial_indices = torch.randperm(n)[0:k].to(training_set.device)
    codebook = training_set[initial_indices, :] + noise

    return codebook


@torch.no_grad()
def kmeans(
    training_set: torch.Tensor,
    k: int,
    n_iters: int,
    slow_cb_update: bool = False,
    resolve_empty_clusters: bool = False,
    epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain k-means in PyTorch

    Parameters:
        training_set: n-by-d matrix of training examples
        k: Number of centroids
        n_iters: Number of iterations
        resolve_empty_clusters: If k-means produces empty centroids, create new ones until no empty centroids remain
        epsilon: Noise to add to initial codebook
    Returns:
        codebook: k-by-d centroids
        codes: n-long vector with assignments to the centroids
    """
    codebook = get_initial_codebook(training_set, k, epsilon)

    for _ in range(n_iters):
        codes = assign_clusters(training_set, codebook, resolve_empty_clusters)
        if slow_cb_update:
            slow_update_codebook(codes, training_set, codebook)
        else:
            update_codebook(codes, training_set, codebook)

    return codebook, codes


def pairwise_squared_distances(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Torch implementation of fast pairwise distances using $||a-b||^2 = ||a||^2 - 2ab + ||b||^2$.

    Parameters:
        x: n-by-m matrix
        y: p-by-m matrix. Optional, defaults to x
    Returns:
        dist: n-by-p matrix with pairwise distance between x and y
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)

    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def resolve_empty_clusters(
    training_set: torch.Tensor, codes: torch.Tensor, codebook: torch.Tensor, epsilon: float = 1e-1
) -> torch.Tensor:
    """Checks if there exists a centroid in the codebook that is not assigned to any element in the training set.
    If so, create a new centroid by adding noise to the most popular centroid.

    Parameters:
        training_set: k means training set
        codes: Assignments from training set to codebook
        codebook: k means centroids
        epsilon: Value used to perturb most popular centroid
    Returns:
        codes: New assignment of training set vectors without empty centroids
    """

    k = codebook.size(0)

    has_empty_cluster = len(codes.unique()) < k

    if not has_empty_cluster:
        return codes

    modal_codes = codes.mode().indices

    # TODO: vectorize this somehow
    for k in range(codebook.size(0)):
        if k not in codes:
            # initialize this unassigned centroid based on the most assigned-to centroid
            has_empty_cluster = True

            if modal_codes.dim() == 0:
                non_empty_cluster_idx = codes[modal_codes]
            else:
                non_empty_cluster_idx = codes[k % modal_codes.size(0)]

            non_empty_cluster = codebook[non_empty_cluster_idx, :]

            codebook[k, :] = non_empty_cluster + torch.normal(0, epsilon, size=(1,), device=codebook.device)

    if has_empty_cluster:
        codes = assign_clusters(training_set, codebook, handle_empty_clusters=True)

    return codes


def assign_clusters(
    training_set: torch.Tensor, codebook: torch.Tensor, handle_empty_clusters: bool = True
) -> torch.Tensor:
    """Given a training set and codebook, finds an optimal assignment of vectors in the training set to elements in the
    codebook. This implementation is batched to avoid running OOM in the GPU.

    Parameters:
        training_set: n x d matrix of training examples
        codebook: k x d matrix representing codebook
        handle_empty_clusters: If k-means produces empty centroids, create new ones until no empty centroids remain
    Returns:
        codes: n-long vector with assignment of training_examples to the codebook
    """

    N = training_set.size(0)

    # Training set batch size to assign to codebooks.
    # Small numbers use less GPU memory, but large numbers run faster, reduce this if you
    # run out of GPU memory.
    if N > 16:
        TRAINING_SET_BATCH_MAX_BATCH_SIZE = N // 16
    else:
        TRAINING_SET_BATCH_MAX_BATCH_SIZE = N

    batch_size = min(TRAINING_SET_BATCH_MAX_BATCH_SIZE, N)

    num_batches = int(N // batch_size + bool(N % batch_size))

    codes = []  # codes for each batch

    for i in range(num_batches):
        batch_start_idx = int(i * batch_size)
        batch_end_idx = int(min((i + 1) * batch_size, N))

        if batch_end_idx == 0:
            # Edge case when the dataset is very small
            assert num_batches == 1
            batch_end_idx = N

        batch_training_set = training_set[batch_start_idx:batch_end_idx, :]
        batch_distances = pairwise_squared_distances(batch_training_set, codebook)
        del batch_training_set

        batch_codes = batch_distances.argmin(dim=1)
        del batch_distances

        codes.append(batch_codes)

    # Combine the codes for the entire training set
    codes = torch.cat(codes, dim=0)

    # Resolve unassigned centroids if needed
    if handle_empty_clusters:
        codes = resolve_empty_clusters(training_set, codes, codebook)

    return codes


def update_codebook(codes: torch.Tensor, training_set: torch.Tensor, codebook: torch.Tensor) -> None:
    """Updates the codebook according to the given code assignments.
    This is the centroid update step in k-means, taking the mean of all the vectors assigned to a centroid.

    Parameters:
        codes: Assignments from training set to centroids
        training_set: Training set elements
        codebook: Codebook tensor that needs to be updated
    Returns:
        None. The codebook will be updated in place
    """
    codes_range = torch.arange(0, codebook.size(0), device=training_set.device)

    index_mask = (codes.view(-1, 1) == codes_range).float()
    normalized_index_mask = index_mask / index_mask.sum(dim=0).clamp(min=1)

    new_codebook = normalized_index_mask.t().mm(training_set)
    codebook.data.copy_(new_codebook.data)


def slow_update_codebook(codes: torch.Tensor, training_set: torch.Tensor, codebook: torch.Tensor) -> None:
    """Updates the codebook according to the given code assignments. This is an iterative approach that is slower, but
    uses less memory. We use this to compress the fully connected layer of ResNet50.

    Parameters:
        codes: Assignments from training set to centroids
        training_set: Training set elements
        codebook: Codebook tensor that needs to be updated
    """
    new_codebook = torch.zeros(codebook.size(), device=training_set.device)
    for i in range(codebook.size(0)):
        cluster = training_set[codes == i]
        if cluster.size(0) > 0:
            new_codebook[i] = cluster.mean(dim=0)

    codebook.data.copy_(new_codebook.data)
