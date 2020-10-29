# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
from torch.utils.data.distributed import DistributedSampler

# fmt: off
try:
    import horovod.torch as hvd
    HAVE_HOROVOD = True
except ImportError:
    HAVE_HOROVOD = False
# fmt: on


def initialize_horovod() -> bool:
    """Initialize horovod if it is available.

    Returns:
        Whether this is the zeroth worker if horovod is available. True otherwise.
    """

    if HAVE_HOROVOD:
        hvd.init()
        torch.set_num_threads(4)
        torch.cuda.set_device(hvd.local_rank())
        print(f"Horovod: {hvd.rank() + 1}/{hvd.size()}")
        return hvd.rank() == 0
    else:
        return True


# === Optimizer ===
def get_distributed_learning_rate(learning_rate: float) -> bool:
    """Returns a learning rate scaled by the number of workers if horovod is available"""
    if HAVE_HOROVOD:
        return learning_rate * hvd.size()
    else:
        return learning_rate


def get_distributed_optimizer(optimizer: torch.optim.Optimizer, model: torch.nn.Module):
    """Distributes the learning rate if horovod is available"""
    if HAVE_HOROVOD:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    else:
        return optimizer


# === Learning rate scheduler ===
def distribute_optimizer_state(optimizer: torch.optim.Optimizer):
    """Distributes the optimizer state if horovod is available"""
    if HAVE_HOROVOD:
        state_dict = hvd.broadcast_object(optimizer.state_dict(), root_rank=0)
        if hvd.rank() > 0:
            optimizer.load_state_dict(state_dict)


# === Dataset sampler ===
def get_distributed_sampler(dataset: torch.utils.data.Dataset, shuffle: bool) -> Union[DistributedSampler, None]:
    """Returns a distributed sampler if horovod is available"""
    if HAVE_HOROVOD:
        return DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=shuffle)
    else:
        return None
