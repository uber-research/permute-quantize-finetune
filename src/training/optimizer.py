# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch

from ..utils.horovod_utils import get_distributed_learning_rate, get_distributed_optimizer


def get_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Gets the optimizer for training and distributes it to horovod workers.

    Parameters:
        model: Network which we wish to train
        config: Config dict specifying training hyper-parameters
    Returns:
        optimizer: Optimizer to use during fine-tuning
    """
    optimizer_type = config["optimizer"]
    learning_rate = config["learning_rate"]

    learning_rate = get_distributed_learning_rate(learning_rate)

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=config["momentum"], weight_decay=config["weight_decay"]
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer must be either `sgd` or `adam`, not {optimizer_type}")

    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = learning_rate

    optimizer = get_distributed_optimizer(optimizer, model)

    return optimizer
