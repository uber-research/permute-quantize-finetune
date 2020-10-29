# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from bisect import bisect_right
from typing import Dict, List, Optional

import torch

from ..utils.horovod_utils import distribute_optimizer_state, get_distributed_learning_rate


class LR_Scheduler(object):
    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def step_epoch(self) -> bool:
        return False

    def step_batch(self) -> bool:
        return False

    def step(self, metric: Optional[float] = None) -> None:
        if self.lr_scheduler is None:
            return

        if metric is None:
            self.lr_scheduler.step()
        else:
            self.lr_scheduler.step(metric)


class ReduceLROnPlateau(LR_Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        n_batches: int,
        patience: int,
        factor: float,
        min_lr: float,
    ):
        super().__init__(
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr)
        )

    def step_epoch(self) -> bool:
        return True


class CosineAnnealingLR(LR_Scheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, n_epochs: int, n_batches: int, eta_min: float, last_epoch: int
    ):
        t_max = n_batches * n_epochs
        base_lr = optimizer.param_groups[0]["lr"]
        last_epoch = (last_epoch + 1) * n_batches - 1

        # This is to bypass pytorch's weird requirements for last_epoch.
        # This is the closed form directly taken from
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if last_epoch > 0:
            learning_rate = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * last_epoch / t_max)) / 2
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        super().__init__(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min, last_epoch=last_epoch)
        )

    def step_batch(self) -> bool:
        return True


class MultiStepLR(LR_Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        n_batches: int,
        gamma: float,
        milestones: List[int],
        last_epoch: int,
    ):
        base_lr = optimizer.param_groups[0]["lr"]

        # This is to bypass pytorch's weird requirements for last_epoch.
        # This is the closed form directly taken from
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        # In the pytorch code, milestones is a Counter and that makes the following code buggy.
        # We bypass that here, so there is no issues.
        if last_epoch > 0:
            learning_rate = base_lr * gamma ** bisect_right(milestones, last_epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        super().__init__(
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
        )

    def step_epoch(self) -> bool:
        return True

    def step(self, metric: float) -> None:
        super().step()


def get_learning_rate_scheduler(
    config: Dict, optimizer: torch.optim.Optimizer, n_epochs: int, n_batches: int
) -> LR_Scheduler:
    """
    Get the scheduler for the learning rate according to the config
    Parameters:
        config: Main config dict
        optimizer: Optimizer which needs its LR scheduled
        n_epochs: Total number of epochs
        n_batches: Number of batches in training set
    """
    scheduler_args = config.get("lr_scheduler", {"type": "plateau", "patience": 15, "factor": 0.1, "min_lr": 1.e-7})

    scheduler_type = scheduler_args["type"]
    scheduler = None

    if scheduler_type == "plateau":
        min_lr = scheduler_args.get("min_lr", 1.e-8)
        min_lr = get_distributed_learning_rate(min_lr)

        scheduler = ReduceLROnPlateau(
            optimizer,
            n_epochs,
            n_batches,
            patience=scheduler_args["patience"],
            factor=scheduler_args["factor"],
            min_lr=min_lr,
        )
    elif scheduler_type == "cosine":
        min_lr = scheduler_args.get("min_lr", 0)
        min_lr = get_distributed_learning_rate(min_lr)

        scheduler = CosineAnnealingLR(
            optimizer, n_epochs, n_batches, eta_min=min_lr, last_epoch=scheduler_args.get("last_epoch", -1)
        )
    elif scheduler_type == "multistep":
        scheduler = MultiStepLR(
            optimizer,
            n_epochs,
            n_batches,
            milestones=scheduler_args.get("milestones", [3, 6, 9]),
            gamma=scheduler_args["factor"],
            last_epoch=scheduler_args.get("last_epoch", -1),
        )
    elif scheduler_type == "none":
        return LR_Scheduler(None)
    else:
        raise (f"Undefined LR scheduler type: {scheduler_type}")

    distribute_optimizer_state(optimizer)

    return scheduler
