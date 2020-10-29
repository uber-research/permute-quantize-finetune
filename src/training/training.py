# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import abstractmethod
from typing import Optional, Tuple

import torch
from tensorboardX import SummaryWriter

from ..utils.logging import log_to_summary_writer
from .AbstractDataHandler import AbstractDataHandler
from .AbstractDataLogger import AbstractDataLogger
from .lr_scheduler import LR_Scheduler
from .training_types import FinalSummary, IntermediateSummary, TQDMState


class ModelTrainer(AbstractDataHandler):
    latest_state = {}

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: LR_Scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def update(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Does one training step on a batch of data

        Parameters:
            data: A tuple with inputs and expected outputs
        """

        inputs = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)

        self.optimizer.zero_grad()
        outputs, loss = self.pass_to_model(inputs, targets)
        self.update_state(targets, outputs, loss)

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler.step_batch():
            self.lr_scheduler.step()

    @abstractmethod
    def pass_to_model(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def update_state(self, targets: torch.Tensor, outputs, loss: torch.Tensor):
        pass

    def get_tqdm_state(self) -> TQDMState:
        return TQDMState(self.latest_state)

    def get_intermediate_summary(self) -> IntermediateSummary:
        return IntermediateSummary({"learning_rate": self.optimizer.param_groups[0]["lr"], **self.latest_state})

    def get_final_metric(self) -> float:
        return -math.inf  # unused value


class TrainingLogger(AbstractDataLogger):
    def __init__(self, summary_writer: Optional[SummaryWriter]):
        super().__init__("Training")
        self.summary_writer = summary_writer

    def log_intermediate_summary(self, idx: int, summary: IntermediateSummary):
        log_to_summary_writer("Train", idx, summary, self.summary_writer)

    def log_final_summary(self, epoch: int, summary: FinalSummary):
        statement = ", ".join(f"{k}: {v}" for k, v in summary.items())
        print(f'{self.get_desc("Epoch", epoch)}: {statement}')


@torch.enable_grad()
def train_one_epoch(
    epoch: int,
    train_sampler: Optional[torch.utils.data.DistributedSampler],
    train_data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    trainer: ModelTrainer,
    logger: TrainingLogger,
    verbose: bool,
) -> None:
    """Perform one epoch of training given a model, a trainer, and possibly writing to tensorboard

    Parameters:
        epoch: Current epoch count
        train_data_loader: PyTorch dataloader for the training set
        model: Network to train
        trainer: Model trainer
        logger: Training logger
        verbose: Whether to write to logs
    """
    model.train()
    trainer.reset()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    trainer.handle_data(epoch, train_data_loader, logger, verbose)
