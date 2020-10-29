# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from tensorboardX import SummaryWriter

from ..utils.logging import log_to_summary_writer
from .AbstractDataHandler import AbstractDataHandler
from .AbstractDataLogger import AbstractDataLogger
from .training_types import FinalSummary, IntermediateSummary


class ModelValidator(AbstractDataHandler):
    def __init__(self):
        super().__init__()

    def get_intermediate_summary(self):
        return IntermediateSummary({})


class ValidationLogger(AbstractDataLogger):
    def __init__(self, batches_per_epoch: int, summary_writer: Optional[SummaryWriter]):
        super().__init__("Validation")
        self.batches_per_epoch = batches_per_epoch
        self.summary_writer = summary_writer

    def log_intermediate_summary(self, idx: int, summary: IntermediateSummary):
        pass

    def log_final_summary(self, epoch: int, summary: FinalSummary):
        log_to_summary_writer("Validation", epoch * self.batches_per_epoch, summary, self.summary_writer)
        statement = ", ".join(f"{k} ({v:.2f})" for k, v in summary.items())
        print(f'{self.get_desc("Epoch", epoch)}: {statement}')


@torch.no_grad()
def validate_one_epoch(
    epoch: int,
    val_data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    validator: ModelValidator,
    logger: ValidationLogger,
    verbose: bool,
) -> float:
    """Perform validation given a model, a validator, and possibly writing to tensorboard

    Parameters:
        epoch: Current epoch count
        val_data_loader: Pytorch data loader for the validation set
        model: Network to validate
        validator: Model validator
        logger: Validation logger
        verbose: Whether to write to logs
    Returns:
        metric: Final metric given by the validator
    """
    model.eval()
    validator.reset()
    return validator.handle_data(epoch, val_data_loader, logger, verbose)
