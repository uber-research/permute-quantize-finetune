# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .AbstractDataLogger import AbstractDataLogger
from .training_types import FinalSummary, IntermediateSummary, TQDMState


class AbstractDataHandler(ABC):
    """Abstract class for handling a dataset. Can be instantiated as a trainer or validator"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, data):
        pass

    @abstractmethod
    def get_tqdm_state(self) -> TQDMState:
        pass

    @abstractmethod
    def get_intermediate_summary(self) -> IntermediateSummary:
        pass

    @abstractmethod
    def get_final_summary(self) -> FinalSummary:
        pass

    @abstractmethod
    def get_final_metric(self) -> float:
        pass

    def handle_data(
        self, epoch: int, data_loader: torch.utils.data.DataLoader, logger: AbstractDataLogger, verbose: bool
    ) -> float:
        """Traverses the data loader and processes the data according to the `update` method. Keeps track of progress.

        Parameters:
            epoch:
        """

        n_batches = len(data_loader)
        progress_data = tqdm(data_loader, desc=logger.get_desc("Epoch", epoch), disable=not verbose)
        for batch_idx, data in enumerate(progress_data):
            # Epoch starts at 1, but index starts at 0
            idx = (epoch - 1) * n_batches + batch_idx

            self.update(data)
            if verbose:
                progress_data.set_postfix(self.get_tqdm_state())
                logger.log_intermediate_summary(idx, self.get_intermediate_summary())

        if verbose:
            logger.log_final_summary(epoch, self.get_final_summary())
        return self.get_final_metric()
