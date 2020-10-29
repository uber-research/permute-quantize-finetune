# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch

from .training import ModelTrainer
from .training_types import FinalSummary, TQDMState
from .validating import ModelValidator

# fmt: off
try:
    import horovod.torch as hvd
    HAVE_HOROVOD = True
except ImportError:
    HAVE_HOROVOD = False
# fmt: on


class ImagenetAccumulator(object):
    """Horovod-aware accumulator that keeps track of accuracy so far"""

    def __init__(self):
        self._n_total = 0
        self._n_correct = 0
        self._total_loss = 0.
        self._count = 0

    def reset(self):
        self._n_total = 0
        self._n_correct = 0
        self._total_loss = 0.
        self._count = 0

    def accumulate(self, targets: torch.Tensor, outputs: torch.Tensor, loss: torch.Tensor):
        """Updates the number of correct predictions and average loss so far.

        Parameters
            targets: The expected classes of the inputs to the model
            outputs: The classes that the model predicted
            loss: The loss that the model incurred on making its predictions
        """

        targets = targets.detach()
        outputs = outputs.detach()
        loss = loss.detach().cpu()

        _, predicted = outputs.max(dim=1)
        n_total = torch.tensor(targets.size(0), dtype=torch.float)
        n_correct = predicted.eq(targets).cpu().float().sum()

        if HAVE_HOROVOD:
            n_total = hvd.allreduce(n_total, average=False, name="accum_n_total")
            n_correct = hvd.allreduce(n_correct, average=False, name="accum_n_correct")
            loss = hvd.allreduce(loss, average=True, name="accum_loss")

        n_total = round(n_total.item())
        n_correct = round(n_correct.item())

        self._n_total += n_total
        self._n_correct += n_correct
        self._total_loss += loss.item()
        self._count += 1

        self._latest_state = {
            "loss": loss.item(),
            "acc": n_correct / n_total * 100,
            "correct": n_correct,
            "total": n_total,
        }

    def get_latest_state(self):
        return self._latest_state

    def get_average_state(self):
        return {
            "loss": self._total_loss / self._count,
            "acc": self._n_correct / self._n_total * 100,
            "correct": self._n_correct,
            "total": self._n_total,
        }


class ImagenetValidator(ModelValidator):
    def __init__(self, model, criterion):
        super().__init__()
        self._accumulator = ImagenetAccumulator()

        self._model = model
        self._criterion = criterion

    def reset(self):
        self._accumulator.reset()

    def update(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """Computes loss between what the model produces and the ground truth, and updates the accumulator

        Parameters:
            data: 2-tuple with inputs and targets for the model
        """

        inputs = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)

        outputs = self._model(inputs)
        loss = self._criterion(outputs, targets)

        self._accumulator.accumulate(targets, outputs, loss)

    def get_tqdm_state(self):
        state = self._accumulator.get_average_state()
        return TQDMState(
            {"loss": f'{state["loss"]:.2f}', "accuracy": f'{state["acc"]:.2f}% ({state["correct"]}/{state["total"]})'}
        )

    def get_final_summary(self):
        state = self._accumulator.get_average_state()
        return FinalSummary({"loss": state["loss"], "accuracy": state["acc"]})

    def get_final_metric(self):
        state = self._accumulator.get_average_state()
        return state["acc"]


class ImagenetTrainer(ModelTrainer):
    def __init__(self, model, optimizer, lr_scheduler, criterion):
        super().__init__(model, optimizer, lr_scheduler)
        self.accumulator = ImagenetAccumulator()
        self.criterion = criterion

    def reset(self):
        self.accumulator.reset()

    def pass_to_model(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss

    def update_state(self, targets: torch.Tensor, outputs: torch.Tensor, loss: torch.Tensor):
        self.accumulator.accumulate(targets, outputs, loss)
        state = self.accumulator.get_latest_state()
        self.latest_state = {"loss": state["loss"], "accuracy": state["acc"]}

    def get_final_summary(self):
        state = self.accumulator.get_average_state()
        return FinalSummary(
            {"loss": f'{state["loss"]:.2f}', "accuracy": f'{state["acc"]:.02f}% ({state["correct"]}/{state["total"]})'}
        )


def get_imagenet_criterion():
    """Gets the typical training loss for Imagenet classification"""
    return torch.nn.CrossEntropyLoss()
