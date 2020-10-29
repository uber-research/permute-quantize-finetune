# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from ..dataloading.coco_eval import CocoEvaluator
from ..training.training import ModelTrainer
from ..training.training_types import FinalSummary, TQDMState
from ..training.validating import ModelValidator
from ..utils import to_cpu, to_cuda


class CoCoValidator(ModelValidator):
    def __init__(self, model, val_data_loader):
        super().__init__()
        self.model = model

        self.iou_types = ["bbox", "segm"]
        self.coco = val_data_loader.dataset.coco
        self.evaluator = CocoEvaluator(self.coco, self.iou_types)

    def reset(self):
        self.evaluator = CocoEvaluator(self.coco, self.iou_types)

    def update(self, data):
        image_ids = data[0]
        images = to_cuda(data[1])

        outputs = self.model(images)
        outputs = to_cpu(outputs)

        self.evaluator.update({image_id: output for image_id, output in zip(image_ids, outputs)})

    def get_tqdm_state(self):
        return TQDMState({})

    def get_final_summary(self):
        self.evaluator.synchronize_between_processes()
        self.evaluator.accumulate()
        self.evaluator.summarize()

        state = self.evaluator.coco_eval

        return FinalSummary({"BoxAP": state["bbox"].stats[0] * 100, "MaskAP": state["segm"].stats[0] * 100})

    def get_final_metric(self):
        state = self.evaluator.coco_eval
        return (state["bbox"].stats[0] + state["segm"].stats[0]) * 100


class CoCoTrainer(ModelTrainer):
    def __init__(self, model, optimizer, lr_scheduler, criterion):
        super().__init__(model, optimizer, lr_scheduler)
        self.criterion = criterion

    def update(self, data):
        inputs = to_cuda(data[0])
        targets = to_cuda(data[1])

        self.optimizer.zero_grad()
        outputs, loss = self.pass_to_model(inputs, targets)

        self.update_state(targets, outputs)
        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler.step_batch():
            self.lr_scheduler.step()

    def reset(self):
        pass

    def pass_to_model(self, inputs, targets):
        outputs = self.model(inputs, targets)
        loss = self.criterion(outputs)

        return outputs, loss

    @torch.no_grad()
    def update_state(self, targets, outputs):
        total_loss = sum(loss.item() for loss in outputs.values())
        self.latest_state = {
            "loss_total": total_loss,
            **{f"{loss_name}": loss_value.item() for loss_name, loss_value in outputs.items()},
        }

    def get_final_summary(self):
        return FinalSummary({})


def get_coco_criterion():
    """Get the loss function typically used to train on CoCo.
    Returns:
        a function from outputs => loss mapping, on which .backward() can be called
    """

    def loss_function(out):
        # We add all the loses that the network returns (classifier, box regression, mask, etc)
        total_loss = 0.

        for loss_name, loss_value in out.items():
            total_loss += loss_value

        return total_loss

    return loss_function
