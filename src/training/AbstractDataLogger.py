# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

from .training_types import FinalSummary, IntermediateSummary


class AbstractDataLogger(ABC):
    """Abstract class for logging progress during training or validation"""

    def __init__(self, desc: str):
        self.desc = desc

    def get_desc(self, infix: str, value: int):
        return f"{self.desc} {infix} #{value}"

    @abstractmethod
    def log_intermediate_summary(self, idx: int, summary: IntermediateSummary):
        pass

    @abstractmethod
    def log_final_summary(self, epoch: int, summary: FinalSummary):
        pass
