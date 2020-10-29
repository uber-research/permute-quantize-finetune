# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, NewType

Summary = NewType("Summary", Dict)
TQDMState = NewType("TQDMState", Dict)
IntermediateSummary = NewType("InterSummary", Summary)
FinalSummary = NewType("FinalSummary", Summary)
