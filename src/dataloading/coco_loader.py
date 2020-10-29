# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

from torch.utils.data import DataLoader

from .coco_dataset import CocoTrainingSet, CocoValidationSet

COCO_PATH_PLACEHOLDER = "<your_coco_path_here>"


def _coco_collate_fn(data):
    return tuple(zip(*data))


def check_coco_path_is_set(coco_path: str):
    """Ensure the coco path is set to other than the default"""
    if coco_path == COCO_PATH_PLACEHOLDER:
        raise ValueError(f"{COCO_PATH_PLACEHOLDER} is not a valid path. Did you forget to update the config file?")


def load_coco_val(coco_path: str, batch_size: int, num_workers: int, shuffle: Optional[bool] = False) -> DataLoader:
    """Returns a validation set for COCO according to the images directory and the annotations file"""

    check_coco_path_is_set(coco_path)

    images_dir = os.path.join(coco_path, "val2017")
    annotations_file = os.path.join(coco_path, "annotations", "instances_val2017.json")

    dataset = CocoValidationSet(images_dir, annotations_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=_coco_collate_fn,
    )


def load_coco_train(coco_path: str, batch_size: int, num_workers: int, shuffle: Optional[bool] = True) -> DataLoader:
    """Returns a training set for COCO according to the images directory and the annotations file"""

    check_coco_path_is_set(coco_path)

    images_dir = os.path.join(coco_path, "train2017")
    annotations_file = os.path.join(coco_path, "annotations", "instances_train2017.json")

    dataset = CocoTrainingSet(images_dir, annotations_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=_coco_collate_fn,
    )
