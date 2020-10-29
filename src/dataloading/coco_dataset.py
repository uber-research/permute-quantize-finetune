# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection


def _flip_coco_person_keypoints(kps, width):
    """Flips the keypoints about the vertical axis of the image. This is used for data augmentation"""
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class RandomHorizontalFlip(object):
    """Transform that flips images, boxes, and masks"""

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class CocoValidationSet(CocoDetection):
    def __init__(self, img_folder, annotation_file, transform=transforms.ToTensor()):
        super().__init__(img_folder, annotation_file, transform)

    def __getitem__(self, idx):
        image, target = super(CocoValidationSet, self).__getitem__(idx)
        image_id = self.ids[idx]

        return image_id, image, target


class CocoTrainingSet(CocoDetection):
    def __init__(self, img_folder, annotation_file, transform=transforms.ToTensor()):
        super().__init__(img_folder, annotation_file, transform)
        self._transforms = RandomHorizontalFlip(0.5)

        # filter for only images with valid box/mask annotations etc.
        self._filter_invalid_image_ids(self.ids, self.coco)

    def __getitem__(self, idx):
        img, target = super(CocoTrainingSet, self).__getitem__(idx)

        # convert the target into PyTorch training format for maskrcnn_resnet50_fpn
        target = self._convert_target(img, target)

        # randomly flip images and masks during training
        return self._transforms(img, target)

    def _filter_invalid_image_ids(self, image_ids, coco):
        """Remove image_ids without annotations etc."""

        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        MIN_KEYPOINTS_PER_IMAGE = 10

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False
            # keypoints task have a slight different critera for considering
            # if an annotation is valid
            if "keypoints" not in anno[0]:
                return True
            # for keypoint detection tasks, only consider valid images those
            # containing at least min_keypoints_per_image
            if _count_visible_keypoints(anno) >= MIN_KEYPOINTS_PER_IMAGE:
                return True
            return False

        filtered_ids = []

        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)

            if _has_valid_annotation(anno):
                # CocoTrainingSet._remove_degenerate_boxes(anno)
                filtered_ids.append(img_id)

        self.ids = filtered_ids

    def _convert_target(self, image, target):
        """Convert image and target to format compatible with pytorch training"""

        masks = []
        boxes = []
        labels = []

        W = image.size(1)
        H = image.size(2)

        for observation in target:
            # convert boxes from xywh to x1y1x2y2 format
            x, y, w, h = observation["bbox"]

            # If width or height are zero, skip the annotation. This avoids unpleasant NaNs during training.
            if w <= 0 or h <= 0:
                continue

            boxes.append(torch.Tensor([x, y, x + w, y + h]))

            # convert polygon annotations to bitmask
            np_mask = self.coco.annToMask(observation)
            masks.append(torch.from_numpy(np_mask))

            labels.append(observation["category_id"])

        if len(boxes) == 0:
            logging.warning("Got zero boxes")

        return {
            "boxes": torch.stack(boxes, 0) if len(boxes) > 0 else torch.zeros((0, 4)),
            "labels": torch.Tensor(labels).long(),
            "masks": torch.stack(masks, 0) if len(masks) > 0 else torch.zeros((0, W, H)),
        }
