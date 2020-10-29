# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Tuple

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder

from ..utils.horovod_utils import get_distributed_sampler

IMAGENET_PATH_PLACEHOLDER = "<your_imagenet_path_here>"

# These correspond to the average and standard deviation RGB colours of Imagenet and are commonly used for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDEV = [0.229, 0.224, 0.225]

IMAGENET_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)

IMAGENET_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STDEV),
    ]
)


def _load_imagenet(
    rootpath: str, suffix: str, transform: transforms.Compose, num_workers: int, batch_size: int, shuffle: bool = False
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the imagenet dataset

    Parameters:
        roothpath: Path to the imagenet folder before `train` or `val` folder
        suffix: Either `train` or `val`. Will be appended to `rootpath`
        transform: Operations to apply to the data before passing it to the model (eg. for data augmentation)
        num_workers: Number of pytorch workers to use when loading data
        batch_size: Size of batch to give to the networks
        shuffle: Whether to randomly shuffle the data
    Returns:
        sampler: A PyTorch DataSampler that decides the order in which the data is fetched
        loader: A PyTorch DataLoader that fetches the data for the model
    """
    if rootpath == IMAGENET_PATH_PLACEHOLDER:
        raise ValueError(f"{IMAGENET_PATH_PLACEHOLDER} is not a valid path. Did you forget to update the config file?")

    dirname = os.path.join(rootpath, suffix)
    dataset = ImageFolder(dirname, transform)

    sampler = get_distributed_sampler(dataset, shuffle)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True)
    return sampler, loader


def load_imagenet_val(
    rootpath: str, num_workers: int, batch_size: int, shuffle: bool = False
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the training partition of Imagenet"""
    return _load_imagenet(rootpath, "val", IMAGENET_VAL_TRANSFORM, num_workers, batch_size, shuffle)


def load_imagenet_train(
    rootpath: str, num_workers: int, batch_size: int, shuffle: bool = True
) -> Tuple[Sampler, DataLoader]:
    """Creates a sampler and dataloader for the validation partition of Imagenet"""
    return _load_imagenet(rootpath, "train", IMAGENET_TRAIN_TRANSFORM, num_workers, batch_size, shuffle)
