# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads a pretrained compressed model and evaluate its performance on imagenet"""

import os

from .compression.model_compression import compress_model
from .dataloading.imagenet_loader import load_imagenet_val
from .training.imagenet_utils import ImagenetValidator, get_imagenet_criterion
from .training.validating import ValidationLogger, validate_one_epoch
from .utils.config_loader import load_config
from .utils.horovod_utils import initialize_horovod
from .utils.logging import log_compression_ratio, setup_pretty_logging
from .utils.model_size import compute_model_nbits
from .utils.models import get_uncompressed_model
from .utils.state_dict_utils import load_state_dict


def main():
    setup_pretty_logging()
    verbose = initialize_horovod()

    # specify config file to use in case user does not pass in a --config argument
    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../config/evaluate_resnet.yaml")
    config = load_config(file_path, default_config_path=default_config)

    compression_config = config["model"]["compression_parameters"]

    model = get_uncompressed_model(config["model"]["arch"], pretrained=False).cuda()
    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits)

    model = load_state_dict(model, os.path.join(file_path, config["model"]["state_dict_compressed"]))

    dataloader_config = config["dataloader"]
    val_data_sampler, val_data_loader = load_imagenet_val(
        dataloader_config["imagenet_path"],
        dataloader_config["num_workers"],
        dataloader_config["batch_size"],
        shuffle=dataloader_config["validation_shuffle"],
    )
    validator = ImagenetValidator(model, get_imagenet_criterion())
    logger = ValidationLogger(1, None)
    validate_one_epoch(0, val_data_loader, model, validator, logger, verbose)


if __name__ == "__main__":
    main()
