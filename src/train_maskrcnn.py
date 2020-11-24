# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from datetime import datetime

from .compression.model_compression import compress_model
from .dataloading.coco_loader import load_coco_train, load_coco_val
from .permutation.model_permutation import permute_model
from .training.coco_utils import CoCoTrainer, CoCoValidator, get_coco_criterion
from .training.lr_scheduler import get_learning_rate_scheduler
from .training.optimizer import get_optimizer
from .training.training import TrainingLogger, train_one_epoch
from .training.validating import ValidationLogger, validate_one_epoch
from .utils.config_loader import load_config
from .utils.logging import (
    get_tensorboard_logger,
    initialize_horovod,
    log_compression_ratio,
    log_config,
    setup_pretty_logging,
)
from .utils.model_size import compute_model_nbits
from .utils.models import get_uncompressed_model
from .utils.state_dict_utils import save_state_dict_compressed

_MODEL_OUTPUT_PATH_SUFFIX = "trained_models"


def main():
    setup_pretty_logging()
    verbose = initialize_horovod()
    start_timestamp = datetime.now()

    file_path = os.path.dirname(__file__)
    default_config = os.path.join(file_path, "../config/train_maskrcnn.yaml")
    config = load_config(file_path, default_config_path=default_config)
    summary_writer = get_tensorboard_logger(config["output_path"])
    log_config(config, summary_writer)

    model_config = config["model"]
    compression_config = model_config["compression_parameters"]
    model = get_uncompressed_model(model_config["arch"], pretrained=True).cuda()

    if "permutations" in model_config and model_config.get("use_permutations", True):
        permute_model(
            model,
            compression_config["fc_subvector_size"],
            compression_config["pw_subvector_size"],
            compression_config["large_subvectors"],
            permutation_groups=model_config.get("permutations", []),
            layer_specs=compression_config.get("layer_specs", None),
            permutation_specs=model_config.get("permutation_specs", {}),
            sls_iterations=model_config["sls_iterations"],
        )

    uncompressed_model_size_bits = compute_model_nbits(model)
    model = compress_model(model, **compression_config).cuda()
    compressed_model_size_bits = compute_model_nbits(model)
    log_compression_ratio(uncompressed_model_size_bits, compressed_model_size_bits, summary_writer=summary_writer)

    # Create training and validation dataloaders
    dataloader_config = config["dataloader"]
    val_data_loader = load_coco_val(
        dataloader_config["coco_path"], dataloader_config["batch_size"], dataloader_config["num_workers"]
    )
    train_data_loader = load_coco_train(
        dataloader_config["coco_path"], dataloader_config["batch_size"], dataloader_config["num_workers"]
    )

    # Get CoCo optimizer, criterion, trainer and validator
    optimizer = get_optimizer(model, config)
    criterion = get_coco_criterion()
    n_epochs = config["epochs"]
    assert n_epochs > 0
    batches_per_epoch = len(train_data_loader)
    lr_scheduler = get_learning_rate_scheduler(config, optimizer, n_epochs, batches_per_epoch)

    trainer = CoCoTrainer(model, optimizer, lr_scheduler, criterion)
    training_logger = TrainingLogger(summary_writer)
    validator = CoCoValidator(model, val_data_loader)
    validation_logger = ValidationLogger(batches_per_epoch, summary_writer)

    # Keep track of the best validation accuracy we have seen to save the best model at the end of every epoch
    best_acc = -math.inf
    best_acc_epoch = -1
    last_acc = -math.inf

    if not config.get("skip_initial_validation", False):
        last_acc = validate_one_epoch(0, val_data_loader, model, validator, validation_logger, verbose)
        best_acc = last_acc
        best_acc_epoch = 0

    # Save model after zero epochs
    save_state_dict_compressed(model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "0.pth"))

    for epoch in range(1, n_epochs + 1):

        train_one_epoch(epoch, None, train_data_loader, model, trainer, training_logger, verbose)

        # Save the current state of the model after every epoch
        save_state_dict_compressed(
            model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, f"{epoch}.pth")
        )

        last_acc = validate_one_epoch(epoch, val_data_loader, model, validator, validation_logger, verbose)
        if lr_scheduler.step_epoch():
            lr_scheduler.step(last_acc)

        if last_acc > best_acc:
            save_state_dict_compressed(
                model, os.path.join(config["output_path"], _MODEL_OUTPUT_PATH_SUFFIX, "best.pth")
            )
            best_acc = last_acc
            best_acc_epoch = epoch

    # Done training!
    if verbose:
        print("Done training!")
        summary_writer.close()
        with open(os.path.join(config["output_path"], "results.txt"), "w") as f:
            print(f"{start_timestamp:%Y-%m-%d %H:%M:%S}", file=f)
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}", file=f)
            print(last_acc, file=f)
            print(best_acc, file=f)
            print(best_acc_epoch, file=f)


if __name__ == "__main__":
    main()
