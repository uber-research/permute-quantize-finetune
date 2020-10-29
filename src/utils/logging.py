# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from tensorboardX import SummaryWriter

from ..training.training_types import Summary


def setup_pretty_logging(log_level=logging.INFO, force_stdout=False) -> None:
    """Make sure that logs print in a nice format -- borrowed from @andreib

    Parameters:
        log_level: Default level to log at (eg. logging.DEBUG or logging.INFO)
        force_stdout:  Whether to output to stdout instead of stderr
    """
    log_format = "%(levelname)s:[%(asctime)s] %(message)s"
    log_date_format = "%Y/%m/%d %H:%M:%S"
    if force_stdout:
        logging.basicConfig(
            format=log_format, datefmt=log_date_format, level=log_level, handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(format=log_format, datefmt=log_date_format, level=log_level)


def get_tensorboard_logger(output_path: str) -> SummaryWriter:
    """Gets a logger than can write to tensorboard

    Parameters:
        output_path: The tensorboard output directory
    Returns:
        summary_writer: Tensorboard writer to specified output directory
    """
    tensorboard_output_path = Path(output_path, "tensorboard")
    tensorboard_output_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(tensorboard_output_path))


def log_config(config: Dict, summary_writer: Optional[SummaryWriter]) -> None:
    """Logs the config dict to tensorboard

    Parameters:
        config: Config dict that specifies the network, compression, hyperparams etc.
        summary_writer: Tensorboard writer
    """
    logging.info(json.dumps(config, indent=4, sort_keys=True))

    if summary_writer is not None:
        summary_writer.add_text("config", yaml.dump(config, default_flow_style=False).replace("\n", "  \n"))


def bits_to_mb(bits: int) -> float:
    """Convert from bits to mega bytes"""
    return bits / 8 / 1024 / 1024


def log_compression_ratio(
    uncompressed_model_size_bits: int, compressed_model_size_bits: int, summary_writer: Optional[SummaryWriter] = None
) -> None:
    """Compute stats about model compression and log them to both the standard logger and tensorboard

    Parameters:
        uncompressed_model_size_bits: The size of the uncompressed model in bits
        compressed_model_size_bits: The size of the compressed model in bits
        summary_writer: Tensorbard logger to writer compression params etc. Optional. Not passing it disables logging
    """
    model_size_log = "\n" + "\n".join(
        [
            f"uncompressed (bits): {uncompressed_model_size_bits}",
            f"compressed (bits):   {compressed_model_size_bits}",
            f"uncompressed (MB):   {bits_to_mb(uncompressed_model_size_bits):.2f}",
            f"compressed (MB):     {bits_to_mb(compressed_model_size_bits):.2f}",
            f"compression ratio:   {(uncompressed_model_size_bits / compressed_model_size_bits):.2f}",
        ]
    )
    logging.info(model_size_log)

    if summary_writer is not None:
        summary_writer.add_text("model", model_size_log)


def log_to_summary_writer(
    prefix: str, idx: int, state: Summary, summary_writer: Optional[SummaryWriter] = None
) -> None:
    """Write a summary to tensorboard

    Parameters:
        prefix: A string to be prepended to the fields to be logged
        idx: Index where the values will be indexed (typically, iteration number for tensorboard)
        state: A dictionary with numbers to log. This is a dictionary, and values must be scalars
        summary_writer: Tensorbard logger to writer compression params etc. Optional. If None, logging is disabled
    """
    if summary_writer is None:
        return

    for key, value in state.items():
        summary_writer.add_scalar(f"{prefix}/{key}", value, idx)
