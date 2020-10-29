# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import os
from typing import Dict, Optional

import yaml


def parse_overridden_arg(string: str):
    """Given an overridden arg from command line, parse into int, float, string etc."""
    if string.lower() == "true":
        return True

    if string.lower() == "false":
        return False

    try:
        return ast.literal_eval(string)
    except SyntaxError:
        return string
    except ValueError:
        return string


def load_config(calling_path: str, default_config_path: Optional[str] = None) -> Dict:
    """Loads arguments from a config file, but allows them to be overridden on the command line for rapid testing

    Parameters:
        calling_path:        The path of the file that is calling this function. If a config is loaded from the
                             command line, the path should be relative to the caller.
        default_config_path: The default config file path
    Returns:
        config: The corresponding file, parsed
    """

    def set_value(layered_keys, value, initial_dict):
        initkey = layered_keys[0]
        if len(layered_keys) == 1:
            initial_dict[initkey] = value
        else:
            if initkey in initial_dict:
                next_dict = initial_dict[initkey]
                set_value(layered_keys[1:], value, next_dict)
            else:
                next_dict = dict()
                initial_dict[initkey] = next_dict
                set_value(layered_keys[1:], value, next_dict)

    # This will load "--config" from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config_path)
    args, unknown_args = parser.parse_known_args()

    with open(os.path.join(calling_path, args.config)) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # This parses everything other than "--config" from the command line
    unknown_arg_parser = argparse.ArgumentParser()

    for unknown_arg in unknown_args:
        if "--" in unknown_arg:
            unknown_arg_parser.add_argument(unknown_arg)

    overridden_args = vars(unknown_arg_parser.parse_args(unknown_args))

    for key in overridden_args.keys():
        new_value = parse_overridden_arg(overridden_args[key])

        layered_keys = key.split(".")
        set_value(layered_keys, new_value, config)

    return config
