from __future__ import annotations

import random
import socket
import traceback
from datetime import timedelta
from typing import Any, Callable, Literal, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore

def create_config():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    config.set_timestamp()
    config.save_config()
    return config

if __name__ == "__main__":
    create_config()
