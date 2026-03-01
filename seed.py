"""Reproducible seeding across all random sources."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Engine seeding: The game engine (lets-play-brisca) uses random.shuffle()
# internally with no external seed/shuffle API endpoint. Card deal randomness
# is NOT controllable from the RL training side. Reproducibility applies to
# policy initialization, exploration noise, and batch sampling only.
#
# PYTHONHASHSEED: Python dict/set ordering depends on PYTHONHASHSEED, which
# must be set as an environment variable before the interpreter starts.
# For fully deterministic runs, callers should set PYTHONHASHSEED=0.


def set_all_seeds(seed: int) -> None:
    """Seed all random sources for reproducible training and evaluation."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"seed must be in range [0, 2**32 - 1], got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("All seeds set to %d", seed)
