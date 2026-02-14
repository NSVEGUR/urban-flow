"""
UrbanFlow – Utilities
======================
Reproducibility, timing, and logging helpers.
"""

import os
import random
import time
import logging
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(label: str = "Block"):
    """Context manager that prints elapsed time for a code block.

    Usage::

        with timer("Training GRU"):
            train_model(...)
    """
    start = time.perf_counter()
    logging.info(f"⏱  [{label}] started …")
    yield
    elapsed = time.perf_counter() - start
    mins, secs = divmod(elapsed, 60)
    logging.info(f"⏱  [{label}] finished in {int(mins)}m {secs:.1f}s")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
