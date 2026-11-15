"""Global reproducibility helpers.

A single seed (42 by default) is applied to Python, NumPy, PyTorch and the
process hash seed. Every generator accepts an explicit seed as well; this module
exists so stochastic library code outside our generators is also pinned.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed all RNGs used by the project and return the seed.

    Args:
        seed: Integer seed. The default is the project-wide seed 42.

    Returns:
        The seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time in some environments
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - exercised only without torch
        pass
    return seed


def seeded_rng(seed: int) -> random.Random:
    """Return an independent ``random.Random`` instance for a generator."""
    return random.Random(seed)


def seeded_np_rng(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator (PCG64) for a generator."""
    return np.random.default_rng(seed)
