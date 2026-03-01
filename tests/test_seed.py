"""Tests for seed.py — reproducible seeding across random sources."""

import logging
import random

import numpy as np
import torch

from seed import set_all_seeds


class TestSetAllSeeds:
    """Tests for set_all_seeds function."""

    def test_numpy_reproducibility_with_same_seed(self):
        """Same seed produces identical numpy random values."""
        set_all_seeds(42)
        val1 = np.random.random()
        set_all_seeds(42)
        val2 = np.random.random()
        assert val1 == val2

    def test_torch_reproducibility_with_same_seed(self):
        """Same seed produces identical torch random values."""
        set_all_seeds(42)
        val1 = torch.rand(1).item()
        set_all_seeds(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_stdlib_random_reproducibility_with_same_seed(self):
        """Same seed produces identical stdlib random values."""
        set_all_seeds(42)
        val1 = random.random()
        set_all_seeds(42)
        val2 = random.random()
        assert val1 == val2

    def test_different_seeds_produce_different_numpy_values(self):
        """Different seeds produce different numpy random values."""
        set_all_seeds(42)
        val1 = np.random.random()
        set_all_seeds(99)
        val2 = np.random.random()
        assert val1 != val2

    def test_seed_zero_works(self):
        """Edge case: seed=0 works correctly."""
        set_all_seeds(0)
        val1 = np.random.random()
        set_all_seeds(0)
        val2 = np.random.random()
        assert val1 == val2

    def test_logging_includes_seed_value(self, caplog):
        """Logging output includes the seed value."""
        with caplog.at_level(logging.INFO, logger="seed"):
            set_all_seeds(42)
        assert "42" in caplog.text
