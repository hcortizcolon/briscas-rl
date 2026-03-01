"""Tests for seed.py — reproducible seeding across random sources."""

import logging
import random
from unittest.mock import patch

import numpy as np
import pytest
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

    def test_torch_multi_element_reproducibility(self):
        """Same seed produces identical multi-element torch tensors."""
        set_all_seeds(42)
        val1 = torch.rand(5).tolist()
        set_all_seeds(42)
        val2 = torch.rand(5).tolist()
        assert val1 == val2

    def test_logging_includes_seed_value(self, caplog):
        """Logging output includes the seed value."""
        with caplog.at_level(logging.INFO, logger="seed"):
            set_all_seeds(42)
        assert "42" in caplog.text

    def test_cuda_branch_when_available(self):
        """CUDA seeding is called when GPU is available."""
        with patch("seed.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            set_all_seeds(42)
            mock_torch.manual_seed.assert_called_once_with(42)
            mock_torch.cuda.manual_seed_all.assert_called_once_with(42)
            assert mock_torch.backends.cudnn.deterministic is True
            assert mock_torch.backends.cudnn.benchmark is False

    def test_cuda_branch_when_not_available(self):
        """CUDA seeding is skipped when no GPU."""
        with patch("seed.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            set_all_seeds(42)
            mock_torch.manual_seed.assert_called_once_with(42)
            mock_torch.cuda.manual_seed_all.assert_not_called()

    def test_negative_seed_raises_value_error(self):
        """Negative seeds are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            set_all_seeds(-1)

    def test_seed_too_large_raises_value_error(self):
        """Seeds >= 2**32 are rejected."""
        with pytest.raises(ValueError, match="must be in range"):
            set_all_seeds(2**32)

    def test_non_int_seed_raises_type_error(self):
        """Non-integer seeds are rejected."""
        with pytest.raises(TypeError, match="must be an int"):
            set_all_seeds(42.0)
