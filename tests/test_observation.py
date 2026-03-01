"""Tests for gym_env/observation.py — card encoding and observation space."""

import numpy as np
import pytest

from gym_env.observation import (
    OBSERVATION_SIZE,
    TOTAL_POINTS,
    build_observation_space,
    encode_card,
    SUIT_INDEX,
    RANK_INDEX,
)
from gym_env.engine_adapter import Card


class TestEncodeCard:
    """Test encode_card produces correct card IDs."""

    def test_oros_1(self):
        card = Card(rank=1, suit="Oros", suit_symbol="🪙", display_name="1 de Oros", points=11)
        assert encode_card(card) == 0  # suit=0, rank_index=0

    def test_oros_12(self):
        card = Card(rank=12, suit="Oros", suit_symbol="🪙", display_name="12 de Oros", points=4)
        assert encode_card(card) == 9  # suit=0, rank_index=9

    def test_copas_3(self):
        card = Card(rank=3, suit="Copas", suit_symbol="🏆", display_name="3 de Copas", points=10)
        assert encode_card(card) == 12  # suit=1, rank_index=2

    def test_espadas_7(self):
        card = Card(rank=7, suit="Espadas", suit_symbol="⚔️", display_name="7 de Espadas", points=0)
        assert encode_card(card) == 26  # suit=2, rank_index=6

    def test_bastos_10(self):
        card = Card(rank=10, suit="Bastos", suit_symbol="🏑", display_name="10 de Bastos", points=2)
        assert encode_card(card) == 37  # suit=3, rank_index=7

    def test_bastos_12(self):
        card = Card(rank=12, suit="Bastos", suit_symbol="🏑", display_name="12 de Bastos", points=4)
        assert encode_card(card) == 39  # suit=3, rank_index=9 → max card ID


class TestSuitAndRankMappings:
    """Test suit and rank index mappings are complete and correct."""

    def test_suit_index_has_all_four_suits(self):
        assert set(SUIT_INDEX.keys()) == {"Oros", "Copas", "Espadas", "Bastos"}

    def test_suit_index_values(self):
        assert SUIT_INDEX["Oros"] == 0
        assert SUIT_INDEX["Copas"] == 1
        assert SUIT_INDEX["Espadas"] == 2
        assert SUIT_INDEX["Bastos"] == 3

    def test_rank_index_has_all_ten_ranks(self):
        assert set(RANK_INDEX.keys()) == {1, 2, 3, 4, 5, 6, 7, 10, 11, 12}

    def test_rank_index_values_are_contiguous(self):
        values = sorted(RANK_INDEX.values())
        assert values == list(range(10))


class TestObservationSpace:
    """Test observation_space bounds match expected low/high per element."""

    def test_shape(self):
        space = build_observation_space()
        assert space.shape == (13,)

    def test_dtype(self):
        space = build_observation_space()
        assert space.dtype == np.float32

    def test_low_bounds(self):
        space = build_observation_space()
        expected_low = np.array([-1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(space.low, expected_low)

    def test_high_bounds(self):
        space = build_observation_space()
        expected_high = np.array([39, 39, 39, 39, 3, 39, 39, 10, 10, 10, 10, 120, 120], dtype=np.float32)
        np.testing.assert_array_equal(space.high, expected_high)


class TestCardSeenPerSuitCounts:
    """Test cards_seen_per_suit counts derived correctly from card ID set."""

    def test_empty_set(self):
        cards_seen: set[int] = set()
        counts = [sum(1 for cid in cards_seen if cid // 10 == s) for s in range(4)]
        assert counts == [0, 0, 0, 0]

    def test_mixed_suits(self):
        # card IDs: 0 (Oros), 1 (Oros), 10 (Copas), 20 (Espadas), 30 (Bastos), 31 (Bastos)
        cards_seen = {0, 1, 10, 20, 30, 31}
        counts = [sum(1 for cid in cards_seen if cid // 10 == s) for s in range(4)]
        assert counts == [2, 1, 1, 2]

    def test_all_cards_of_one_suit(self):
        # All 10 Oros cards: IDs 0-9
        cards_seen = set(range(10))
        counts = [sum(1 for cid in cards_seen if cid // 10 == s) for s in range(4)]
        assert counts == [10, 0, 0, 0]


class TestConstants:
    """Test constants are correctly defined."""

    def test_observation_size(self):
        assert OBSERVATION_SIZE == 13

    def test_total_points(self):
        assert TOTAL_POINTS == 120
