"""Tests for gym_env/observation.py — card encoding and observation space."""

import numpy as np
import pytest

from gym_env.observation import (
    OBSERVATION_SIZE,
    TOTAL_POINTS,
    NUM_CARDS,
    BITMAP_START,
    BITMAP_END,
    TRUMP_ID,
    TRUMP_SUIT,
    TRICK_START,
    DECK_REMAINING,
    AGENT_SCORE,
    OPPONENT_SCORE,
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
    """Test observation_space bounds match expected layout."""

    def test_shape(self):
        space = build_observation_space()
        assert space.shape == (OBSERVATION_SIZE,)

    def test_observation_size_is_50(self):
        assert OBSERVATION_SIZE == 50

    def test_dtype(self):
        space = build_observation_space()
        assert space.dtype == np.float32

    def test_hand_bounds(self):
        space = build_observation_space()
        for i in range(3):
            assert space.low[i] == -1
            assert space.high[i] == 39

    def test_trump_bounds(self):
        space = build_observation_space()
        assert space.low[TRUMP_ID] == 0
        assert space.high[TRUMP_ID] == 39
        assert space.low[TRUMP_SUIT] == 0
        assert space.high[TRUMP_SUIT] == 3

    def test_trick_bounds(self):
        space = build_observation_space()
        for i in range(2):
            assert space.low[TRICK_START + i] == -1
            assert space.high[TRICK_START + i] == 39

    def test_bitmap_bounds(self):
        space = build_observation_space()
        for i in range(BITMAP_START, BITMAP_END):
            assert space.low[i] == 0
            assert space.high[i] == 1

    def test_deck_remaining_bounds(self):
        space = build_observation_space()
        assert space.low[DECK_REMAINING] == 0
        assert space.high[DECK_REMAINING] == 34

    def test_score_bounds(self):
        space = build_observation_space()
        assert space.low[AGENT_SCORE] == 0
        assert space.high[AGENT_SCORE] == 120
        assert space.low[OPPONENT_SCORE] == 0
        assert space.high[OPPONENT_SCORE] == 120


class TestConstants:
    """Test constants are correctly defined."""

    def test_num_cards(self):
        assert NUM_CARDS == 40

    def test_total_points(self):
        assert TOTAL_POINTS == 120

    def test_bitmap_span(self):
        assert BITMAP_END - BITMAP_START == NUM_CARDS
