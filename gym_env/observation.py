"""Card encoding constants and observation space definition for Briscas environment."""

import gymnasium
import numpy as np

from gym_env.engine_adapter import Card

OBSERVATION_SIZE = 13
TOTAL_POINTS = 120

SUIT_INDEX: dict[str, int] = {
    "Oros": 0,
    "Copas": 1,
    "Espadas": 2,
    "Bastos": 3,
}

RANK_INDEX: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    10: 7,
    11: 8,
    12: 9,
}


def encode_card(card: Card) -> int:
    """Encode a Card to a contiguous integer ID in range 0-39."""
    return SUIT_INDEX[card.suit] * 10 + RANK_INDEX[card.rank]


def build_observation_space() -> gymnasium.spaces.Box:
    """Build the Gymnasium observation space with per-element bounds."""
    low = np.array([-1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    high = np.array([39, 39, 39, 39, 3, 39, 39, 10, 10, 10, 10, 120, 120], dtype=np.float32)
    return gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
