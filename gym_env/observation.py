"""Card encoding constants and observation space definition for Briscas environment."""

import gymnasium
import numpy as np

from gym_env.engine_adapter import Card, TrickCard

NUM_CARDS = 40
TOTAL_POINTS = 120

# Observation layout (50 features):
#   [0-2]   Hand card IDs sorted, padded with -1
#   [3]     Trump card ID
#   [4]     Trump suit index (0-3)
#   [5-6]   Trick card IDs (-1 if empty)
#   [7-46]  Cards-played bitmap (40 binary values, 1 = seen in a previous trick)
#   [47]    Deck remaining (0-34)
#   [48]    Agent score (0-120)
#   [49]    Opponent score (0-120)
OBSERVATION_SIZE = 50

# Slice constants for readability
HAND_START = 0
TRUMP_ID = 3
TRUMP_SUIT = 4
TRICK_START = 5
BITMAP_START = 7
BITMAP_END = BITMAP_START + NUM_CARDS  # 47
DECK_REMAINING = 47
AGENT_SCORE = 48
OPPONENT_SCORE = 49

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
    low = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    high = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # Hand cards: -1 (empty) to 39
    low[HAND_START:HAND_START + 3] = -1
    high[HAND_START:HAND_START + 3] = NUM_CARDS - 1

    # Trump card ID and suit
    high[TRUMP_ID] = NUM_CARDS - 1
    high[TRUMP_SUIT] = 3

    # Trick cards: -1 (empty) to 39
    low[TRICK_START:TRICK_START + 2] = -1
    high[TRICK_START:TRICK_START + 2] = NUM_CARDS - 1

    # Cards-played bitmap: 0 or 1
    high[BITMAP_START:BITMAP_END] = 1

    # Deck remaining
    high[DECK_REMAINING] = 34

    # Scores
    high[AGENT_SCORE] = TOTAL_POINTS
    high[OPPONENT_SCORE] = TOTAL_POINTS

    return gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)


def build_observation(
    hand: list[Card],
    trump: Card,
    trick: list[TrickCard],
    cards_seen: set[int],
    deck_remaining: int,
    agent_score: int,
    opponent_score: int,
) -> np.ndarray:
    """Build the 50-feature observation vector from raw game data."""
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    obs[0:3] = -1.0
    hand_ids = sorted(encode_card(c) for c in hand)
    for i, cid in enumerate(hand_ids):
        obs[i] = cid

    obs[TRUMP_ID] = encode_card(trump)
    obs[TRUMP_SUIT] = SUIT_INDEX[trump.suit]

    obs[TRICK_START:TRICK_START + 2] = -1.0
    for i, tc in enumerate(trick):
        obs[TRICK_START + i] = encode_card(tc.card)

    for cid in cards_seen:
        obs[BITMAP_START + cid] = 1.0

    obs[DECK_REMAINING] = deck_remaining
    obs[AGENT_SCORE] = agent_score
    obs[OPPONENT_SCORE] = opponent_score

    return obs


def sorted_hand_index(hand: list[Card], sorted_idx: int) -> int:
    """Map a sorted-hand index back to the engine's hand index."""
    order = sorted(range(len(hand)), key=lambda i: encode_card(hand[i]))
    return order[sorted_idx]
