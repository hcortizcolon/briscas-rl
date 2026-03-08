"""In-process Briscas game engine and LocalAdapter for training without HTTP overhead."""

import random
from enum import Enum

from gym_env.engine_adapter import (
    Card as AdapterCard,
    EngineAdapter,
    GameState,
    PlayerInfo,
    TrickCard,
)


# ---------------------------------------------------------------------------
# Standalone Briscas game engine
# ---------------------------------------------------------------------------

class Suit(Enum):
    OROS = "Oros"
    COPAS = "Copas"
    ESPADAS = "Espadas"
    BASTOS = "Bastos"


VALID_RANKS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

POINT_VALUES = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 10: 2, 11: 3, 12: 4}

RANK_STRENGTH = {1: 8, 3: 7, 12: 6, 11: 5, 10: 4, 7: 3, 6: 2, 5: 1, 4: 0, 2: -1}

SUIT_SYMBOLS = {
    Suit.OROS: "\U0001fa99",
    Suit.COPAS: "\U0001f377",
    Suit.ESPADAS: "\u2694\ufe0f",
    Suit.BASTOS: "\U0001f3cf",
}

RANK_NAMES = {
    1: "As", 2: "Dos", 3: "Tres", 4: "Cuatro", 5: "Cinco",
    6: "Seis", 7: "Siete", 10: "Sota", 11: "Caballo", 12: "Rey",
}


class Card:
    __slots__ = ("rank", "suit")

    def __init__(self, rank: int, suit: Suit) -> None:
        self.rank = rank
        self.suit = suit

    @property
    def points(self) -> int:
        return POINT_VALUES[self.rank]

    @property
    def display_name(self) -> str:
        return f"{RANK_NAMES[self.rank]} de {self.suit.value}"

    @property
    def suit_symbol(self) -> str:
        return SUIT_SYMBOLS[self.suit]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


def _compare_cards(card1: Card, card2: Card, trump_suit: Suit, led_suit: Suit) -> int:
    """Return 1 if card1 wins, 2 if card2 wins."""
    c1_trump = card1.suit == trump_suit
    c2_trump = card2.suit == trump_suit

    if c1_trump and not c2_trump:
        return 1
    if c2_trump and not c1_trump:
        return 2
    if c1_trump and c2_trump:
        return 1 if RANK_STRENGTH[card1.rank] > RANK_STRENGTH[card2.rank] else 2

    c1_follows = card1.suit == led_suit
    c2_follows = card2.suit == led_suit

    if not c1_follows and c2_follows:
        return 2
    if c1_follows and not c2_follows:
        return 1
    if c1_follows and c2_follows:
        return 1 if RANK_STRENGTH[card1.rank] > RANK_STRENGTH[card2.rank] else 2
    return 1


def _trick_winner(trick: list[tuple[int, Card]], trump_suit: Suit) -> tuple[int, Card]:
    """Return (winner_player_idx, winning_card)."""
    winner_idx = 0
    led_suit = trick[0][1].suit
    for i in range(1, len(trick)):
        if _compare_cards(trick[winner_idx][1], trick[i][1], trump_suit, led_suit) == 2:
            winner_idx = i
    return trick[winner_idx]


# ---------------------------------------------------------------------------
# AI strategy (reimplements AdvancedStrategy from lets-play-brisca)
# ---------------------------------------------------------------------------

def _ai_choose_card_index(
    hand: list[Card],
    trick: list[tuple[int, Card]],
    trump_suit: Suit,
    deck_remaining: int,
) -> int:
    """Return index in *hand* of the card the AI should play."""
    if not trick:
        return _ai_lead(hand, trump_suit, deck_remaining)
    return _ai_follow(hand, trick, trump_suit)


def _ai_lead(hand: list[Card], trump_suit: Suit, deck_remaining: int) -> int:
    if deck_remaining <= 5:
        non_trump_pts = [i for i, c in enumerate(hand) if c.points > 0 and c.suit != trump_suit]
        if non_trump_pts:
            return max(non_trump_pts, key=lambda i: hand[i].points)

    zero_pts = [i for i, c in enumerate(hand) if c.points == 0]
    if zero_pts:
        non_trump_zero = [i for i in zero_pts if hand[i].suit != trump_suit]
        if non_trump_zero:
            return non_trump_zero[0]
        return zero_pts[0]

    return min(range(len(hand)), key=lambda i: hand[i].points)


def _ai_follow(hand: list[Card], trick: list[tuple[int, Card]], trump_suit: Suit) -> int:
    led_suit = trick[0][1].suit
    winning_card = trick[0][1]
    for i in range(1, len(trick)):
        if _compare_cards(winning_card, trick[i][1], trump_suit, led_suit) == 2:
            winning_card = trick[i][1]

    trick_points = sum(c.points for _, c in trick)

    winning_indices = [
        i for i, c in enumerate(hand)
        if _compare_cards(winning_card, c, trump_suit, led_suit) == 2
    ]

    # In a 2-player game the follower is always the last player.
    if trick_points >= 10 and winning_indices:
        # Last player — use weakest winning card
        return min(winning_indices, key=lambda i: (hand[i].points, hand[i].rank))

    zero_pts = [i for i, c in enumerate(hand) if c.points == 0]
    if zero_pts:
        return zero_pts[0]

    return min(range(len(hand)), key=lambda i: hand[i].points)


# ---------------------------------------------------------------------------
# Game object
# ---------------------------------------------------------------------------

class BriscasGame:
    """Two-player Briscas game (player 0 = RL agent, player 1 = AI)."""

    def __init__(self) -> None:
        # Build and shuffle deck
        cards = [Card(rank, suit) for suit in Suit for rank in VALID_RANKS]
        random.shuffle(cards)
        self.deck: list[Card] = cards

        # Deal 3 cards each
        self.hands: list[list[Card]] = [[], []]
        for _ in range(3):
            for p in range(2):
                self.hands[p].append(self.deck.pop())

        # Trump = bottom card of deck
        self.trump_card: Card = self.deck[0]

        self.won_cards: list[list[Card]] = [[], []]
        self.current_trick: list[tuple[int, Card]] = []
        self.current_player: int = random.randint(0, 1)
        self.round_number: int = 1
        self.game_over: bool = False

    # -- low-level operations used by the adapter --

    def remove_and_play(self, player_idx: int, card_index: int) -> None:
        card = self.hands[player_idx].pop(card_index)
        self.current_trick.append((player_idx, card))

    def is_trick_complete(self) -> bool:
        return len(self.current_trick) == 2

    def resolve_trick(self) -> None:
        winner, _ = _trick_winner(self.current_trick, self.trump_card.suit)
        for _, card in self.current_trick:
            self.won_cards[winner].append(card)
        self._replenish(winner)
        self.current_player = winner
        self.current_trick = []
        if all(len(h) == 0 for h in self.hands):
            self.game_over = True
        else:
            self.round_number += 1

    def ai_choose_card_index(self) -> int:
        return _ai_choose_card_index(
            self.hands[1],
            self.current_trick,
            self.trump_card.suit,
            len(self.deck),
        )

    def score(self, player_idx: int) -> int:
        return sum(c.points for c in self.won_cards[player_idx])

    def winner_name(self) -> str | None:
        if not self.game_over:
            return None
        s0, s1 = self.score(0), self.score(1)
        if s0 > s1:
            return "rl_agent"
        if s1 > s0:
            return "Claude"
        return None

    def _replenish(self, winner: int) -> None:
        if not self.deck:
            return
        for p in [winner, 1 - winner]:
            if self.deck:
                self.hands[p].append(self.deck.pop())


# ---------------------------------------------------------------------------
# LocalAdapter — EngineAdapter backed by in-process BriscasGame
# ---------------------------------------------------------------------------

class LocalAdapter(EngineAdapter):
    """Runs the Briscas game engine in-process (no HTTP)."""

    def __init__(self) -> None:
        self._game: BriscasGame | None = None

    def new_game(self) -> GameState:
        self._game = BriscasGame()
        return self._serialize()

    def play_card(self, card_index: int) -> GameState:
        g = self._game
        g.remove_and_play(0, card_index)
        if not g.is_trick_complete():
            g.current_player = 1
        state = self._serialize()
        if g.is_trick_complete():
            g.resolve_trick()
        return state

    def process_ai_turn(self) -> GameState:
        g = self._game
        if g.current_player == 0:
            return self._serialize()
        card_idx = g.ai_choose_card_index()
        g.remove_and_play(1, card_idx)
        if not g.is_trick_complete():
            g.current_player = 0
        state = self._serialize()
        if g.is_trick_complete():
            g.resolve_trick()
        return state

    def get_state(self) -> GameState:
        return self._serialize()

    def delete_game(self) -> None:
        self._game = None

    # -- serialization helpers --

    def _serialize(self) -> GameState:
        g = self._game
        return GameState(
            hand=[self._to_adapter_card(c) for c in g.hands[0]],
            trump=self._to_adapter_card(g.trump_card),
            trick=[
                TrickCard(
                    player="rl_agent" if pi == 0 else "Claude",
                    card=self._to_adapter_card(c),
                )
                for pi, c in g.current_trick
            ],
            players=[
                PlayerInfo(
                    name="rl_agent",
                    is_current=g.current_player == 0,
                    is_human=True,
                    score=g.score(0),
                    hand_size=len(g.hands[0]),
                ),
                PlayerInfo(
                    name="Claude",
                    is_current=g.current_player == 1,
                    is_human=False,
                    score=g.score(1),
                    hand_size=len(g.hands[1]),
                ),
            ],
            deck_remaining=len(g.deck),
            round_number=g.round_number,
            game_over=g.game_over,
            winner=g.winner_name(),
            is_your_turn=g.current_player == 0,
        )

    @staticmethod
    def _to_adapter_card(c: Card) -> AdapterCard:
        return AdapterCard(
            rank=c.rank,
            suit=c.suit.value,
            suit_symbol=c.suit_symbol,
            display_name=c.display_name,
            points=c.points,
        )
