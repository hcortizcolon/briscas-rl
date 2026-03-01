"""Tests for gym_env/briscas_env.py — BriscasEnv Gymnasium wrapper."""

from unittest.mock import MagicMock, call

import gymnasium
import numpy as np
import pytest

from gym_env.engine_adapter import Card, EngineAdapter, GameState, PlayerInfo, TrickCard
from gym_env.briscas_env import BriscasEnv
from gym_env.observation import encode_card


# --- Fixtures ---

def _card(rank: int, suit: str, points: int = 0) -> Card:
    """Helper to create a Card with minimal required fields."""
    symbols = {"Oros": "🪙", "Copas": "🏆", "Espadas": "⚔️", "Bastos": "🏑"}
    return Card(
        rank=rank,
        suit=suit,
        suit_symbol=symbols[suit],
        display_name=f"{rank} de {suit}",
        points=points,
    )


def _players(agent_score: int = 0, opponent_score: int = 0) -> list[PlayerInfo]:
    return [
        PlayerInfo(name="rl_agent", is_current=True, is_human=True, score=agent_score, hand_size=3),
        PlayerInfo(name="ai", is_current=False, is_human=False, score=opponent_score, hand_size=3),
    ]


def _state(
    hand: list[Card] | None = None,
    trump: Card | None = None,
    trick: list[TrickCard] | None = None,
    players: list[PlayerInfo] | None = None,
    game_over: bool = False,
    winner: str | None = None,
    is_your_turn: bool = True,
    deck_remaining: int = 30,
    round_number: int = 1,
) -> GameState:
    if hand is None:
        hand = [_card(1, "Oros", 11), _card(3, "Copas", 10), _card(7, "Espadas")]
    if trump is None:
        trump = _card(5, "Bastos")
    if trick is None:
        trick = []
    if players is None:
        players = _players()
    return GameState(
        hand=hand,
        trump=trump,
        trick=trick,
        players=players,
        deck_remaining=deck_remaining,
        round_number=round_number,
        game_over=game_over,
        winner=winner,
        is_your_turn=is_your_turn,
    )


def _make_env(adapter: EngineAdapter | None = None) -> BriscasEnv:
    if adapter is None:
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
    return BriscasEnv(adapter=adapter)


# --- Test Classes ---

class TestObservationShape:
    """Test observation shape is (13,) and dtype is float32."""

    def test_reset_observation_shape(self):
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == (13,)

    def test_reset_observation_dtype(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs.dtype == np.float32


class TestObservationValues:
    """Test observation values match expected encoding for known game state."""

    def test_known_state_encoding(self):
        hand = [_card(1, "Oros", 11), _card(3, "Copas", 10), _card(7, "Espadas")]
        trump = _card(5, "Bastos")
        state = _state(hand=hand, trump=trump, players=_players(agent_score=15, opponent_score=22))

        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        # h0=encode(1 Oros)=0, h1=encode(3 Copas)=12, h2=encode(7 Espadas)=26
        # trump=encode(5 Bastos)=34, trump_suit=3
        # trick empty: -1, -1
        # cards_seen: all 0
        # agent_points=15, opp_points=22
        expected = np.array([0, 12, 26, 34, 3, -1, -1, 0, 0, 0, 0, 15, 22], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)


class TestHandSorting:
    """Test hand sorting produces consistent encoding regardless of input order."""

    def test_unsorted_hand_gets_sorted(self):
        # Provide hand in reverse order of card IDs
        hand = [_card(7, "Espadas"), _card(1, "Oros", 11), _card(3, "Copas", 10)]
        state = _state(hand=hand)
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        # Sorted by card ID: Oros1=0, Copas3=12, Espadas7=26
        assert obs[0] == 0.0
        assert obs[1] == 12.0
        assert obs[2] == 26.0


class TestHandPadding:
    """Test hand padding with -1 for hands smaller than 3."""

    def test_hand_size_2(self):
        hand = [_card(1, "Oros", 11), _card(3, "Copas", 10)]
        state = _state(hand=hand)
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[0] == 0.0   # Oros 1
        assert obs[1] == 12.0  # Copas 3
        assert obs[2] == -1.0  # padded

    def test_hand_size_1(self):
        hand = [_card(1, "Oros", 11)]
        state = _state(hand=hand)
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[0] == 0.0
        assert obs[1] == -1.0
        assert obs[2] == -1.0

    def test_hand_size_0(self):
        state = _state(hand=[])
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[0] == -1.0
        assert obs[1] == -1.0
        assert obs[2] == -1.0


class TestTrickSlots:
    """Test trick slot ordering: slot 0 = lead card, slot 1 = response card, -1 when empty."""

    def test_empty_trick(self):
        state = _state(trick=[])
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[5] == -1.0
        assert obs[6] == -1.0

    def test_one_card_in_trick(self):
        trick = [TrickCard(player="ai", card=_card(10, "Bastos", 2))]
        state = _state(trick=trick)
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[5] == float(encode_card(_card(10, "Bastos", 2)))  # 37
        assert obs[6] == -1.0

    def test_two_cards_in_trick(self):
        trick = [
            TrickCard(player="rl_agent", card=_card(1, "Oros", 11)),
            TrickCard(player="ai", card=_card(12, "Oros", 4)),
        ]
        state = _state(trick=trick)
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = state
        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()

        assert obs[5] == float(encode_card(_card(1, "Oros", 11)))   # 0
        assert obs[6] == float(encode_card(_card(12, "Oros", 4)))   # 9


class TestActionMasking:
    """Test action masking: action % len(hand) for hand sizes 1, 2, 3."""

    def test_hand_size_3_action_0(self):
        adapter = MagicMock(spec=EngineAdapter)
        initial = _state()
        after_play = _state(is_your_turn=True)
        adapter.new_game.return_value = initial
        adapter.play_card.return_value = after_play
        env = BriscasEnv(adapter=adapter)
        env.reset()
        env.step(0)
        adapter.play_card.assert_called_with(0)

    def test_hand_size_3_action_2(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        adapter.play_card.return_value = _state(is_your_turn=True)
        env = BriscasEnv(adapter=adapter)
        env.reset()
        env.step(2)
        adapter.play_card.assert_called_with(2)

    def test_hand_size_2_action_2_wraps(self):
        adapter = MagicMock(spec=EngineAdapter)
        hand = [_card(1, "Oros", 11), _card(3, "Copas", 10)]
        adapter.new_game.return_value = _state(hand=hand)
        adapter.play_card.return_value = _state(hand=[_card(1, "Oros", 11)], is_your_turn=True)
        env = BriscasEnv(adapter=adapter)
        env.reset()
        env.step(2)  # 2 % 2 = 0
        adapter.play_card.assert_called_with(0)

    def test_hand_size_1_action_1_wraps(self):
        adapter = MagicMock(spec=EngineAdapter)
        hand = [_card(1, "Oros", 11)]
        adapter.new_game.return_value = _state(hand=hand)
        adapter.play_card.return_value = _state(hand=[], game_over=True, is_your_turn=False)
        env = BriscasEnv(adapter=adapter)
        env.reset()
        env.step(1)  # 1 % 1 = 0
        adapter.play_card.assert_called_with(0)


class TestRewardIntermediate:
    """Test reward is 0.0 for intermediate steps."""

    def test_intermediate_reward_zero(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        adapter.play_card.return_value = _state(game_over=False, is_your_turn=True)
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0


class TestRewardTerminal:
    """Test reward is normalized point differential at game end."""

    def test_win_reward(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        terminal = _state(
            hand=[],
            game_over=True,
            winner="rl_agent",
            players=_players(agent_score=80, opponent_score=40),
            is_your_turn=False,
        )
        adapter.play_card.return_value = terminal
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, reward, terminated, truncated, _ = env.step(0)
        assert reward == pytest.approx((80 - 40) / 120)
        assert terminated is True
        assert truncated is False

    def test_loss_reward(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        terminal = _state(
            hand=[],
            game_over=True,
            winner="ai",
            players=_players(agent_score=30, opponent_score=90),
            is_your_turn=False,
        )
        adapter.play_card.return_value = terminal
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx((30 - 90) / 120)

    def test_draw_reward(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        terminal = _state(
            hand=[],
            game_over=True,
            winner=None,
            players=_players(agent_score=60, opponent_score=60),
            is_your_turn=False,
        )
        adapter.play_card.return_value = terminal
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0)


class TestTerminatedTruncated:
    """Test terminated=True at game over, truncated=False always."""

    def test_not_terminated_mid_game(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        adapter.play_card.return_value = _state(game_over=False, is_your_turn=True)
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is False
        assert truncated is False

    def test_terminated_at_game_over(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        adapter.play_card.return_value = _state(
            hand=[], game_over=True, players=_players(60, 60), is_your_turn=False
        )
        env = BriscasEnv(adapter=adapter)
        env.reset()
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is True
        assert truncated is False


class TestResetClears:
    """Test reset() clears cards_seen and returns fresh observation."""

    def test_reset_clears_state(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        env = BriscasEnv(adapter=adapter)

        # First game
        env.reset()
        # Play a step that adds to cards_seen
        trick = [
            TrickCard(player="rl_agent", card=_card(1, "Oros", 11)),
            TrickCard(player="ai", card=_card(12, "Oros", 4)),
        ]
        adapter.play_card.return_value = _state(trick=trick, is_your_turn=True)
        env.step(0)

        # Reset should clear cards_seen
        adapter.new_game.return_value = _state()
        obs, _ = env.reset()
        # After reset, cards_seen should be empty so suit counts are all 0
        assert obs[7] == 0.0  # s0
        assert obs[8] == 0.0  # s1
        assert obs[9] == 0.0  # s2
        assert obs[10] == 0.0  # s3


class TestResetReturnType:
    """Test reset() returns (observation, info) tuple."""

    def test_reset_returns_tuple(self):
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_info_is_dict(self):
        env = _make_env()
        _, info = env.reset()
        assert isinstance(info, dict)


class TestResetDeletesActiveGame:
    """Test reset() calls delete_game() when a game is already active."""

    def test_second_reset_calls_delete(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        env = BriscasEnv(adapter=adapter)

        env.reset()
        adapter.delete_game.assert_not_called()

        env.reset()
        adapter.delete_game.assert_called_once()

    def test_first_reset_does_not_call_delete(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()
        env = BriscasEnv(adapter=adapter)

        env.reset()
        adapter.delete_game.assert_not_called()


class TestSpaces:
    """Test env.observation_space contains observation from reset(), env.action_space is Discrete(3)."""

    def test_observation_in_space(self):
        env = _make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_space_is_discrete_3(self):
        env = _make_env()
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 3


class TestFullGameLoop:
    """Test full game loop: mock multi-turn game, verify observations update each step."""

    def test_full_game(self):
        adapter = MagicMock(spec=EngineAdapter)

        hand_r1 = [_card(1, "Oros", 11), _card(3, "Copas", 10), _card(7, "Espadas")]
        trump = _card(5, "Bastos")
        initial = _state(hand=hand_r1, trump=trump, players=_players(0, 0))
        adapter.new_game.return_value = initial

        # Turn 1: agent plays card 0, opponent responds, trick resolves
        after_t1 = _state(
            hand=[_card(3, "Copas", 10), _card(7, "Espadas")],
            trump=trump,
            trick=[
                TrickCard(player="rl_agent", card=_card(1, "Oros", 11)),
                TrickCard(player="ai", card=_card(2, "Oros")),
            ],
            players=_players(11, 0),
            is_your_turn=True,
            game_over=False,
        )
        # Turn 2: agent plays, game ends
        after_t2 = _state(
            hand=[],
            trump=trump,
            trick=[],
            players=_players(70, 50),
            is_your_turn=False,
            game_over=True,
            winner="rl_agent",
        )

        adapter.play_card.side_effect = [after_t1, after_t2]

        env = BriscasEnv(adapter=adapter)
        obs, _ = env.reset()
        assert obs.shape == (13,)

        # Step 1
        obs1, r1, term1, trunc1, _ = env.step(0)
        assert r1 == 0.0
        assert term1 is False
        assert trunc1 is False
        assert obs1.shape == (13,)
        # Observation should differ from initial (different hand, trick visible, score changed)
        assert not np.array_equal(obs, obs1)

        # Step 2 — terminal
        obs2, r2, term2, trunc2, _ = env.step(0)
        assert r2 == pytest.approx((70 - 50) / 120)
        assert term2 is True
        assert trunc2 is False


class TestOpponentLeadsAfterWinningTrick:
    """Test step() handles opponent leading after winning trick (process_ai_turn loop)."""

    def test_process_ai_turn_called_when_not_our_turn(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()

        # After play_card, it's NOT our turn (opponent won trick and must lead)
        after_play = _state(is_your_turn=False, game_over=False)
        # After process_ai_turn, it IS our turn
        after_ai = _state(is_your_turn=True, game_over=False)

        adapter.play_card.return_value = after_play
        adapter.process_ai_turn.return_value = after_ai

        env = BriscasEnv(adapter=adapter)
        env.reset()
        env.step(0)

        adapter.process_ai_turn.assert_called_once()

    def test_cards_seen_captured_from_intermediate_state(self):
        """Cards from a resolved trick must be tracked even when opponent wins and leads next."""
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state()

        # play_card returns state with trick cards visible (before resolution)
        trick = [
            TrickCard(player="rl_agent", card=_card(1, "Oros", 11)),
            TrickCard(player="ai", card=_card(12, "Oros", 4)),
        ]
        after_play = _state(trick=trick, is_your_turn=False, game_over=False)

        # process_ai_turn resolves trick, opponent leads — trick is now cleared/new
        after_ai = _state(trick=[], is_your_turn=True, game_over=False)

        adapter.play_card.return_value = after_play
        adapter.process_ai_turn.return_value = after_ai

        env = BriscasEnv(adapter=adapter)
        env.reset()
        obs, _, _, _, _ = env.step(0)

        # Cards from the resolved trick must appear in cards_seen counts
        # Oros 1 (card ID 0) and Oros 12 (card ID 9) → 2 Oros cards seen
        assert obs[7] == 2.0   # s0 = Oros count
        assert obs[8] == 0.0   # s1 = Copas count
        assert obs[9] == 0.0   # s2 = Espadas count
        assert obs[10] == 0.0  # s3 = Bastos count


class TestStepGuards:
    """Test step() raises on invalid calls."""

    def test_step_on_empty_hand_raises(self):
        adapter = MagicMock(spec=EngineAdapter)
        adapter.new_game.return_value = _state(hand=[])
        env = BriscasEnv(adapter=adapter)
        env.reset()
        with pytest.raises(ValueError, match="empty hand"):
            env.step(0)
