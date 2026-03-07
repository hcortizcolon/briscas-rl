"""Tests for gym_env/local_adapter.py — LocalAdapter and in-process game engine."""

import pytest

from gym_env.engine_adapter import Card, EngineAdapter, GameState, PlayerInfo, TrickCard
from gym_env.local_adapter import (
    BriscasGame,
    Card as EngineCard,
    LocalAdapter,
    Suit,
    _compare_cards,
    _trick_winner,
)
from gym_env.briscas_env import BriscasEnv
from gym_env.observation import OBSERVATION_SIZE, encode_card


# ---------------------------------------------------------------------------
# Card comparison tests
# ---------------------------------------------------------------------------


class TestCompareCards:
    def test_trump_beats_non_trump(self):
        trump = EngineCard(2, Suit.OROS)
        non_trump = EngineCard(1, Suit.COPAS)  # Ace, but not trump
        assert _compare_cards(trump, non_trump, Suit.OROS, Suit.COPAS) == 1
        assert _compare_cards(non_trump, trump, Suit.OROS, Suit.COPAS) == 2

    def test_higher_trump_beats_lower_trump(self):
        ace = EngineCard(1, Suit.OROS)
        two = EngineCard(2, Suit.OROS)
        assert _compare_cards(ace, two, Suit.OROS, Suit.COPAS) == 1
        assert _compare_cards(two, ace, Suit.OROS, Suit.COPAS) == 2

    def test_led_suit_beats_off_suit(self):
        led = EngineCard(2, Suit.COPAS)
        off = EngineCard(1, Suit.ESPADAS)
        assert _compare_cards(led, off, Suit.OROS, Suit.COPAS) == 1
        assert _compare_cards(off, led, Suit.OROS, Suit.COPAS) == 2

    def test_first_card_wins_when_neither_follows(self):
        c1 = EngineCard(2, Suit.ESPADAS)
        c2 = EngineCard(1, Suit.BASTOS)
        assert _compare_cards(c1, c2, Suit.OROS, Suit.COPAS) == 1


class TestTrickWinner:
    def test_two_card_trick(self):
        trick = [(0, EngineCard(1, Suit.OROS)), (1, EngineCard(3, Suit.OROS))]
        winner_idx, winner_card = _trick_winner(trick, Suit.COPAS)
        assert winner_idx == 0  # Ace beats 3 in Brisca strength

    def test_trump_wins(self):
        trick = [(0, EngineCard(12, Suit.COPAS)), (1, EngineCard(2, Suit.OROS))]
        winner_idx, _ = _trick_winner(trick, Suit.OROS)  # Oros is trump
        assert winner_idx == 1  # Trump 2 beats non-trump king


# ---------------------------------------------------------------------------
# BriscasGame tests
# ---------------------------------------------------------------------------


class TestBriscasGame:
    def test_initial_state(self):
        g = BriscasGame()
        assert len(g.hands[0]) == 3
        assert len(g.hands[1]) == 3
        assert g.trump_card is not None
        # 40 cards total, 6 dealt; trump is deck[0] (bottom) so it stays in deck
        assert len(g.deck) == 34
        assert g.game_over is False
        assert g.current_player == 0

    def test_play_and_resolve(self):
        g = BriscasGame()
        g.remove_and_play(0, 0)
        assert len(g.current_trick) == 1
        assert len(g.hands[0]) == 2
        assert not g.is_trick_complete()

        g.remove_and_play(1, 0)
        assert g.is_trick_complete()

        g.resolve_trick()
        assert len(g.current_trick) == 0
        # Hands replenished: both should have 3 cards again
        assert len(g.hands[0]) == 3
        assert len(g.hands[1]) == 3

    def test_full_game_completes(self):
        g = BriscasGame()
        while not g.game_over:
            # Current player plays first card
            p = g.current_player
            g.remove_and_play(p, 0)
            if not g.is_trick_complete():
                other = 1 - p
                g.remove_and_play(other, 0)
            g.resolve_trick()

        assert g.game_over
        total = g.score(0) + g.score(1)
        assert total == 120  # All 120 points accounted for

    def test_scores_sum_to_120(self):
        """Play 10 full games and verify scores always sum to 120."""
        for _ in range(10):
            g = BriscasGame()
            while not g.game_over:
                p = g.current_player
                g.remove_and_play(p, 0)
                if not g.is_trick_complete():
                    g.remove_and_play(1 - p, 0)
                g.resolve_trick()
            assert g.score(0) + g.score(1) == 120


# ---------------------------------------------------------------------------
# LocalAdapter interface tests
# ---------------------------------------------------------------------------


class TestLocalAdapterInterface:
    def test_implements_engine_adapter(self):
        adapter = LocalAdapter()
        assert isinstance(adapter, EngineAdapter)


class TestLocalAdapterNewGame:
    def test_returns_game_state(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        assert isinstance(state, GameState)
        assert len(state.hand) == 3
        assert state.trump is not None
        assert state.deck_remaining == 34
        assert state.game_over is False
        assert state.is_your_turn is True

    def test_hand_cards_are_adapter_cards(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        for card in state.hand:
            assert isinstance(card, Card)
            assert card.suit in ("Oros", "Copas", "Espadas", "Bastos")

    def test_players_info(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        assert len(state.players) == 2
        assert state.players[0].name == "rl_agent"
        assert state.players[0].is_human is True
        assert state.players[1].name == "Claude"
        assert state.players[1].is_human is False


class TestLocalAdapterPlayCard:
    def test_play_card_returns_state_with_trick(self):
        adapter = LocalAdapter()
        adapter.new_game()
        state = adapter.play_card(0)
        assert len(state.trick) == 1
        assert state.trick[0].player == "rl_agent"

    def test_play_card_reduces_hand(self):
        adapter = LocalAdapter()
        initial = adapter.new_game()
        state = adapter.play_card(0)
        assert len(state.hand) == len(initial.hand) - 1


class TestLocalAdapterProcessAI:
    def test_process_ai_after_human_play(self):
        adapter = LocalAdapter()
        adapter.new_game()
        adapter.play_card(0)  # Human plays, trick has 1 card
        state = adapter.process_ai_turn()
        # AI should have played, completing the trick
        assert len(state.trick) == 2

    def test_process_ai_when_human_turn_returns_immediately(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        # It's human's turn, process_ai should return current state
        ai_state = adapter.process_ai_turn()
        assert ai_state.is_your_turn is True


class TestLocalAdapterDeleteGame:
    def test_delete_cleans_up(self):
        adapter = LocalAdapter()
        adapter.new_game()
        adapter.delete_game()
        assert adapter._game is None


class TestLocalAdapterGetState:
    def test_get_state_after_resolution(self):
        adapter = LocalAdapter()
        adapter.new_game()
        adapter.play_card(0)
        adapter.process_ai_turn()  # Completes and resolves trick
        state = adapter.get_state()
        # After trick resolution, trick should be empty
        assert len(state.trick) == 0


# ---------------------------------------------------------------------------
# Full game lifecycle through LocalAdapter
# ---------------------------------------------------------------------------


class TestLocalAdapterFullGame:
    def test_full_game_lifecycle(self):
        """Play a complete game through the adapter, verify it terminates correctly."""
        adapter = LocalAdapter()
        state = adapter.new_game()

        moves = 0
        max_moves = 200  # Safety limit

        while not state.game_over and moves < max_moves:
            if state.is_your_turn and len(state.hand) > 0:
                state = adapter.play_card(0)
                moves += 1

            while not state.game_over:
                if len(state.trick) >= 2:
                    if len(state.hand) == 0 and state.deck_remaining == 0:
                        state = adapter.get_state()
                    else:
                        state = adapter.process_ai_turn()
                elif not state.is_your_turn:
                    state = adapter.process_ai_turn()
                else:
                    break

        assert state.game_over
        assert moves < max_moves
        # Scores should sum to 120
        total = state.players[0].score + state.players[1].score
        assert total == 120

    def test_multiple_games(self):
        """Play 20 games, verify all complete with valid scores."""
        adapter = LocalAdapter()

        for _ in range(20):
            state = adapter.new_game()
            while not state.game_over:
                if state.is_your_turn and len(state.hand) > 0:
                    state = adapter.play_card(0)
                while not state.game_over:
                    if len(state.trick) >= 2:
                        if len(state.hand) == 0 and state.deck_remaining == 0:
                            state = adapter.get_state()
                        else:
                            state = adapter.process_ai_turn()
                    elif not state.is_your_turn:
                        state = adapter.process_ai_turn()
                    else:
                        break

            assert state.game_over
            total = state.players[0].score + state.players[1].score
            assert total == 120
            adapter.delete_game()


class TestLocalAdapterObservationCompatibility:
    """Verify LocalAdapter produces states compatible with BriscasEnv observation encoding."""

    def test_card_ids_in_valid_range(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        for card in state.hand:
            cid = encode_card(card)
            assert 0 <= cid <= 39

    def test_trump_card_encodable(self):
        adapter = LocalAdapter()
        state = adapter.new_game()
        cid = encode_card(state.trump)
        assert 0 <= cid <= 39

    def test_trick_cards_encodable(self):
        adapter = LocalAdapter()
        adapter.new_game()
        state = adapter.play_card(0)
        for tc in state.trick:
            cid = encode_card(tc.card)
            assert 0 <= cid <= 39


# ---------------------------------------------------------------------------
# BriscasEnv + LocalAdapter integration (no mocks)
# ---------------------------------------------------------------------------


class TestBriscasEnvWithLocalAdapter:
    """End-to-end: real BriscasEnv + real LocalAdapter, no mocks."""

    def test_reset_and_step(self):
        env = BriscasEnv(adapter=LocalAdapter())
        obs, info = env.reset()
        assert obs.shape == (OBSERVATION_SIZE,)
        assert env.observation_space.contains(obs)

        obs2, reward, terminated, truncated, info = env.step(0)
        assert obs2.shape == (OBSERVATION_SIZE,)
        assert env.observation_space.contains(obs2)
        assert truncated is False
        env.close()

    def test_full_game_terminates(self):
        env = BriscasEnv(adapter=LocalAdapter())
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            steps += 1
        assert done
        assert "game_result" in info
        assert info["game_result"] in ("win", "loss", "draw")
        env.close()

    def test_multiple_games_via_reset(self):
        env = BriscasEnv(adapter=LocalAdapter())
        for _ in range(10):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated
            assert info["game_result"] in ("win", "loss", "draw")
        env.close()
