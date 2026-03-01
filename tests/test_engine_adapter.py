"""Unit tests for gym_env.engine_adapter with mocked HTTP responses."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import requests

from gym_env.engine_adapter import (
    Card,
    EngineAdapter,
    EngineConnectionError,
    GameState,
    RESTAdapter,
)


# --- Fixtures ---


def _make_card(rank=1, suit="Oros", points=11):
    return {
        "rank": rank,
        "suit": suit,
        "suit_symbol": "🪙",
        "display_name": f"{rank} de {suit}",
        "points": points,
    }


def _make_state(
    hand=None, game_over=False, winner=None, is_your_turn=True, trick=None
):
    return {
        "hand": hand or [_make_card(1), _make_card(3, points=10), _make_card(7, points=0)],
        "trump": _make_card(12, suit="Copas", points=4),
        "trick": trick or [],
        "players": [
            {"name": "rl_agent", "is_current": True, "is_human": True, "score": 0, "hand_size": 3},
            {"name": "AI", "is_current": False, "is_human": False, "score": 0, "hand_size": 3},
        ],
        "deck_remaining": 34,
        "round_number": 1,
        "game_over": game_over,
        "winner": winner,
        "is_your_turn": is_your_turn,
    }


def _mock_response(json_data, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_error_response(status_code):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} Server Error"
    )
    return resp


# --- Abstract interface contract ---


class TestEngineAdapterInterface:
    def test_is_abstract(self):
        assert inspect.isabstract(EngineAdapter)

    def test_abstract_methods_defined(self):
        abstract_methods = EngineAdapter.__abstractmethods__
        expected = {"new_game", "play_card", "process_ai_turn", "get_state", "delete_game"}
        assert abstract_methods == expected

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EngineAdapter()


# --- RESTAdapter unit tests ---


class TestRESTAdapterNewGame:
    def test_new_game_returns_game_state(self):
        adapter = RESTAdapter("http://localhost:5000")
        state_data = _make_state()
        resp = _mock_response({"success": True, "game_id": "abc", "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            state = adapter.new_game()
            mock_req.assert_called_once_with(
                "POST",
                "http://localhost:5000/api/game/new",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
                json={"player_name": "rl_agent", "ai_difficulty": "basic"},
            )

        assert isinstance(state, GameState)
        assert len(state.hand) == 3
        assert state.hand[0].rank == 1
        assert state.hand[0].points == 11
        assert state.trump.suit == "Copas"
        assert state.deck_remaining == 34
        assert state.game_over is False


class TestRESTAdapterPlayCard:
    def test_play_card_sends_card_index(self):
        adapter = RESTAdapter()
        state_data = _make_state(
            trick=[{"player": "rl_agent", "card": _make_card(1)}]
        )
        resp = _mock_response({"success": True, "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            state = adapter.play_card(0)
            mock_req.assert_called_once_with(
                "POST",
                "http://localhost:5000/api/game/play",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
                json={"card_index": 0},
            )

        assert len(state.trick) == 1
        assert state.trick[0].player == "rl_agent"


class TestRESTAdapterProcessAI:
    def test_process_ai_turn(self):
        adapter = RESTAdapter()
        state_data = _make_state(is_your_turn=True)
        resp = _mock_response({"success": True, "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            state = adapter.process_ai_turn()
            mock_req.assert_called_once_with(
                "POST", "http://localhost:5000/api/game/process-ai",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
            )

        assert isinstance(state, GameState)


class TestRESTAdapterGetState:
    def test_get_state(self):
        adapter = RESTAdapter()
        state_data = _make_state()
        resp = _mock_response({"success": True, "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            state = adapter.get_state()
            mock_req.assert_called_once_with(
                "GET", "http://localhost:5000/api/game/state",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
            )

        assert isinstance(state, GameState)


class TestRESTAdapterDeleteGame:
    def test_delete_game(self):
        adapter = RESTAdapter()
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.raise_for_status.return_value = None

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.delete_game()
            mock_req.assert_called_once_with(
                "DELETE", "http://localhost:5000/api/game/delete",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
            )


class TestRESTAdapterGameLifecycle:
    def test_full_lifecycle(self):
        """Test new_game → play_card → process_ai → get_state → game_over flow."""
        adapter = RESTAdapter()

        responses = [
            # new_game
            _mock_response({"success": True, "game_id": "abc", "state": _make_state()}),
            # play_card
            _mock_response({"success": True, "state": _make_state(
                trick=[{"player": "rl_agent", "card": _make_card(1)}]
            )}),
            # process_ai_turn
            _mock_response({"success": True, "state": _make_state(
                trick=[
                    {"player": "rl_agent", "card": _make_card(1)},
                    {"player": "AI", "card": _make_card(7, points=0)},
                ]
            )}),
            # get_state (game over)
            _mock_response({"success": True, "state": _make_state(
                game_over=True, winner="rl_agent"
            )}),
            # delete_game
            _mock_response({"success": True}),
        ]

        with patch.object(adapter._session, "request", side_effect=responses):
            state = adapter.new_game()
            assert not state.game_over

            state = adapter.play_card(0)
            assert len(state.trick) == 1

            state = adapter.process_ai_turn()
            assert len(state.trick) == 2

            state = adapter.get_state()
            assert state.game_over is True
            assert state.winner == "rl_agent"

            adapter.delete_game()


class TestRESTAdapterErrorHandling:
    def test_connection_error_raises_engine_connection_error(self):
        adapter = RESTAdapter()
        with patch.object(
            adapter._session,
            "request",
            side_effect=requests.ConnectionError("Connection refused"),
        ):
            with pytest.raises(EngineConnectionError, match="Cannot connect"):
                adapter.new_game()

    def test_http_error_raises_engine_connection_error(self):
        adapter = RESTAdapter()
        resp = _mock_error_response(500)
        with patch.object(adapter._session, "request", return_value=resp):
            with pytest.raises(EngineConnectionError, match="returned error 500"):
                adapter.new_game()

    def test_timeout_raises_engine_connection_error(self):
        adapter = RESTAdapter()
        with patch.object(
            adapter._session,
            "request",
            side_effect=requests.Timeout("Request timed out"),
        ):
            with pytest.raises(EngineConnectionError, match="Request failed"):
                adapter.new_game()

    def test_non_200_on_play_card(self):
        adapter = RESTAdapter()
        resp = _mock_error_response(400)
        with patch.object(adapter._session, "request", return_value=resp):
            with pytest.raises(EngineConnectionError):
                adapter.play_card(0)


class TestRESTAdapterMalformedResponses:
    def test_invalid_json_raises_engine_connection_error(self):
        adapter = RESTAdapter()
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("No JSON object could be decoded")

        with patch.object(adapter._session, "request", return_value=resp):
            with pytest.raises(EngineConnectionError, match="Invalid JSON"):
                adapter.new_game()

    def test_success_false_raises_engine_connection_error(self):
        adapter = RESTAdapter()
        resp = _mock_response({"success": False, "error": "game not found"})

        with patch.object(adapter._session, "request", return_value=resp):
            with pytest.raises(EngineConnectionError, match="returned failure"):
                adapter.get_state()


class TestRESTAdapterTimeout:
    def test_default_timeout_is_set(self):
        adapter = RESTAdapter()
        resp = _mock_response({"success": True, "state": _make_state()})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.get_state()
            _, kwargs = mock_req.call_args
            assert kwargs["timeout"] == RESTAdapter.DEFAULT_TIMEOUT

    def test_delete_game_does_not_parse_json(self):
        adapter = RESTAdapter()
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.raise_for_status.return_value = None

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.delete_game()
            mock_req.assert_called_once()
            resp.json.assert_not_called()


class TestRESTAdapterSessionPersistence:
    def test_uses_requests_session(self):
        adapter = RESTAdapter()
        assert isinstance(adapter._session, requests.Session)

    def test_session_reused_across_calls(self):
        """Verify the same session object is used for multiple requests (cookie persistence)."""
        adapter = RESTAdapter()
        state_data = _make_state()
        resp = _mock_response({"success": True, "game_id": "abc", "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.new_game()
            adapter.get_state()
            assert mock_req.call_count == 2


class TestRESTAdapterBaseURL:
    def test_custom_base_url(self):
        adapter = RESTAdapter("http://custom-host:8080")
        resp = _mock_response({"success": True, "state": _make_state()})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.get_state()
            mock_req.assert_called_once_with(
                "GET", "http://custom-host:8080/api/game/state",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
            )

    def test_trailing_slash_stripped(self):
        adapter = RESTAdapter("http://localhost:5000/")
        resp = _mock_response({"success": True, "state": _make_state()})

        with patch.object(adapter._session, "request", return_value=resp) as mock_req:
            adapter.get_state()
            mock_req.assert_called_once_with(
                "GET", "http://localhost:5000/api/game/state",
                timeout=RESTAdapter.DEFAULT_TIMEOUT,
            )


class TestGameStateParsing:
    def test_card_parsing(self):
        adapter = RESTAdapter()
        card_data = _make_card(12, "Copas", 4)
        state_data = _make_state(hand=[card_data])
        resp = _mock_response({"success": True, "state": state_data})

        with patch.object(adapter._session, "request", return_value=resp):
            state = adapter.get_state()

        card = state.hand[0]
        assert isinstance(card, Card)
        assert card.rank == 12
        assert card.suit == "Copas"
        assert card.points == 4
        assert card.display_name == "12 de Copas"

    def test_player_info_parsing(self):
        adapter = RESTAdapter()
        resp = _mock_response({"success": True, "state": _make_state()})

        with patch.object(adapter._session, "request", return_value=resp):
            state = adapter.get_state()

        assert len(state.players) == 2
        assert state.players[0].name == "rl_agent"
        assert state.players[0].is_human is True
        assert state.players[1].name == "AI"
        assert state.players[1].is_human is False
