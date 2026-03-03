"""Tests for training/train.py — train_agent() and WinRateCallback."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gym_env.engine_adapter import (
    Card,
    EngineAdapter,
    EngineConnectionError,
    GameState,
    PlayerInfo,
)
from training.train import VALIDATION_NUM_GAMES, WinRateCallback, load_agent, train_agent, validate_worst_agent


# --- Helpers ---

def _card(rank: int, suit: str, points: int = 0) -> Card:
    symbols = {"Oros": "🪙", "Copas": "🏆", "Espadas": "⚔️", "Bastos": "🏑"}
    return Card(rank=rank, suit=suit, suit_symbol=symbols[suit],
                display_name=f"{rank} de {suit}", points=points)


def _players(agent_score: int = 0, opponent_score: int = 0) -> list[PlayerInfo]:
    return [
        PlayerInfo(name="rl_agent", is_current=True, is_human=True, score=agent_score, hand_size=3),
        PlayerInfo(name="ai", is_current=False, is_human=False, score=opponent_score, hand_size=3),
    ]


def _state(hand=None, game_over=False, agent_score=0, opponent_score=0, is_your_turn=True) -> GameState:
    if hand is None:
        hand = [_card(1, "Oros", 11), _card(3, "Copas", 10), _card(7, "Espadas")]
    return GameState(
        hand=hand, trump=_card(5, "Bastos"), trick=[], players=_players(agent_score, opponent_score),
        deck_remaining=30, round_number=1, game_over=game_over, winner=None, is_your_turn=is_your_turn,
    )


# --- load_agent() Unit Tests ---

class TestLoadAgentUnit:
    """Unit tests for load_agent() with mocked SB3."""

    @patch("training.train.DQN")
    def test_calls_dqn_load_with_correct_path(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "my_model")
        (tmp_path / "my_model.zip").write_bytes(b"fake")
        mock_model = MagicMock()
        mock_dqn_cls.load.return_value = mock_model

        load_agent(model_path)

        mock_dqn_cls.load.assert_called_once_with(model_path)

    @patch("training.train.DQN")
    def test_returns_model_and_metadata_tuple(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        metadata = {"agent_type": "best", "seed": 42, "total_timesteps": 5000,
                     "reward_type": "normalized_differential", "timestamp": "2026-01-01T00:00:00Z"}
        (tmp_path / "agent.json").write_text(json.dumps(metadata))
        mock_dqn_cls.load.return_value = MagicMock()

        model, meta = load_agent(model_path)

        assert model is mock_dqn_cls.load.return_value
        assert meta == metadata

    @patch("training.train.DQN")
    def test_returns_none_metadata_when_json_missing(self, mock_dqn_cls, tmp_path, caplog):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        mock_dqn_cls.load.return_value = MagicMock()

        import logging
        with caplog.at_level(logging.WARNING, logger="training.train"):
            model, meta = load_agent(model_path)

        assert meta is None
        assert any("No metadata file found" in r.message for r in caplog.records)

    def test_raises_file_not_found_when_zip_missing(self, tmp_path):
        model_path = str(tmp_path / "nonexistent_model")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_agent(model_path)

    @patch("training.train.DQN")
    def test_strips_zip_extension_from_input(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        mock_dqn_cls.load.return_value = MagicMock()

        load_agent(model_path + ".zip")

        mock_dqn_cls.load.assert_called_once_with(model_path)

    @patch("training.train.DQN")
    def test_handles_path_without_zip_extension(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        mock_dqn_cls.load.return_value = MagicMock()

        load_agent(model_path)

        mock_dqn_cls.load.assert_called_once_with(model_path)

    @patch("training.train.DQN")
    def test_metadata_contents_correct(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        expected = {"agent_type": "worst", "seed": 7, "total_timesteps": 10000,
                    "reward_type": "normalized_differential", "timestamp": "2026-02-01T12:00:00Z"}
        (tmp_path / "agent.json").write_text(json.dumps(expected))
        mock_dqn_cls.load.return_value = MagicMock()

        _, meta = load_agent(model_path)

        assert meta["agent_type"] == "worst"
        assert meta["seed"] == 7
        assert meta["total_timesteps"] == 10000
        assert meta["reward_type"] == "normalized_differential"
        assert meta["timestamp"] == "2026-02-01T12:00:00Z"

    @patch("training.train.DQN")
    def test_corrupted_zip_propagates_exception(self, mock_dqn_cls, tmp_path):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        mock_dqn_cls.load.side_effect = Exception("BadZipFile")

        with pytest.raises(Exception, match="BadZipFile"):
            load_agent(model_path)

    @patch("training.train.DQN")
    def test_malformed_json_returns_none_metadata(self, mock_dqn_cls, tmp_path, caplog):
        model_path = str(tmp_path / "agent")
        (tmp_path / "agent.zip").write_bytes(b"fake")
        (tmp_path / "agent.json").write_text("{bad json")
        mock_dqn_cls.load.return_value = MagicMock()

        import logging
        with caplog.at_level(logging.WARNING, logger="training.train"):
            model, meta = load_agent(model_path)

        assert meta is None
        assert model is mock_dqn_cls.load.return_value
        assert any("corrupted" in r.message for r in caplog.records)

    @patch("training.train.DQN")
    def test_double_zip_extension(self, mock_dqn_cls, tmp_path):
        # "path.zip.zip" → strips one .zip → "path.zip", SB3 loads "path.zip"
        base = str(tmp_path / "agent.zip")
        (tmp_path / "agent.zip.zip").write_bytes(b"fake")
        mock_dqn_cls.load.return_value = MagicMock()

        load_agent(base + ".zip")

        mock_dqn_cls.load.assert_called_once_with(base)


# --- WinRateCallback Unit Tests ---

class TestWinRateCallback:
    """Test WinRateCallback behavior with simulated data."""

    def _make_callback(self, window_size=1000, log_freq=100):
        cb = WinRateCallback(window_size=window_size, log_freq=log_freq)
        cb.num_timesteps = 0
        return cb

    def test_no_games_initial_state(self):
        cb = self._make_callback()
        assert cb.games_played == 0
        assert cb.win_rate == 0.0

    def test_single_win(self):
        cb = self._make_callback()
        cb.num_timesteps = 10
        cb.locals = {
            "dones": np.array([True]),
            "infos": [{"game_result": "win"}],
        }
        cb._on_step()
        assert cb.games_played == 1
        assert cb.win_rate == 1.0

    def test_single_loss(self):
        cb = self._make_callback()
        cb.num_timesteps = 10
        cb.locals = {
            "dones": np.array([True]),
            "infos": [{"game_result": "loss"}],
        }
        cb._on_step()
        assert cb.games_played == 1
        assert cb.win_rate == 0.0

    def test_mixed_results(self):
        cb = self._make_callback(window_size=10)
        results = ["win", "loss", "win", "draw", "win"]
        for i, result in enumerate(results):
            cb.num_timesteps = (i + 1) * 10
            cb.locals = {
                "dones": np.array([True]),
                "infos": [{"game_result": result}],
            }
            cb._on_step()
        assert cb.games_played == 5
        assert cb.win_rate == pytest.approx(3 / 5)

    def test_not_done_step_ignored(self):
        cb = self._make_callback()
        cb.locals = {
            "dones": np.array([False]),
            "infos": [{}],
        }
        cb._on_step()
        assert cb.games_played == 0

    def test_log_every_100_games(self):
        cb = self._make_callback(log_freq=100)
        for i in range(100):
            cb.num_timesteps = (i + 1) * 10
            cb.locals = {
                "dones": np.array([True]),
                "infos": [{"game_result": "win"}],
            }
            cb._on_step()
        assert cb.games_played == 100
        # Log should have fired at game 100

    def test_returns_true(self):
        cb = self._make_callback()
        cb.locals = {
            "dones": np.array([True]),
            "infos": [{"game_result": "win"}],
        }
        cb.num_timesteps = 1
        assert cb._on_step() is True

    def test_exposed_attributes(self):
        cb = self._make_callback()
        assert hasattr(cb, "games_played")
        assert hasattr(cb, "win_rate")


# --- train_agent() Unit Tests (Mocked SB3) ---

class TestTrainAgentUnit:
    """Unit tests for train_agent() with mocked SB3."""

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_dqn_instantiated_with_correct_params(self, mock_checkpoint, mock_dqn, mock_adapter, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("best", 1000, 42, str(tmp_path / "test_model"), "http://localhost:5000")

        mock_dqn.assert_called_once()
        args, kwargs = mock_dqn.call_args
        assert args[0] == "MlpPolicy"
        assert kwargs["verbose"] == 1
        assert kwargs["learning_starts"] == 1000
        assert kwargs["seed"] == 42

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_learn_called_with_timesteps_and_callbacks(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("best", 5000, 42, "/tmp/test_model")

        mock_model.learn.assert_called_once()
        _, kwargs = mock_model.learn.call_args
        assert kwargs["total_timesteps"] == 5000
        assert len(kwargs["callback"]) == 2

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_model_saved_to_output_path(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("best", 5000, 42, "/tmp/test_model")

        mock_model.save.assert_called_once_with("/tmp/test_model")

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_metadata_json_written(self, mock_checkpoint, mock_dqn, mock_adapter, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        output_path = str(tmp_path / "best_agent_5k")
        train_agent("best", 5000, 42, output_path)

        metadata_path = output_path + ".json"
        assert os.path.exists(metadata_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["agent_type"] == "best"
        assert metadata["seed"] == 42
        assert metadata["total_timesteps"] == 5000
        assert metadata["reward_type"] == "normalized_differential"
        assert "timestamp" in metadata

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_metadata_path_matches_model_path(self, mock_checkpoint, mock_dqn, mock_adapter, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        output_path = str(tmp_path / "my_agent")
        train_agent("best", 100, 42, output_path)

        # model saves to output_path + ".zip" (SB3), metadata to output_path + ".json"
        assert os.path.exists(output_path + ".json")

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_worst_agent_uses_negative_reward_scale(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("worst", 100, 42, "/tmp/test_worst")

        # The BriscasEnv should have been created with reward_scale=-1.0
        # Verify by checking the env passed to DQN
        args, kwargs = mock_dqn.call_args
        env = args[1]
        assert env.reward_scale == -1.0

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_best_agent_uses_positive_reward_scale(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("best", 100, 42, "/tmp/test_best")

        args, kwargs = mock_dqn.call_args
        env = args[1]
        assert env.reward_scale == 1.0

    @patch("training.train.RESTAdapter")
    def test_preflight_check_raises_on_connection_error(self, mock_adapter):
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.side_effect = EngineConnectionError("Connection refused")
        mock_adapter.return_value = mock_adapter_instance

        with pytest.raises(EngineConnectionError):
            train_agent("best", 100, 42, "/tmp/test_model")

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_env_close_called_on_exception(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_model.learn.side_effect = RuntimeError("Training exploded")
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        with pytest.raises(RuntimeError, match="Training exploded"):
            train_agent("best", 100, 42, "/tmp/test_model")

        # env.close() should have been called via finally
        mock_adapter_instance.delete_game.assert_called()

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_dqn_learning_starts_1000(self, mock_checkpoint, mock_dqn, mock_adapter):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        train_agent("best", 100, 42, "/tmp/test_model")

        _, kwargs = mock_dqn.call_args
        assert kwargs["learning_starts"] == 1000


# --- Integration Test (Real SB3, Mocked Adapter) ---

def _mock_adapter_for_integration():
    """Create a mock adapter that simulates multiple games."""
    adapter = MagicMock(spec=EngineAdapter)
    hand = [_card(1, "Oros", 11), _card(3, "Copas", 10), _card(7, "Espadas")]
    trump = _card(5, "Bastos")

    initial_state = GameState(
        hand=hand, trump=trump, trick=[], players=_players(0, 0),
        deck_remaining=30, round_number=1, game_over=False, winner=None, is_your_turn=True,
    )

    # Each play_card call alternates between mid-game and terminal states
    # to simulate short games (1 step each)
    terminal_state = GameState(
        hand=[], trump=trump, trick=[], players=_players(70, 50),
        deck_remaining=0, round_number=1, game_over=True, winner="rl_agent", is_your_turn=False,
    )

    adapter.new_game.return_value = initial_state
    adapter.play_card.return_value = terminal_state

    return adapter


@pytest.mark.integration
def test_integration_train_agent(tmp_path):
    """Real SB3 + mocked adapter, ~100 timesteps."""
    adapter = _mock_adapter_for_integration()
    output_path = str(tmp_path / "best_agent_test")

    # Capture WinRateCallback instance to verify it tracked games
    captured_callbacks = []
    _OriginalWinRateCallback = WinRateCallback

    def _spy_winrate(*args, **kwargs):
        cb = _OriginalWinRateCallback(*args, **kwargs)
        captured_callbacks.append(cb)
        return cb

    with patch("training.train.RESTAdapter", return_value=adapter), \
         patch("training.train.WinRateCallback", side_effect=_spy_winrate):
        train_agent(
            agent_type="best",
            total_timesteps=100,
            seed=42,
            output_path=output_path,
        )

    # Verify model file created (SB3 appends .zip)
    assert os.path.exists(output_path + ".zip")

    # Verify metadata file created
    metadata_path = output_path + ".json"
    assert os.path.exists(metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["agent_type"] == "best"
    assert metadata["seed"] == 42
    assert metadata["total_timesteps"] == 100

    # Verify WinRateCallback actually tracked games with correct results
    assert len(captured_callbacks) == 1
    cb = captured_callbacks[0]
    assert cb.games_played > 0, "WinRateCallback should have tracked at least 1 game"
    assert cb.win_rate > 0.0, "Win rate should be > 0 (all games are wins in mock)"


@pytest.mark.integration
def test_integration_train_worst_agent(tmp_path):
    """Real SB3 + mocked adapter for worst agent — verify training + real validation end-to-end."""
    adapter = _mock_adapter_for_integration()
    output_path = str(tmp_path / "worst_agent_test")

    with patch("training.train.RESTAdapter", return_value=adapter), \
         patch("training.train.VALIDATION_NUM_GAMES", 10):
        train_agent(
            agent_type="worst",
            total_timesteps=100,
            seed=42,
            output_path=output_path,
        )

    # Verify model file created (SB3 appends .zip)
    assert os.path.exists(output_path + ".zip")

    # Verify metadata file created with validation fields
    metadata_path = output_path + ".json"
    assert os.path.exists(metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["agent_type"] == "worst"
    assert "validation_win_rate" in metadata
    assert metadata["validation_games"] == 10


# --- validate_worst_agent() Unit Tests ---

class TestValidateWorstAgent:
    """Unit tests for validate_worst_agent() with mocked env."""

    def _mock_env_and_model(self, results):
        """Create mock model and env that produce given game results."""
        mock_model = MagicMock()
        mock_adapter = MagicMock(spec=EngineAdapter)

        obs = np.zeros(42, dtype=np.float32)
        action_array = np.array([0])
        mock_model.predict.return_value = (action_array, None)

        # Track game index
        game_idx = {"i": 0}

        def mock_reset(**kwargs):
            return obs, {}

        def mock_step(action):
            result = results[game_idx["i"]]
            game_idx["i"] += 1
            info = {"game_result": result}
            return obs, 0.0, True, False, info

        return mock_model, mock_adapter, mock_reset, mock_step

    @patch("training.train.BriscasEnv")
    def test_returns_correct_win_rate(self, mock_env_cls):
        results = ["win", "loss", "loss", "win", "loss"]  # 2/5 = 0.4
        mock_model, mock_adapter, mock_reset, mock_step = self._mock_env_and_model(results)
        mock_env = MagicMock()
        mock_env.reset = mock_reset
        mock_env.step = mock_step
        mock_env_cls.return_value = mock_env

        rate = validate_worst_agent(mock_model, mock_adapter, num_games=5)
        assert rate == pytest.approx(0.4)

    @patch("training.train.BriscasEnv")
    def test_warning_logged_when_win_rate_above_threshold(self, mock_env_cls, caplog):
        # All wins → 1.0 > 0.45 → warning
        results = ["win"] * 5
        mock_model, mock_adapter, mock_reset, mock_step = self._mock_env_and_model(results)
        mock_env = MagicMock()
        mock_env.reset = mock_reset
        mock_env.step = mock_step
        mock_env_cls.return_value = mock_env

        import logging
        with caplog.at_level(logging.WARNING, logger="training.train"):
            validate_worst_agent(mock_model, mock_adapter, num_games=5)

        assert any("anti-optimal" in r.message for r in caplog.records)

    @patch("training.train.BriscasEnv")
    def test_no_warning_when_win_rate_below_threshold(self, mock_env_cls, caplog):
        # All losses → 0.0 < 0.45 → no warning
        results = ["loss"] * 5
        mock_model, mock_adapter, mock_reset, mock_step = self._mock_env_and_model(results)
        mock_env = MagicMock()
        mock_env.reset = mock_reset
        mock_env.step = mock_step
        mock_env_cls.return_value = mock_env

        import logging
        with caplog.at_level(logging.WARNING, logger="training.train"):
            validate_worst_agent(mock_model, mock_adapter, num_games=5)

        assert not any("anti-optimal" in r.message for r in caplog.records)

    @patch("training.train.BriscasEnv")
    def test_env_close_called_on_exception(self, mock_env_cls):
        mock_model = MagicMock()
        mock_adapter = MagicMock(spec=EngineAdapter)
        mock_env = MagicMock()
        mock_env.reset.side_effect = EngineConnectionError("Engine down")
        mock_env_cls.return_value = mock_env

        with pytest.raises(EngineConnectionError):
            validate_worst_agent(mock_model, mock_adapter, num_games=5)

        mock_env.close.assert_called_once()

    @patch("training.train.BriscasEnv")
    def test_warning_at_exact_threshold(self, mock_env_cls, caplog):
        # 9/20 = 0.45 = exactly at threshold, >= triggers warning
        results = ["win"] * 9 + ["loss"] * 11
        mock_model, mock_adapter, mock_reset, mock_step = self._mock_env_and_model(results)
        mock_env = MagicMock()
        mock_env.reset = mock_reset
        mock_env.step = mock_step
        mock_env_cls.return_value = mock_env

        import logging
        with caplog.at_level(logging.WARNING, logger="training.train"):
            validate_worst_agent(mock_model, mock_adapter, num_games=20)

        assert any("anti-optimal" in r.message for r in caplog.records)

    def test_raises_on_zero_num_games(self):
        with pytest.raises(ValueError, match="num_games must be positive"):
            validate_worst_agent(MagicMock(), MagicMock(), num_games=0)

    @patch("training.train.BriscasEnv")
    def test_env_created_with_reward_scale_1(self, mock_env_cls):
        results = ["loss"] * 3
        mock_model, mock_adapter, mock_reset, mock_step = self._mock_env_and_model(results)
        mock_env = MagicMock()
        mock_env.reset = mock_reset
        mock_env.step = mock_step
        mock_env_cls.return_value = mock_env

        validate_worst_agent(mock_model, mock_adapter, num_games=3)

        mock_env_cls.assert_called_once_with(adapter=mock_adapter, reward_scale=1.0)


# --- train_agent() validation integration tests ---

class TestTrainAgentValidation:
    """Tests for validation integration in train_agent()."""

    @patch("training.train.validate_worst_agent")
    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_validation_failure_caught_gracefully(self, mock_checkpoint, mock_dqn, mock_adapter, mock_validate, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance
        mock_validate.side_effect = EngineConnectionError("Engine down during validation")

        output_path = str(tmp_path / "worst_agent")
        # Should NOT raise — validation failure is non-fatal
        train_agent("worst", 100, 42, output_path)

        # Base metadata should still exist
        with open(output_path + ".json") as f:
            metadata = json.load(f)
        assert metadata["agent_type"] == "worst"
        assert "validation_win_rate" not in metadata
        assert "validation_games" not in metadata

    @patch("training.train.validate_worst_agent", return_value=0.3)
    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_metadata_includes_validation_fields_for_worst(self, mock_checkpoint, mock_dqn, mock_adapter, mock_validate, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        output_path = str(tmp_path / "worst_agent")
        train_agent("worst", 100, 42, output_path)

        with open(output_path + ".json") as f:
            metadata = json.load(f)
        assert metadata["validation_win_rate"] == 0.3
        assert metadata["validation_games"] == VALIDATION_NUM_GAMES

    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_metadata_no_validation_fields_for_best(self, mock_checkpoint, mock_dqn, mock_adapter, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance

        output_path = str(tmp_path / "best_agent")
        train_agent("best", 100, 42, output_path)

        with open(output_path + ".json") as f:
            metadata = json.load(f)
        assert "validation_win_rate" not in metadata
        assert "validation_games" not in metadata

    @patch("training.train.validate_worst_agent")
    @patch("training.train.RESTAdapter")
    @patch("training.train.DQN")
    @patch("training.train.CheckpointCallback")
    def test_metadata_no_validation_fields_when_validation_fails(self, mock_checkpoint, mock_dqn, mock_adapter, mock_validate, tmp_path):
        mock_model = MagicMock()
        mock_dqn.return_value = mock_model
        mock_adapter_instance = MagicMock(spec=EngineAdapter)
        mock_adapter_instance.new_game.return_value = _state()
        mock_adapter.return_value = mock_adapter_instance
        mock_validate.side_effect = RuntimeError("Unexpected error")

        output_path = str(tmp_path / "worst_agent")
        train_agent("worst", 100, 42, output_path)

        with open(output_path + ".json") as f:
            metadata = json.load(f)
        assert "validation_win_rate" not in metadata
        assert "validation_games" not in metadata


# --- load_agent() Integration Test ---

@pytest.mark.integration
def test_integration_load_agent(tmp_path):
    """Train a model with real SB3, save it, then load with load_agent()."""
    adapter = _mock_adapter_for_integration()
    output_path = str(tmp_path / "best_agent_roundtrip")

    with patch("training.train.RESTAdapter", return_value=adapter):
        train_agent(
            agent_type="best",
            total_timesteps=100,
            seed=42,
            output_path=output_path,
        )

    # Now load the model with load_agent
    model, metadata = load_agent(output_path)

    # Verify model can predict (observation space is 13-dim)
    obs = np.zeros(13, dtype=np.float32)
    action, _states = model.predict(obs, deterministic=True)
    assert isinstance(action, np.ndarray)
    assert 0 <= action.item() <= 2, f"Action {action.item()} out of valid range [0, 2]"

    # Verify metadata round-trip
    assert metadata["agent_type"] == "best"
    assert metadata["seed"] == 42
    assert metadata["total_timesteps"] == 100
    assert metadata["reward_type"] == "normalized_differential"
    assert "timestamp" in metadata
