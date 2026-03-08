"""Tests for evaluation/evaluate.py — run_evaluation() and agent name extraction."""

import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.evaluate import run_evaluation, _extract_agent_name


# --- Helpers ---

def _mock_model():
    """Create a mock DQN model that returns action 0."""
    model = MagicMock()
    model.predict.return_value = (np.array([0]), None)
    return model


def _mock_env_step_sequence(num_games):
    """Build side_effect lists for env.reset and env.step to simulate num_games games."""
    obs = np.zeros(50, dtype=np.float32)
    resets = [(obs, {}) for _ in range(num_games)]
    steps = []
    for g in range(num_games):
        # Each game: one intermediate step then terminal
        steps.append((obs, 0.0, False, False, {}))
        steps.append((
            obs, 1.0, True, False,
            {"game_result": "win", "agent_points": 70, "opponent_points": 50},
        ))
    return resets, steps


# --- Unit tests ---

class TestRunEvaluationValidation:
    """Test argument validation in run_evaluation."""

    def test_both_random_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one agent must be a trained model"):
            run_evaluation("random", "random", 10, 42, output_dir=str(tmp_path))

    def test_no_random_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Exactly one agent must be 'random'"):
            run_evaluation("models/a.zip", "models/b.zip", 10, 42, output_dir=str(tmp_path))

    def test_num_games_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="num_games must be a positive integer"):
            run_evaluation("models/best.zip", "random", 0, 42, output_dir=str(tmp_path))

    def test_num_games_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="num_games must be a positive integer"):
            run_evaluation("models/best.zip", "random", -5, 42, output_dir=str(tmp_path))


class TestRunEvaluationGames:
    """Test run_evaluation plays correct number of games and produces valid CSV."""

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_plays_exactly_n_games(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        num_games = 5
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(num_games)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("models/best.zip", "random", num_games, 42, output_dir=str(tmp_path))

        assert mock_env.reset.call_count == num_games
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == num_games

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_csv_columns(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(3)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("models/best.zip", "random", 3, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "game_id", "agent1_points", "agent2_points", "first_player", "point_differential",
            ]

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_first_player_recorded_from_env(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """first_player in CSV reflects actual game state, not hardcoded 0."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        obs = np.zeros(50, dtype=np.float32)
        # Game 0: model first, Game 1: AI first
        resets = [
            (obs, {"first_player": 0}),
            (obs, {"first_player": 1}),
        ]
        steps = []
        for _ in range(2):
            steps.append((obs, 0.0, False, False, {}))
            steps.append((
                obs, 1.0, True, False,
                {"game_result": "win", "agent_points": 70, "opponent_points": 50},
            ))
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        # agent1=model, agent2=random → model first_player=0 maps to csv 0, AI first maps to csv 1
        csv_path = run_evaluation("models/best.zip", "random", 2, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["first_player"] == "0"
        assert rows[1]["first_player"] == "1"

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_random_agent_no_model_predict(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """When agent2 is 'random', only the model for agent1 calls predict."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(2)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        run_evaluation("models/best.zip", "random", 2, 42, output_dir=str(tmp_path))

        # load_agent only called once (for the model, not for "random")
        mock_load.assert_called_once_with("models/best.zip")
        # model.predict is called (agent plays via predict)
        assert model.predict.call_count > 0

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_point_differential(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("models/best.zip", "random", 1, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # agent1 is model (agent_points=70), agent2 is random (opponent_points=50)
        assert int(row["agent1_points"]) == 70
        assert int(row["agent2_points"]) == 50
        assert int(row["point_differential"]) == 20

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_point_differential_random_as_agent1(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """When random is agent1, points are swapped correctly."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("random", "models/best.zip", 1, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # agent1 is random (opponent_points=50), agent2 is model (agent_points=70)
        assert int(row["agent1_points"]) == 50
        assert int(row["agent2_points"]) == 70
        assert int(row["point_differential"]) == -20

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_output_directory_created(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        out_dir = str(tmp_path / "new_results")
        csv_path = run_evaluation("models/best.zip", "random", 1, 42, output_dir=out_dir)

        assert os.path.isdir(out_dir)
        assert os.path.isfile(csv_path)


    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_env_close_called_on_exception(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        mock_env = MagicMock()
        obs = np.zeros(50, dtype=np.float32)
        mock_env.reset.return_value = (obs, {})
        mock_env.step.side_effect = RuntimeError("Engine exploded")
        mock_env_cls.return_value = mock_env

        with pytest.raises(RuntimeError, match="Engine exploded"):
            run_evaluation("models/best.zip", "random", 1, 42, output_dir=str(tmp_path))

        mock_env.close.assert_called_once()

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_env_created_with_reward_scale_1(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        run_evaluation("models/best.zip", "random", 1, 42, output_dir=str(tmp_path))

        mock_adapter = mock_adapter_cls.return_value
        mock_env_cls.assert_called_once_with(adapter=mock_adapter, reward_scale=1.0)

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_output_path_override(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        custom_path = str(tmp_path / "custom" / "my_results.csv")
        csv_path = run_evaluation("models/best.zip", "random", 1, 42, output_path=custom_path)

        assert csv_path == custom_path
        assert os.path.isfile(csv_path)


class TestCsvFilenameConvention:
    """Test CSV filename follows naming convention."""

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_filename_format(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("models/best_agent_50k", "random", 1, 42, output_dir=str(tmp_path))

        assert os.path.basename(csv_path) == "best_agent_50k_vs_random_1g_42s.csv"


class TestExtractAgentName:
    """Test _extract_agent_name utility."""

    def test_random_literal(self):
        assert _extract_agent_name("random") == "random"

    def test_path_basename(self):
        assert _extract_agent_name("models/best_agent_50k") == "best_agent_50k"

    def test_strips_zip(self):
        assert _extract_agent_name("models/best_agent_50k.zip") == "best_agent_50k"

    def test_nested_path(self):
        assert _extract_agent_name("checkpoints/run1/model") == "model"


# --- Integration test ---

@pytest.mark.integration
def test_integration_train_and_evaluate(tmp_path):
    """Train a minimal model and run evaluation games against random."""
    from stable_baselines3 import DQN
    from gym_env.briscas_env import BriscasEnv
    from gym_env.local_adapter import LocalAdapter

    adapter = LocalAdapter()
    env = BriscasEnv(adapter=adapter)
    try:
        model = DQN("MlpPolicy", env, verbose=0, learning_starts=50, seed=42)
        model.learn(total_timesteps=100)
        model_path = str(tmp_path / "test_model")
        model.save(model_path)
    finally:
        env.close()

    out_dir = str(tmp_path / "eval_results")
    csv_path = run_evaluation(
        agent1=model_path,
        agent2="random",
        num_games=10,
        seed=42,
        output_dir=out_dir,
    )

    assert os.path.isfile(csv_path)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10
    for row in rows:
        assert 0 <= int(row["agent1_points"]) <= 120
        assert 0 <= int(row["agent2_points"]) <= 120
        assert int(row["point_differential"]) == int(row["agent1_points"]) - int(row["agent2_points"])
