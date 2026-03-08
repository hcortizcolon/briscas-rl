"""Tests for evaluation/evaluate.py — run_evaluation() and agent name extraction."""

import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.evaluate import (
    run_evaluation,
    _extract_agent_name,
    compute_summary_statistics,
    print_summary_statistics,
    compute_summary_statistics_from_csv,
)


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

    def test_both_advanced_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one agent must be a trained model"):
            run_evaluation("advanced", "advanced", 10, 42, output_dir=str(tmp_path))

    def test_both_random_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one agent must be a trained model"):
            run_evaluation("random", "random", 10, 42, output_dir=str(tmp_path))

    def test_mixed_engine_strategies_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one agent must be a trained model"):
            run_evaluation("random", "advanced", 10, 42, output_dir=str(tmp_path))

    def test_no_engine_strategy_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Exactly one agent must be an engine strategy"):
            run_evaluation("models/a.zip", "models/b.zip", 10, 42, output_dir=str(tmp_path))

    def test_num_games_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="num_games must be a positive integer"):
            run_evaluation("models/best.zip", "advanced", 0, 42, output_dir=str(tmp_path))

    def test_num_games_negative_raises(self, tmp_path):
        with pytest.raises(ValueError, match="num_games must be a positive integer"):
            run_evaluation("models/best.zip", "advanced", -5, 42, output_dir=str(tmp_path))


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

        csv_path = run_evaluation("models/best.zip", "advanced", num_games, 42, output_dir=str(tmp_path))

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

        csv_path = run_evaluation("models/best.zip", "advanced", 3, 42, output_dir=str(tmp_path))

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

        # agent1=model, agent2=advanced → model first_player=0 maps to csv 0, AI first maps to csv 1
        csv_path = run_evaluation("models/best.zip", "advanced", 2, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["first_player"] == "0"
        assert rows[1]["first_player"] == "1"

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_advanced_agent_no_model_predict(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """When agent2 is 'advanced', only the model for agent1 calls predict."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(2)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        run_evaluation("models/best.zip", "advanced", 2, 42, output_dir=str(tmp_path))

        # load_agent only called once (for the model, not for "advanced")
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

        csv_path = run_evaluation("models/best.zip", "advanced", 1, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # agent1 is model (agent_points=70), agent2 is advanced (opponent_points=50)
        assert int(row["agent1_points"]) == 70
        assert int(row["agent2_points"]) == 50
        assert int(row["point_differential"]) == 20

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_point_differential_advanced_as_agent1(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """When advanced is agent1, points are swapped correctly."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(1)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("advanced", "models/best.zip", 1, 42, output_dir=str(tmp_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # agent1 is advanced (opponent_points=50), agent2 is model (agent_points=70)
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
        csv_path = run_evaluation("models/best.zip", "advanced", 1, 42, output_dir=out_dir)

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
            run_evaluation("models/best.zip", "advanced", 1, 42, output_dir=str(tmp_path))

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

        run_evaluation("models/best.zip", "advanced", 1, 42, output_dir=str(tmp_path))

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
        csv_path = run_evaluation("models/best.zip", "advanced", 1, 42, output_path=custom_path)

        assert csv_path == custom_path
        assert os.path.isfile(csv_path)


class TestCsvFilenameConvention:
    """Test CSV filename follows naming convention."""

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_random_strategy_runs(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path):
        """run_evaluation works with 'random' as the engine strategy."""
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(2)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        csv_path = run_evaluation("models/best.zip", "random", 2, 42, output_dir=str(tmp_path))

        mock_adapter_cls.assert_called_once_with(strategy="random")
        assert os.path.basename(csv_path) == "best_vs_random_2g_42s.csv"

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

        csv_path = run_evaluation("models/best_agent_50k", "advanced", 1, 42, output_dir=str(tmp_path))

        assert os.path.basename(csv_path) == "best_agent_50k_vs_advanced_1g_42s.csv"


class TestExtractAgentName:
    """Test _extract_agent_name utility."""

    def test_advanced_literal(self):
        assert _extract_agent_name("advanced") == "advanced"

    def test_random_literal(self):
        assert _extract_agent_name("random") == "random"

    def test_path_basename(self):
        assert _extract_agent_name("models/best_agent_50k") == "best_agent_50k"

    def test_strips_zip(self):
        assert _extract_agent_name("models/best_agent_50k.zip") == "best_agent_50k"

    def test_nested_path(self):
        assert _extract_agent_name("checkpoints/run1/model") == "model"


# --- Summary Statistics Tests ---

class TestSummaryStatistics:
    """Test compute_summary_statistics() and print_summary_statistics()."""

    def test_win_loss_draw_counts_and_percentages(self):
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
            {"agent1_points": 30, "agent2_points": 90, "point_differential": -60},
            {"agent1_points": 60, "agent2_points": 60, "point_differential": 0},
            {"agent1_points": 70, "agent2_points": 50, "point_differential": 20},
        ]
        stats = compute_summary_statistics(results)
        assert stats["agent1_wins"] == 2
        assert stats["agent1_losses"] == 1
        assert stats["agent1_draws"] == 1
        assert stats["agent2_wins"] == 1
        assert stats["agent2_losses"] == 2
        assert stats["agent2_draws"] == 1
        assert stats["total_games"] == 4
        assert stats["agent1_win_pct"] == pytest.approx(50.0)
        assert stats["agent1_loss_pct"] == pytest.approx(25.0)
        assert stats["agent1_draw_pct"] == pytest.approx(25.0)
        assert stats["agent2_win_pct"] == pytest.approx(25.0)
        assert stats["agent2_loss_pct"] == pytest.approx(50.0)
        assert stats["agent2_draw_pct"] == pytest.approx(25.0)

    def test_avg_point_differential(self):
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
            {"agent1_points": 30, "agent2_points": 90, "point_differential": -60},
        ]
        stats = compute_summary_statistics(results)
        assert stats["avg_point_differential"] == pytest.approx(-10.0)

    def test_std_point_differential(self):
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
            {"agent1_points": 30, "agent2_points": 90, "point_differential": -60},
            {"agent1_points": 60, "agent2_points": 60, "point_differential": 0},
        ]
        stats = compute_summary_statistics(results)
        import statistics
        expected = statistics.stdev([40, -60, 0])
        assert stats["std_point_differential"] == pytest.approx(expected)

    def test_upset_rate(self):
        # agent1 wins 3, agent2 wins 1 → upset rate = 1/4 * 100 = 25.0
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
            {"agent1_points": 70, "agent2_points": 50, "point_differential": 20},
            {"agent1_points": 90, "agent2_points": 30, "point_differential": 60},
            {"agent1_points": 30, "agent2_points": 90, "point_differential": -60},
        ]
        stats = compute_summary_statistics(results)
        assert stats["upset_rate"] == pytest.approx(25.0)

    def test_edge_case_all_agent1_wins(self):
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
            {"agent1_points": 70, "agent2_points": 50, "point_differential": 20},
        ]
        stats = compute_summary_statistics(results)
        assert stats["agent2_win_pct"] == 0.0
        assert stats["upset_rate"] == 0.0

    def test_edge_case_all_draws(self):
        results = [
            {"agent1_points": 60, "agent2_points": 60, "point_differential": 0},
            {"agent1_points": 60, "agent2_points": 60, "point_differential": 0},
        ]
        stats = compute_summary_statistics(results)
        assert stats["agent1_win_pct"] == 0.0
        assert stats["agent2_win_pct"] == 0.0
        assert stats["upset_rate"] == 0.0
        assert stats["std_point_differential"] == 0.0

    def test_edge_case_single_game(self):
        results = [
            {"agent1_points": 80, "agent2_points": 40, "point_differential": 40},
        ]
        stats = compute_summary_statistics(results)
        assert stats["total_games"] == 1
        assert stats["std_point_differential"] == 0.0

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="No results to analyze"):
            compute_summary_statistics([])

    def test_print_output_format(self, capsys):
        stats = {
            "agent1_wins": 7000, "agent1_losses": 2500, "agent1_draws": 500,
            "agent1_win_pct": 70.0, "agent1_loss_pct": 25.0, "agent1_draw_pct": 5.0,
            "agent2_wins": 2500, "agent2_losses": 7000, "agent2_draws": 500,
            "agent2_win_pct": 25.0, "agent2_loss_pct": 70.0, "agent2_draw_pct": 5.0,
            "avg_point_differential": 15.3,
            "std_point_differential": 22.1,
            "upset_rate": 25.0,
            "total_games": 10000,
        }
        print_summary_statistics(stats, "best_agent_1000k", "random")
        captured = capsys.readouterr()
        assert "=== Evaluation Summary ===" in captured.out
        assert "Agent 1 (best_agent_1000k): Win 70.0% (7000) | Loss 25.0% (2500) | Draw 5.0% (500)" in captured.out
        assert "Agent 2 (random): Win 25.0% (2500) | Loss 70.0% (7000) | Draw 5.0% (500)" in captured.out
        assert "Avg Point Differential: +15.3 (agent1 perspective)" in captured.out
        assert "Std Dev: 22.1" in captured.out
        assert "Upset Rate: 25.0% (minority winner)" in captured.out
        assert "Total Games: 10000" in captured.out
        assert "===========================" in captured.out


class TestSummaryStatisticsFromCsv:
    """Test compute_summary_statistics_from_csv()."""

    def test_reads_csv_correctly(self, tmp_path):
        csv_path = str(tmp_path / "best_agent_vs_random_4g_42s.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "agent1_points", "agent2_points", "first_player", "point_differential"])
            writer.writeheader()
            writer.writerow({"game_id": 0, "agent1_points": 80, "agent2_points": 40, "first_player": 0, "point_differential": 40})
            writer.writerow({"game_id": 1, "agent1_points": 30, "agent2_points": 90, "first_player": 1, "point_differential": -60})
        stats = compute_summary_statistics_from_csv(csv_path)
        assert stats["total_games"] == 2
        assert stats["agent1_wins"] == 1
        assert stats["agent2_wins"] == 1

    def test_name_parsing_from_filename(self, tmp_path, capsys):
        csv_path = str(tmp_path / "best_agent_vs_random_4g_42s.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "agent1_points", "agent2_points", "first_player", "point_differential"])
            writer.writeheader()
            writer.writerow({"game_id": 0, "agent1_points": 80, "agent2_points": 40, "first_player": 0, "point_differential": 40})
        compute_summary_statistics_from_csv(csv_path)
        captured = capsys.readouterr()
        assert "Agent 1 (best_agent)" in captured.out
        assert "Agent 2 (random)" in captured.out

    def test_explicit_name_overrides(self, tmp_path, capsys):
        csv_path = str(tmp_path / "best_agent_vs_random_4g_42s.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "agent1_points", "agent2_points", "first_player", "point_differential"])
            writer.writeheader()
            writer.writerow({"game_id": 0, "agent1_points": 80, "agent2_points": 40, "first_player": 0, "point_differential": 40})
        compute_summary_statistics_from_csv(csv_path, agent1_name="MyModel", agent2_name="Opponent")
        captured = capsys.readouterr()
        assert "Agent 1 (MyModel)" in captured.out
        assert "Agent 2 (Opponent)" in captured.out

    def test_name_parsing_fallback(self, tmp_path, capsys):
        csv_path = str(tmp_path / "results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["game_id", "agent1_points", "agent2_points", "first_player", "point_differential"])
            writer.writeheader()
            writer.writerow({"game_id": 0, "agent1_points": 80, "agent2_points": 40, "first_player": 0, "point_differential": 40})
        compute_summary_statistics_from_csv(csv_path)
        captured = capsys.readouterr()
        assert "Agent 1 (Agent 1)" in captured.out
        assert "Agent 2 (Agent 2)" in captured.out


class TestRunEvaluationPrintsStats:
    """Test that run_evaluation() prints summary statistics."""

    @patch("evaluation.evaluate.BriscasEnv")
    @patch("evaluation.evaluate.LocalAdapter")
    @patch("evaluation.evaluate.load_agent")
    def test_run_evaluation_prints_stats(self, mock_load, mock_adapter_cls, mock_env_cls, tmp_path, capsys):
        model = _mock_model()
        mock_load.return_value = (model, {})

        resets, steps = _mock_env_step_sequence(3)
        mock_env = MagicMock()
        mock_env.reset.side_effect = resets
        mock_env.step.side_effect = steps
        mock_env_cls.return_value = mock_env

        run_evaluation("models/best.zip", "random", 3, 42, output_dir=str(tmp_path))

        captured = capsys.readouterr()
        assert "=== Evaluation Summary ===" in captured.out
        assert "Agent 1 (best)" in captured.out
        assert "Agent 2 (random)" in captured.out
        assert "Total Games: 3" in captured.out


# --- Integration test ---

@pytest.mark.integration
def test_integration_train_and_evaluate(tmp_path):
    """Train a minimal model and run evaluation games against advanced strategy."""
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
        agent2="advanced",
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
