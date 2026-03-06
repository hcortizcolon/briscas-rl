"""Training module — train_agent() and WinRateCallback for DQN training."""

import datetime
import json
import logging
import os
from collections import deque

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gym_env.briscas_env import BriscasEnv
from gym_env.engine_adapter import RESTAdapter

logger = logging.getLogger(__name__)

# 5% margin below 50%; statistically significant at p < 0.001 with 1000 games
WORST_AGENT_WARNING_THRESHOLD = 0.45
VALIDATION_NUM_GAMES = 1000


class WinRateCallback(BaseCallback):
    """Track win rate over a rolling window of completed games."""

    def __init__(self, window_size: int = 1000, log_freq: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window_size = window_size
        self.log_freq = log_freq
        self._results: deque[str] = deque(maxlen=window_size)
        self.games_played: int = 0
        self.win_rate: float = 0.0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        if dones[0]:
            result = infos[0].get("game_result", "unknown")
            self._results.append(result)
            self.games_played += 1
            wins = sum(1 for r in self._results if r == "win")
            self.win_rate = wins / len(self._results)

            if self.games_played == 1:
                logger.info(
                    "First game completed at timestep %d | Result: %s",
                    self.num_timesteps,
                    result,
                )

            if self.games_played % self.log_freq == 0:
                logger.info(
                    "Win rate (last %d games): %.1f%% | Games played: %d",
                    self.window_size,
                    self.win_rate * 100,
                    self.games_played,
                )
        return True


def validate_worst_agent(model, adapter, num_games: int = 1000) -> float:
    """Run evaluation games and return the worst agent's win rate."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    env = BriscasEnv(adapter=adapter, reward_scale=1.0)
    wins = 0
    losses = 0
    draws = 0
    try:
        for _ in range(num_games):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated
            result = info.get("game_result", "unknown")
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            elif result == "draw":
                draws += 1
            else:
                logger.warning("Unexpected game_result: %s", result)
                draws += 1
    finally:
        env.close()

    win_rate = wins / num_games
    logger.info(
        "Worst agent validation: %dW / %dL / %dD over %d games | Win rate: %.1f%%",
        wins, losses, draws, num_games, win_rate * 100,
    )
    if win_rate >= WORST_AGENT_WARNING_THRESHOLD:
        logger.warning(
            "Worst agent may not be producing true anti-optimal play — "
            "win rate %.1f%% is not meaningfully below 50%%",
            win_rate * 100,
        )
    return win_rate


def load_agent(model_path: str) -> tuple[DQN, dict | None]:
    """Load a saved DQN model and its metadata from disk."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    zip_path = model_path + ".zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model file not found: {zip_path}")

    model = DQN.load(model_path)

    metadata = None
    metadata_path = model_path + ".json"
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Metadata file is corrupted at %s", metadata_path)
        if metadata is not None:
            logger.info(
                "Loaded agent from %s | Agent type: %s | Trained: %s timesteps",
                zip_path,
                metadata.get("agent_type", "unknown"),
                metadata.get("total_timesteps", "unknown"),
            )
        else:
            logger.info("Loaded agent from %s | No metadata available", zip_path)
    else:
        logger.warning("No metadata file found at %s", metadata_path)
        logger.info("Loaded agent from %s | No metadata available", zip_path)

    return model, metadata


def train_agent(
    agent_type: str,
    total_timesteps: int,
    seed: int,
    output_path: str,
    engine_url: str = "http://127.0.0.1:5000",
    checkpoint_freq: int = 10000,
    resume_from: str | None = None,
) -> None:
    """Train a DQN agent against the engine's built-in random player."""
    reward_scale = -1.0 if agent_type == "worst" else 1.0
    adapter = RESTAdapter(engine_url)
    env = BriscasEnv(adapter=adapter, reward_scale=reward_scale)

    try:
        # Pre-flight connectivity check
        env.reset()
        logger.info("Engine connection verified at %s", engine_url)

        if resume_from is not None:
            model = DQN.load(resume_from, env=env)
            logger.info("Resuming training from %s", resume_from)
        else:
            model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000, seed=seed)

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="models/checkpoints/",
            name_prefix=f"{agent_type}_agent",
        )
        winrate_cb = WinRateCallback()

        # Create output directories before training
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, winrate_cb],
            reset_num_timesteps=resume_from is None,
        )

        # Save model and metadata
        model.save(output_path)

        metadata = {
            "agent_type": agent_type,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "reward_type": "normalized_differential",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if resume_from is not None:
            metadata["resume_from"] = resume_from
        metadata_path = output_path + ".json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Training complete | Agent: %s | Timesteps: %d | Games played: %d | "
            "Win rate: %.1f%% | Model: %s | Metadata: %s",
            agent_type,
            total_timesteps,
            winrate_cb.games_played,
            winrate_cb.win_rate * 100,
            output_path + ".zip",
            metadata_path,
        )

        # Close training env before validation to avoid shared adapter issues
        env.close()

        if agent_type == "worst":
            try:
                val_win_rate = validate_worst_agent(model, adapter, num_games=VALIDATION_NUM_GAMES)
                with open(metadata_path) as f:
                    meta = json.load(f)
                meta["validation_win_rate"] = val_win_rate
                meta["validation_games"] = VALIDATION_NUM_GAMES
                with open(metadata_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning("Validation failed: %s", e)
    finally:
        env.close()
