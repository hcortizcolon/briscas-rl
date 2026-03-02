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
            terminal_info = infos[0].get("terminal_info", {})
            result = terminal_info.get("game_result", "unknown")
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


def train_agent(
    agent_type: str,
    total_timesteps: int,
    seed: int,
    output_path: str,
    engine_url: str = "http://localhost:5000",
    checkpoint_freq: int = 10000,
) -> None:
    """Train a DQN agent against the engine's built-in random player."""
    reward_scale = -1.0 if agent_type == "worst" else 1.0
    adapter = RESTAdapter(engine_url)
    env = BriscasEnv(adapter=adapter, reward_scale=reward_scale)

    try:
        # Pre-flight connectivity check
        env.reset()
        logger.info("Engine connection verified at %s", engine_url)

        model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000, seed=seed)

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path="models/checkpoints/",
            name_prefix=f"{agent_type}_agent",
        )
        winrate_cb = WinRateCallback()

        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, winrate_cb])

        # Save model and metadata
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs("models/checkpoints/", exist_ok=True)
        model.save(output_path)

        metadata = {
            "agent_type": agent_type,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "reward_type": "normalized_differential",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
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
    finally:
        env.close()
