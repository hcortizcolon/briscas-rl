"""Evaluation module — run matchups between a trained agent and the engine's random player."""

import csv
import logging
import os

from gym_env.briscas_env import BriscasEnv
from gym_env.local_adapter import LocalAdapter
from seed import set_all_seeds
from training.train import load_agent

logger = logging.getLogger(__name__)


def run_evaluation(
    agent1: str,
    agent2: str,
    num_games: int,
    seed: int,
    output_dir: str = "results",
    output_path: str | None = None,
) -> str:
    """Play N games between two agents and write results to CSV."""
    if num_games <= 0:
        raise ValueError("num_games must be a positive integer")

    # Validate agent constraints
    agents = [agent1, agent2]
    random_count = sum(1 for a in agents if a == "random")
    if random_count == 2:
        raise ValueError("At least one agent must be a trained model")
    if random_count == 0:
        raise ValueError(
            "Exactly one agent must be 'random' — model-vs-model is not supported "
            "(engine API does not expose opponent hand)"
        )

    # Determine which agent is the model
    if agent1 == "random":
        model_path = agent2
        model_is_agent2 = True
    else:
        model_path = agent1
        model_is_agent2 = False

    model, _metadata = load_agent(model_path)

    set_all_seeds(seed)

    adapter = LocalAdapter()
    env = BriscasEnv(adapter=adapter, reward_scale=1.0)

    results = []
    try:
        for game_id in range(num_games):
            obs, reset_info = env.reset()
            model_went_first = reset_info.get("first_player", 0) == 0
            if model_is_agent2:
                csv_first_player = 1 if model_went_first else 0
            else:
                csv_first_player = 0 if model_went_first else 1

            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(int(action.item()))
                done = terminated or truncated

            model_points = info["agent_points"]
            random_points = info["opponent_points"]

            if model_is_agent2:
                a1_points = random_points
                a2_points = model_points
            else:
                a1_points = model_points
                a2_points = random_points

            results.append({
                "game_id": game_id,
                "agent1_points": a1_points,
                "agent2_points": a2_points,
                "first_player": csv_first_player,
                "point_differential": a1_points - a2_points,
            })
    finally:
        env.close()

    # Write CSV
    agent1_name = _extract_agent_name(agent1)
    agent2_name = _extract_agent_name(agent2)
    if output_path:
        csv_path = output_path
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{agent1_name}_vs_{agent2_name}_{num_games}g_{seed}s.csv"
        csv_path = os.path.join(output_dir, filename)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["game_id", "agent1_points", "agent2_points", "first_player", "point_differential"]
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info(
        "Evaluation complete: %d games | %s vs %s | Output: %s",
        num_games, agent1_name, agent2_name, csv_path,
    )
    return csv_path


def _extract_agent_name(agent: str) -> str:
    """Extract agent name from path or literal."""
    if agent == "random":
        return "random"
    name = os.path.basename(agent)
    if name.endswith(".zip"):
        name = name[:-4]
    return name
