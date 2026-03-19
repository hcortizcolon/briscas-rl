"""Evaluation module — run matchups between agents (model vs engine or model vs model)."""

import csv
import logging
import os
import statistics

from gym_env.briscas_env import BriscasEnv
from gym_env.local_adapter import LocalAdapter
from gym_env.observation import build_observation, encode_card, sorted_hand_index
from seed import set_all_seeds
from training.train import load_agent

logger = logging.getLogger(__name__)

ENGINE_STRATEGIES = ("advanced", "random")


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

    a1_is_engine = agent1 in ENGINE_STRATEGIES
    a2_is_engine = agent2 in ENGINE_STRATEGIES

    if a1_is_engine and a2_is_engine:
        raise ValueError("At least one agent must be a trained model")

    set_all_seeds(seed)

    if a1_is_engine or a2_is_engine:
        results = _play_model_vs_engine(agent1, agent2, num_games)
    else:
        results = _play_model_vs_model(agent1, agent2, num_games)

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

    stats = compute_summary_statistics(results)
    print_summary_statistics(stats, agent1_name, agent2_name)

    return csv_path


def _play_model_vs_engine(agent1: str, agent2: str, num_games: int) -> list[dict]:
    """Play games between a trained model and an engine strategy."""
    if agent1 in ENGINE_STRATEGIES:
        model_path = agent2
        model_is_agent2 = True
        strategy = agent1
    else:
        model_path = agent1
        model_is_agent2 = False
        strategy = agent2

    model, _metadata = load_agent(model_path)
    adapter = LocalAdapter(strategy=strategy)
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
            engine_points = info["opponent_points"]

            if model_is_agent2:
                a1_points = engine_points
                a2_points = model_points
            else:
                a1_points = model_points
                a2_points = engine_points

            results.append({
                "game_id": game_id,
                "agent1_points": a1_points,
                "agent2_points": a2_points,
                "first_player": csv_first_player,
                "point_differential": a1_points - a2_points,
            })
    finally:
        env.close()

    return results


def _play_model_vs_model(agent1: str, agent2: str, num_games: int) -> list[dict]:
    """Play games between two trained models."""
    model1, _ = load_agent(agent1)
    model2, _ = load_agent(agent2)

    adapter = LocalAdapter(strategy="random")
    results = []

    for game_id in range(num_games):
        state = adapter.new_game()
        cards_seen: set[int] = set()
        first_player = 0 if state.is_your_turn else 1

        while True:
            state = adapter.get_state()
            if state.game_over:
                break

            if state.is_your_turn:
                obs = build_observation(
                    hand=state.hand, trump=state.trump, trick=state.trick,
                    cards_seen=cards_seen, deck_remaining=state.deck_remaining,
                    agent_score=state.players[0].score,
                    opponent_score=state.players[1].score,
                )
                action, _ = model1.predict(obs, deterministic=True)
                mapped = int(action.item()) % len(state.hand)
                engine_idx = sorted_hand_index(state.hand, mapped)
                play_state = adapter.play_card(engine_idx)
            else:
                opp_hand = adapter.get_opponent_hand()
                obs = build_observation(
                    hand=opp_hand, trump=state.trump, trick=state.trick,
                    cards_seen=cards_seen, deck_remaining=state.deck_remaining,
                    agent_score=state.players[1].score,
                    opponent_score=state.players[0].score,
                )
                action, _ = model2.predict(obs, deterministic=True)
                mapped = int(action.item()) % len(opp_hand)
                engine_idx = sorted_hand_index(opp_hand, mapped)
                play_state = adapter.play_opponent_card(engine_idx)

            for tc in play_state.trick:
                cards_seen.add(encode_card(tc.card))

        a1_points = state.players[0].score
        a2_points = state.players[1].score
        results.append({
            "game_id": game_id,
            "agent1_points": a1_points,
            "agent2_points": a2_points,
            "first_player": first_player,
            "point_differential": a1_points - a2_points,
        })

    adapter.delete_game()
    return results


def compute_summary_statistics(results: list[dict]) -> dict:
    """Compute win/loss/draw counts, percentages, point differential stats, and upset rate."""
    if len(results) == 0:
        raise ValueError("No results to analyze")

    total = len(results)
    a1_wins = sum(1 for r in results if r["point_differential"] > 0)
    a2_wins = sum(1 for r in results if r["point_differential"] < 0)
    draws = sum(1 for r in results if r["point_differential"] == 0)

    differentials = [r["point_differential"] for r in results]
    avg_diff = sum(differentials) / total
    std_diff = 0.0 if len(differentials) < 2 else statistics.stdev(differentials)

    minority_wins = min(a1_wins, a2_wins)
    upset_rate = minority_wins / total * 100 if (a1_wins + a2_wins) > 0 else 0.0

    return {
        "agent1_wins": a1_wins,
        "agent1_losses": a2_wins,
        "agent1_draws": draws,
        "agent1_win_pct": a1_wins / total * 100,
        "agent1_loss_pct": a2_wins / total * 100,
        "agent1_draw_pct": draws / total * 100,
        "agent2_wins": a2_wins,
        "agent2_losses": a1_wins,
        "agent2_draws": draws,
        "agent2_win_pct": a2_wins / total * 100,
        "agent2_loss_pct": a1_wins / total * 100,
        "agent2_draw_pct": draws / total * 100,
        "avg_point_differential": avg_diff,
        "std_point_differential": std_diff,
        "upset_rate": upset_rate,
        "total_games": total,
    }


def print_summary_statistics(stats: dict, agent1_name: str, agent2_name: str) -> None:
    """Print formatted summary statistics to stdout."""
    print("=== Evaluation Summary ===")
    print(
        f"Agent 1 ({agent1_name}): "
        f"Win {stats['agent1_win_pct']:.1f}% ({stats['agent1_wins']}) | "
        f"Loss {stats['agent1_loss_pct']:.1f}% ({stats['agent1_losses']}) | "
        f"Draw {stats['agent1_draw_pct']:.1f}% ({stats['agent1_draws']})"
    )
    print(
        f"Agent 2 ({agent2_name}): "
        f"Win {stats['agent2_win_pct']:.1f}% ({stats['agent2_wins']}) | "
        f"Loss {stats['agent2_loss_pct']:.1f}% ({stats['agent2_losses']}) | "
        f"Draw {stats['agent2_draw_pct']:.1f}% ({stats['agent2_draws']})"
    )
    print(f"Avg Point Differential: {stats['avg_point_differential']:+.1f} (agent1 perspective)")
    print(f"Std Dev: {stats['std_point_differential']:.1f}")
    print(f"Upset Rate: {stats['upset_rate']:.1f}% (minority winner)")
    print(f"Total Games: {stats['total_games']}")
    print("===========================")


def compute_summary_statistics_from_csv(
    csv_path: str,
    agent1_name: str | None = None,
    agent2_name: str | None = None,
) -> dict:
    """Read a CSV file and compute/print summary statistics."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        results = [
            {
                "agent1_points": int(row["agent1_points"]),
                "agent2_points": int(row["agent2_points"]),
                "point_differential": int(row["point_differential"]),
            }
            for row in reader
        ]

    if agent1_name is None or agent2_name is None:
        basename = os.path.basename(csv_path)
        name_no_ext = basename.rsplit(".", 1)[0] if "." in basename else basename
        if "_vs_" in name_no_ext:
            parts = name_no_ext.split("_vs_")
            parsed_a1 = parts[0]
            # Remove trailing _Ng_Ss suffix from agent2 name
            a2_raw = parts[1]
            # Find last occurrence of _<digits>g_ pattern to strip suffix
            tokens = a2_raw.split("_")
            cutoff = len(tokens)
            for i, tok in enumerate(tokens):
                if tok.endswith("g") and tok[:-1].isdigit():
                    cutoff = i
                    break
            parsed_a2 = "_".join(tokens[:cutoff]) if cutoff > 0 else a2_raw
            if agent1_name is None:
                agent1_name = parsed_a1
            if agent2_name is None:
                agent2_name = parsed_a2
        else:
            if agent1_name is None:
                agent1_name = "Agent 1"
            if agent2_name is None:
                agent2_name = "Agent 2"

    stats = compute_summary_statistics(results)
    print_summary_statistics(stats, agent1_name, agent2_name)
    return stats


def _extract_agent_name(agent: str) -> str:
    """Extract agent name from path or literal."""
    if agent in ("advanced", "random"):
        return agent
    name = os.path.basename(agent)
    if name.endswith(".zip"):
        name = name[:-4]
    return name
