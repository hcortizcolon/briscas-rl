"""CLI entry point for agent evaluation matchups."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agents via matchup games.")
    parser.add_argument(
        "--agent1",
        required=True,
        help="Path to trained model or 'advanced' or 'random' for engine's built-in strategies.",
    )
    parser.add_argument(
        "--agent2",
        required=True,
        help="Path to trained model or 'advanced' or 'random' for engine's built-in strategies.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10000,
        help="Number of evaluation games to play.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output file path (default: results/{auto}.csv).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    csv_path = run_evaluation(
        agent1=args.agent1,
        agent2=args.agent2,
        num_games=args.games,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Results written to: {csv_path}")


if __name__ == "__main__":
    main()
