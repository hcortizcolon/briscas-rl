"""CLI entry point for DQN agent training."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed import set_all_seeds
from training import train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Briscas.")
    parser.add_argument(
        "--agent",
        choices=["best", "worst"],
        required=True,
        help="Agent type to train.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total training timesteps. Minimum ~50k for any learning (learning_starts=1000).",
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
        help="Output path for model (no extension). Default: models/{agent}_agent_{timesteps}k",
    )
    parser.add_argument(
        "--engine-url",
        type=str,
        default="http://127.0.0.1:5000",
        help="Game engine URL.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Checkpoint save frequency in timesteps.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to an existing model (.zip optional) to continue training from.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    set_all_seeds(args.seed)

    output_path = args.output
    if output_path is None:
        output_path = f"models/{args.agent}_agent_{args.timesteps // 1000}k"

    train_agent(
        agent_type=args.agent,
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_path=output_path,
        engine_url=args.engine_url,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
