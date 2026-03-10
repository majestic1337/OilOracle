"""Command-line interface."""

import argparse

from src.core.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Bachelor Work - Trading & ML System")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("etl", help="Run ETL pipeline")
    subparsers.add_parser("train", help="Run training pipeline")
    subparsers.add_parser("backtest", help="Run backtest")
    subparsers.add_parser("run", help="Run main agent loop")

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logger = get_logger("cli")
    logger.info("Command: %s", args.command or "help")


if __name__ == "__main__":
    main()
