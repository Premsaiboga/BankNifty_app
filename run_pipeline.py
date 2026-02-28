#!/usr/bin/env python3
"""
BankNifty Trading System V2 - Master Pipeline
===============================================
Run this script to build training data, train the model, and validate.

Usage:
    python run_pipeline.py build     # Build training data from historical candles
    python run_pipeline.py train     # Train the ML model
    python run_pipeline.py backtest  # Run backtest with AI filter
    python run_pipeline.py all       # Run entire pipeline (build → train → backtest)
    python run_pipeline.py live      # Start live trading (requires Zerodha connection)
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(script: str, description: str):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"\nFAILED: {description}")
        print("Fix the error above and re-run.")
        sys.exit(1)

    print(f"\nDONE: {description}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "build":
        run_step("ml/build_training_data_v2.py", "Building Training Data V2")

    elif command == "train":
        run_step("ml/train_model_v2.py", "Training ML Model V2")

    elif command == "backtest":
        run_step("backtest/backtest_v2.py", "Running Backtest V2")

    elif command == "all":
        run_step("ml/build_training_data_v2.py", "Step 1/3: Building Training Data V2")
        run_step("ml/train_model_v2.py", "Step 2/3: Training ML Model V2")
        run_step("backtest/backtest_v2.py", "Step 3/3: Running Backtest V2")

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"\nTo start live trading:")
        print(f"  python run_pipeline.py live")
        print(f"\nOr directly:")
        print(f"  python live/zerodha_stream_v2.py")

    elif command == "live":
        print(f"\n{'='*60}")
        print(f"  STARTING LIVE TRADING")
        print(f"{'='*60}")
        print(f"\nMake sure:")
        print(f"  1. Zerodha API key and access token are set in .env")
        print(f"  2. Telegram bot token and chat ID are set in .env")
        print(f"  3. Model is trained (run: python run_pipeline.py all)")
        print(f"  4. Market is open (9:15 AM - 3:30 PM IST)")
        print()

        run_step("live/zerodha_stream_v2.py", "Live Trading V2")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
