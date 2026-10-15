"""Generate data (if needed), train every model and register v1 in production.

Usage:  python scripts/train_all.py [--regenerate] [--reset]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railmind import data  # noqa: E402
from railmind.config import DATA_DIR, MODEL_DIR  # noqa: E402
from railmind.datagen import generate_all  # noqa: E402
from railmind.governance import registry  # noqa: E402
from railmind.governance.store import Store  # noqa: E402
from railmind.passenger import confirmation, delay, fare, fraud, recommender  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate synthetic data")
    ap.add_argument("--reset", action="store_true", help="wipe registry + governance DB first")
    args = ap.parse_args()

    if args.reset:
        Store.reset()
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        registry.clear_cache()
    if args.regenerate or not (DATA_DIR / "trains.csv").exists():
        print("Generating synthetic data …")
        generate_all()
        data.clear_cache()

    jobs = [("confirmation_predictor", confirmation.train), ("delay_predictor", delay.train),
            ("fare_predictor", fare.train), ("fraud_detector", fraud.train),
            ("train_recommender", recommender.train)]
    for name, fn in jobs:
        if registry.production_version(name) and not args.reset:
            print(f"  {name:<24} already in production ({registry.production_version(name)}) — skipping")
            continue
        t0 = time.time()
        version, metrics = fn()
        print(f"  {name:<24} {version}  {metrics}  ({time.time() - t0:.1f}s)")

    # GenAI / agent components are registered as versioned configurations
    from railmind.passenger.support import register_assistant
    from railmind.passenger.planner import register_agent
    register_assistant()
    register_agent()
    print("Done. Launch the app with:  streamlit run app/Home.py")


if __name__ == "__main__":
    main()
