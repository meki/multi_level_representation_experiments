from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .analysis import summarize
from .interactive import (
    AGENTS,
    diagnostic_accuracy,
    generate_sequence,
    run_task,
    train_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the held-out interactive ReFrame benchmark."
    )
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--test-seeds", type=int, default=200)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000]
    )
    parser.add_argument("--main-budget", type=int, default=500)
    parser.add_argument("--probe-limit", type=int, default=60)
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=0,
        help="Per-candidate verification cap; 0 uses the entire remaining budget.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.main_budget not in args.budgets:
        raise SystemExit("--main-budget must also be present in --budgets")
    started = time.time()
    model, training_metadata = train_diagnostic(
        seeds=args.train_seeds, probe_limit=args.probe_limit
    )
    held_out_accuracy, confusion = diagnostic_accuracy(
        model, seeds=args.test_seeds, probe_limit=args.probe_limit
    )

    records: list[dict[str, object]] = []
    for budget in args.budgets:
        for seed in range(1_000, 1_000 + args.test_seeds):
            tasks = generate_sequence(seed, split="test")
            for phase_index, task in enumerate(tasks):
                for agent in AGENTS:
                    records.append(
                        run_task(
                            task=task,
                            agent=agent,
                            model=model,
                            seed=seed,
                            budget=budget,
                            phase_index=phase_index,
                            probe_limit=args.probe_limit,
                            trial_limit=args.trial_limit or None,
                        )
                    )

    raw = pd.DataFrame.from_records(records)
    results_dir = args.output_root / "results"
    figures_dir = args.output_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False, lineterminator="\n")
    summaries = summarize(
        raw,
        results_dir,
        figures_dir,
        main_budget=args.main_budget,
        bootstrap_samples=args.bootstrap_samples,
        confusion=confusion,
    )
    metadata = {
        "study_type": "held-out interactive symbolic benchmark with concrete BFS",
        **training_metadata,
        "test_seeds": args.test_seeds,
        "test_seed_range": [1_000, 999 + args.test_seeds],
        "task_families": 8,
        "tasks_per_budget": args.test_seeds * 8,
        "agents": len(AGENTS),
        "budgets": args.budgets,
        "main_budget": args.main_budget,
        "probe_limit": args.probe_limit,
        "candidate_trial_limit": args.trial_limit or "remaining_budget",
        "candidates_per_edit_class": 4,
        "diagnostic_feature_count": 15,
        "diagnostic_uses_candidate_features": False,
        "held_out_diagnostic_accuracy": held_out_accuracy,
        "bootstrap_samples": args.bootstrap_samples,
        "rows": len(raw),
        "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "duration_seconds": round(time.time() - started, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "matplotlib": version("matplotlib"),
        },
    }
    (results_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(summaries["overall"].to_string(index=False))
    print(f"\nHeld-out diagnostic accuracy: {held_out_accuracy:.3f}")
    print(f"Wrote {len(raw):,} observations to {results_dir}")


if __name__ == "__main__":
    main()
