from __future__ import annotations

import argparse
import json
from pathlib import Path

from .interactive import (
    AGENTS,
    diagnostic_accuracy,
    generate_sequence,
    run_task,
    train_diagnostic,
    train_linear_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit diagnostic accuracy across classifier families."
    )
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--test-seeds", type=int, default=200)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centroid, _ = train_diagnostic(seeds=args.train_seeds, probe_limit=60)
    linear = train_linear_diagnostic(seeds=args.train_seeds, probe_limit=60)
    centroid_accuracy, centroid_confusion = diagnostic_accuracy(
        centroid, seeds=args.test_seeds, probe_limit=60
    )
    linear_accuracy, linear_confusion = diagnostic_accuracy(
        linear, seeds=args.test_seeds, probe_limit=60
    )
    trace_agent = next(agent for agent in AGENTS if agent.name == "Trace-guided")
    successes = {"nearest_centroid": 0, "multinomial_logistic": 0}
    task_count = args.test_seeds * 8
    for seed in range(1_000, 1_000 + args.test_seeds):
        for phase_index, task in enumerate(generate_sequence(seed, split="test")):
            for name, model in (
                ("nearest_centroid", centroid),
                ("multinomial_logistic", linear),
            ):
                result = run_task(
                    task=task,
                    agent=trace_agent,
                    model=model,
                    seed=seed,
                    budget=500,
                    phase_index=phase_index,
                    probe_limit=60,
                    trial_limit=None,
                )
                successes[name] += int(result["success"])
    report = {
        "training_seeds": args.train_seeds,
        "held_out_seeds": args.test_seeds,
        "nearest_centroid_accuracy": centroid_accuracy,
        "multinomial_logistic_accuracy": linear_accuracy,
        "nearest_centroid_trace_guided_success": successes["nearest_centroid"]
        / task_count,
        "multinomial_logistic_trace_guided_success": successes[
            "multinomial_logistic"
        ]
        / task_count,
        "nearest_centroid_confusion": centroid_confusion.tolist(),
        "multinomial_logistic_confusion": linear_confusion.tolist(),
    }
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "classifier_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
