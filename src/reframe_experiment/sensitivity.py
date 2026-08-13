from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .interactive import (
    AGENTS,
    CentroidDiagnostic,
    generate_sequence,
    run_task,
    train_diagnostic,
)


SELECTED = {
    agent.name: agent
    for agent in AGENTS
    if agent.name in {"Random retry", "Candidate score-only", "Trace-guided"}
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coded ReFrame robustness checks.")
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--test-seeds", type=int, default=80)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []

    for probe_limit in (30, 60, 90):
        model, _ = train_diagnostic(seeds=args.train_seeds, probe_limit=probe_limit)
        records.extend(
            _evaluate(
                model=model,
                test_seeds=args.test_seeds,
                budget=args.budget,
                probe_limit=probe_limit,
                trial_limit=500,
                revision_cost_scale=1.0,
                analysis="probe_limit",
                setting=float(probe_limit),
            )
        )

    model, _ = train_diagnostic(seeds=args.train_seeds, probe_limit=60)
    for trial_limit in (140, 220, 500, 1_000):
        records.extend(
            _evaluate(
                model=model,
                test_seeds=args.test_seeds,
                budget=args.budget,
                probe_limit=60,
                trial_limit=trial_limit,
                revision_cost_scale=1.0,
                analysis="trial_limit",
                setting=float(trial_limit),
            )
        )
    for scale in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        records.extend(
            _evaluate(
                model=model,
                test_seeds=args.test_seeds,
                budget=args.budget,
                probe_limit=60,
                trial_limit=500,
                revision_cost_scale=scale,
                analysis="revision_cost_scale",
                setting=scale,
            )
        )

    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["analysis", "setting", "agent"], sort=False)
        .agg(
            success_rate=("success", "mean"),
            mean_total_cost=("total_cost", "mean"),
            diagnosis_accuracy=("diagnosis_correct", "mean"),
            first_candidate_success=("first_candidate_success", "mean"),
            mean_attempts=("attempts", "mean"),
            n=("success", "size"),
        )
        .reset_index()
    )
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "sensitivity_raw.csv", index=False, lineterminator="\n")
    summary.to_csv(results_dir / "sensitivity.csv", index=False, lineterminator="\n")
    print(summary.to_string(index=False))


def _evaluate(
    *,
    model: CentroidDiagnostic,
    test_seeds: int,
    budget: int,
    probe_limit: int,
    trial_limit: int,
    revision_cost_scale: float,
    analysis: str,
    setting: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in range(1_000, 1_000 + test_seeds):
        for phase_index, task in enumerate(generate_sequence(seed, split="test")):
            for agent in SELECTED.values():
                record = run_task(
                    task=task,
                    agent=agent,
                    model=model,
                    seed=seed,
                    budget=budget,
                    phase_index=phase_index,
                    probe_limit=probe_limit,
                    trial_limit=trial_limit,
                    revision_cost_scale=revision_cost_scale,
                )
                record["analysis"] = analysis
                record["setting"] = setting
                records.append(record)
    return records


if __name__ == "__main__":
    main()
