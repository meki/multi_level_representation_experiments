from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .interactive import AGENTS, diagnostic_accuracy, generate_sequence, run_task, train_diagnostic


SELECTED_NAMES = {
    "Raw BFS",
    "Random retry",
    "Candidate score-only",
    "Score round-robin",
    "Trace-guided",
    "Oracle edit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the same revision interface with BFS and A*."
    )
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--test-seeds", type=int, default=100)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [agent for agent in AGENTS if agent.name in SELECTED_NAMES]
    records: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for planner in ("bfs", "astar"):
        model, _ = train_diagnostic(
            seeds=args.train_seeds, probe_limit=60, planner=planner
        )
        accuracy, _ = diagnostic_accuracy(
            model, seeds=args.test_seeds, probe_limit=60, planner=planner
        )
        diagnostic_rows.append(
            {"planner": planner, "diagnostic_accuracy": accuracy}
        )
        for seed in range(1_000, 1_000 + args.test_seeds):
            for phase_index, task in enumerate(generate_sequence(seed, split="test")):
                for agent in selected:
                    records.append(
                        run_task(
                            task=task,
                            agent=agent,
                            model=model,
                            seed=seed,
                            budget=args.budget,
                            phase_index=phase_index,
                            probe_limit=60,
                            trial_limit=None,
                            planner=planner,
                        )
                    )
    raw = pd.DataFrame.from_records(records)
    summary = (
        raw.groupby(["planner", "agent"], sort=False)
        .agg(
            success_rate=("success", "mean"),
            mean_total_cost=("total_cost", "mean"),
            mean_attempts=("attempts", "mean"),
            n=("success", "size"),
        )
        .reset_index()
        .merge(pd.DataFrame(diagnostic_rows), on="planner", how="left")
    )
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "planner_audit_raw.csv", index=False, lineterminator="\n")
    summary.to_csv(results_dir / "planner_audit.csv", index=False, lineterminator="\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
