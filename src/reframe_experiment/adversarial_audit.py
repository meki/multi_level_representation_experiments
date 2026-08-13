from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .interactive import (
    bfs,
    generate_sequence,
    repair_signature,
    trace_features,
    train_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate leakage and functional-answer multiplicity."
    )
    parser.add_argument("--test-seeds", type=int, default=200)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = train_diagnostic(seeds=80, probe_limit=60)
    task_count = 0
    candidate_counts: list[int] = []
    class_counts: list[int] = []
    unique_repair_counts: list[int] = []
    accepted_counts_by_family: dict[str, list[int]] = {}
    class_counts_by_family: dict[str, list[int]] = {}
    unique_counts_by_family: dict[str, list[int]] = {}
    for seed in range(1_000, 1_000 + args.test_seeds):
        for task in generate_sequence(seed, split="test"):
            trace = bfs(task, limit=60)
            features = trace_features(trace, size_scale=task.size_scale)
            model.ranked_causes(features)[0]
            task_count += 1
            if task.cause != "search":
                accepted = [
                    candidate
                    for candidate in task.candidates
                    if candidate.target in task.accepted_targets
                ]
                candidate_count = len(accepted)
                class_count = len({candidate.edit_class for candidate in accepted})
                unique_count = len(
                    {repair_signature(candidate) for candidate in accepted}
                )
                candidate_counts.append(candidate_count)
                class_counts.append(class_count)
                unique_repair_counts.append(unique_count)
                accepted_counts_by_family.setdefault(task.cause, []).append(candidate_count)
                class_counts_by_family.setdefault(task.cause, []).append(class_count)
                unique_counts_by_family.setdefault(task.cause, []).append(unique_count)
    report = {
        "tasks": task_count,
        "candidate_blindness_guarantee": (
            "trace_features accepts only SearchResult and size_scale; candidate "
            "collections, identifiers, classes, operators, and scores are absent "
            "from its callable interface"
        ),
        "regression_test": "test_trace_feature_api_excludes_task_and_candidates",
        "non_search_tasks": len(candidate_counts),
        "accepted_candidates_min": min(candidate_counts),
        "accepted_candidates_mean": float(np.mean(candidate_counts)),
        "accepted_candidates_max": max(candidate_counts),
        "accepted_classes_min": min(class_counts),
        "accepted_classes_mean": float(np.mean(class_counts)),
        "accepted_classes_max": max(class_counts),
        "accepted_unique_repairs_min": min(unique_repair_counts),
        "accepted_unique_repairs_mean": float(np.mean(unique_repair_counts)),
        "accepted_unique_repairs_max": max(unique_repair_counts),
        "by_family": {
            family: {
                "accepted_candidates_min": min(accepted_counts_by_family[family]),
                "accepted_candidates_mean": float(
                    np.mean(accepted_counts_by_family[family])
                ),
                "accepted_candidates_max": max(accepted_counts_by_family[family]),
                "accepted_classes_min": min(class_counts_by_family[family]),
                "accepted_classes_mean": float(np.mean(class_counts_by_family[family])),
                "accepted_classes_max": max(class_counts_by_family[family]),
                "accepted_unique_repairs_min": min(unique_counts_by_family[family]),
                "accepted_unique_repairs_mean": float(
                    np.mean(unique_counts_by_family[family])
                ),
                "accepted_unique_repairs_max": max(unique_counts_by_family[family]),
            }
            for family in sorted(accepted_counts_by_family)
        },
    }
    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "adversarial_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
