from __future__ import annotations

import hashlib
import json
import inspect
from pathlib import Path

import pandas as pd

from reframe_experiment.interactive import (
    AGENTS,
    CAUSES,
    astar,
    bfs,
    diagnostic_accuracy,
    generate_sequence,
    generate_task,
    repair_signature,
    run_task,
    train_diagnostic,
    trace_features,
    train_linear_diagnostic,
)


def test_sequence_is_randomized_but_complete() -> None:
    sequence = generate_sequence(0, split="test")
    assert sorted(task.cause for task in sequence) == sorted(CAUSES)
    assert [task.cause for task in sequence] != list(CAUSES)


def test_surface_labels_change_across_splits() -> None:
    train = generate_task(seed=3, phase_index=1, cause="compose", split="train")
    test = generate_task(seed=3, phase_index=1, cause="compose", split="test")
    train_actions = {action for edges in train.adjacency for action, _ in edges}
    test_actions = {action for edges in test.adjacency for action, _ in edges}
    assert train_actions.isdisjoint(test_actions)


def test_generator_target_changes_real_bfs_outcome() -> None:
    for cause in CAUSES[1:]:
        task = generate_task(seed=11, phase_index=0, cause=cause, split="test")
        candidate = next(
            candidate
            for candidate in task.candidates
            if candidate.target == task.correct_target
        )
        repaired = bfs(task, limit=5_000, candidate=candidate)
        assert repaired.solved, cause
        if cause in {"add", "split", "relate"}:
            assert not bfs(task, limit=5_000).solved, cause


def test_functional_acceptance_has_multiple_repairs() -> None:
    for cause in CAUSES[1:]:
        task = generate_task(seed=17, phase_index=2, cause=cause, split="test")
        assert len(task.accepted_targets) >= 3, cause
        accepted = [
            candidate
            for candidate in task.candidates
            if candidate.target in task.accepted_targets
        ]
        assert len({candidate.edit_class for candidate in accepted}) >= 2, cause
        assert len({repair_signature(candidate) for candidate in accepted}) >= 3, cause
        for target in task.accepted_targets:
            candidate = next(item for item in task.candidates if item.target == target)
            assert bfs(task, limit=max(5_000, task.size_scale * 2), candidate=candidate).solved


def test_trace_feature_api_excludes_task_and_candidates() -> None:
    task = generate_task(seed=19, phase_index=1, cause="split", split="test")
    trace = bfs(task, limit=60)
    assert tuple(inspect.signature(trace_features).parameters) == ("trace", "size_scale")
    features = trace_features(trace, size_scale=task.size_scale)
    assert features.shape == (15,)


def test_score_only_never_prioritizes_unscored_search() -> None:
    model, _ = train_diagnostic(seeds=12)
    agent = next(agent for agent in AGENTS if agent.name == "Candidate score-only")
    for cause in ("merge", "delete", "compose"):
        task = generate_task(seed=29, phase_index=0, cause=cause, split="test")
        result = run_task(
            task=task,
            agent=agent,
            model=model,
            seed=29,
            budget=500,
            phase_index=0,
        )
        assert result["first_target"] not in {"none", "continue_search"}, cause


def test_unaffordable_edits_do_not_block_score_only_search() -> None:
    model, _ = train_diagnostic(seeds=12)
    agent = next(agent for agent in AGENTS if agent.name == "Candidate score-only")
    task = generate_task(seed=1_000, phase_index=0, cause="search", split="test")
    result = run_task(
        task=task,
        agent=agent,
        model=model,
        seed=1_000,
        budget=500,
        phase_index=0,
        revision_cost_scale=100.0,
    )
    assert result["success"]
    assert result["first_target"] == "continue_search"
    assert result["edit_application_cost"] == 0


def test_repairs_transfer_to_astar() -> None:
    for cause in CAUSES[1:]:
        task = generate_task(seed=23, phase_index=0, cause=cause, split="test")
        candidate = next(
            item for item in task.candidates if item.target == task.correct_target
        )
        assert astar(task, limit=5_000, candidate=candidate).solved, cause


def test_runs_are_deterministic_and_budgeted() -> None:
    model, _ = train_diagnostic(seeds=12)
    agent = next(agent for agent in AGENTS if agent.name == "Trace-guided")
    task = generate_sequence(9, split="test")[0]
    kwargs = dict(
        task=task,
        agent=agent,
        model=model,
        seed=9,
        budget=500,
        phase_index=0,
    )
    first = run_task(**kwargs)
    second = run_task(**kwargs)
    assert first == second
    assert first["total_cost"] <= 500
    assert first["total_cost"] == sum(
        first[key]
        for key in (
            "probe_cost",
            "verification_cost",
            "candidate_generation_cost",
            "edit_application_cost",
        )
    )


def test_diagnostic_generalizes_above_chance() -> None:
    model, _ = train_diagnostic(seeds=20)
    accuracy, confusion = diagnostic_accuracy(model, seeds=20)
    assert confusion.shape == (len(CAUSES), len(CAUSES))
    assert accuracy > 0.60


def test_linear_diagnostic_generalizes_above_chance() -> None:
    model = train_linear_diagnostic(seeds=20)
    accuracy, confusion = diagnostic_accuracy(model, seeds=20)
    assert confusion.shape == (len(CAUSES), len(CAUSES))
    assert accuracy > 0.60


def test_committed_headline_table_matches_raw_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "results" / "raw_results.csv"
    table_path = root / "results" / "overall_main_budget.csv"
    if not raw_path.exists() or not table_path.exists():
        return
    raw = pd.read_csv(raw_path)
    table = pd.read_csv(table_path).set_index("agent")
    main = raw.loc[raw["budget"] == 500]
    observed = main.groupby("agent")["success"].mean()
    for agent, value in observed.items():
        assert abs(value - table.loc[agent, "success_rate"]) < 1e-12

    expected = {
        "Raw BFS": 0.25,
        "Fixed coarse": 0.42,
        "Add-only": 0.3975,
        "State revision": 0.763125,
        "Random retry": 0.44625,
        "Trace + random target": 0.845625,
        "Candidate score-only": 0.77875,
        "Score round-robin": 0.625625,
        "Trace-guided": 0.98,
        "Generator class": 0.994375,
        "Oracle edit": 1.0,
    }
    if set(expected) != set(table.index):
        return
    for agent, value in expected.items():
        assert abs(table.loc[agent, "success_rate"] - value) < 1e-12

    metadata = json.loads((root / "results" / "experiment_metadata.json").read_text())
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert digest == metadata["raw_results_sha256"]
