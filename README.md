# ReFrame Experiments

[![CI](https://github.com/meki/multi_level_representation_experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/meki/multi_level_representation_experiments/actions/workflows/ci.yml)

This repository is the executable artifact for a controlled study of
multilevel representation revision in symbolic planning. It contains the
benchmark implementation, locked Python environment, tests, complete raw
results, derived summaries, and figures.

The study is a synthetic mechanism benchmark. It measures whether trace-guided
selection of representation edits improves resource-bounded planning on the
specified generated transition-system family; it does not establish
generalization to external planning domains or human cognition.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From the
repository root, run:

```console
uv sync --frozen
uv run pytest
uv run reframe-experiment
uv run reframe-sensitivity
uv run reframe-planner-audit
uv run reframe-adversarial-audit
uv run reframe-classifier-audit
```

The five experiment and audit commands deterministically overwrite `results/`
and `figures/`. The full main experiment trains the diagnostic model on 640
tasks (80 seeds by eight families), then evaluates 1,600 held-out tasks at each
of five budgets. The primary budget is 500 cost units. The sensitivity run uses
80 held-out seeds.

For a quick smoke run without changing committed artifacts, choose another
output directory:

```console
uv run reframe-experiment --train-seeds 8 --test-seeds 8 --budgets 500 --main-budget 500 --bootstrap-samples 200 --output-root .smoke
```

## Benchmark and costs

Each generated task is a labelled finite transition graph solved by concrete
BFS or A*. The eight task families are `Search`, `Add`, `Merge`, `Split`,
`Relate`, `Compose`, `Delete`, and `Relevel`. Surface labels are permuted
independently between training and test splits. Training uses seeds 0--79 and
evaluation uses seeds 1000--1199. Every edit class supplies four targets. All
candidates are executed under an independent validation limit; every
non-search task admits at least three distinct successful repairs from at least
two edit classes.

All systems share the same budget. Charged costs are probe-search expansions,
candidate generation (3 units per candidate), class-dependent edit application,
and verification-search expansions. In the main run, a candidate may use the
entire remaining budget; failed trials remain charged. `interactive.py` is the
authoritative implementation.

## Conditions

- **Raw BFS:** no representation edit.
- **Fixed coarse:** a single fixed state aggregation.
- **Add-only:** may only introduce unary distinctions.
- **State revision:** retries state-side edits but cannot compose actions.
- **Random retry:** shuffles all 29 actions and retries within budget.
- **Trace + random target:** diagnoses a class but randomizes targets within it.
- **Candidate score-only:** omits diagnosis, ranks all 28 candidates directly,
  and tries unscored continued search only after the scored edits.
- **Score round-robin:** preserves within-class score order but interleaves the
  seven edit classes without diagnosis; continued search is last.
- **Trace-guided:** learned class diagnosis plus trace-derived target ranking.
- **Generator class:** receives the construction class but must rank targets.
- **Oracle edit:** receives the exact edit and target.

The learned diagnosis uses 15 candidate-blind statistics derived from a
60-expansion trace and a nearest-centroid classifier. It cannot inspect
candidate partitions, macros, edit operators, target identifiers, or target
scores. The feature function accepts only a planner trace and a graph-size
scalar; a regression test guards this information barrier.

## Principal archived result

At the pre-specified 500-unit budget, Trace-guided succeeds on 0.980 of the
1,600 held-out tasks, compared with 0.779 for Candidate score-only, 0.626 for
Score round-robin, 0.446 for Random retry, and 0.250 for Raw BFS. The paired
Trace-guided minus Candidate score-only difference is 0.201 (95% paired
bootstrap interval 0.189--0.213). These values are descriptive of the fixed
benchmark population and experimental protocol in this release.

## Outputs

- `results/raw_results.csv`: all 88,000 paired task-agent-budget records.
- `results/overall_main_budget.csv`: agent aggregates at budget 500.
- `results/summary_main_budget.csv`: task-family aggregates at budget 500.
- `results/paired_comparisons.csv`: paired bootstrap contrasts.
- `results/diagnostic_confusion.csv`: held-out diagnostic confusion matrix.
- `results/budget_curve.csv`: success over all five budgets.
- `results/sensitivity.csv`: coded robustness summaries.
- `results/planner_audit.csv`: BFS/A* comparison on 100 held-out seeds.
- `results/adversarial_audit.json`: information-barrier and repair-multiplicity audit.
- `results/classifier_audit.json`: held-out nearest-centroid and multinomial-logistic comparison.
- `results/experiment_metadata.json`: configuration, versions, and raw checksum.
- `results/table_main.tex`: generated LaTeX table.
- `figures/`: PDF and PNG figures used for analysis and visual inspection.

The SHA-256 checksum of the archived `results/raw_results.csv` is
`a5f60b0a0e6ce732ca53998e42732dfea1937de4334d78d7419b6cdc1398521d`.

## Scope, citation, and license

The generator, edit language, and cost model are deliberately explicit. The
main inference population is the generated transition-system family; the A*
audit tests planner transfer, while sensory grounding and open-ended edit
invention are outside the measured construct.

Please cite the archived release described in `CITATION.cff`. The code and
artifact are released under the MIT License; see `LICENSE`.
