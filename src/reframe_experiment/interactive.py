from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
import heapq
from typing import Literal

import numpy as np


EditClass = Literal[
    "add", "merge", "split", "relate", "compose", "delete", "relevel"
]
Cause = Literal[
    "search", "add", "merge", "split", "relate", "compose", "delete", "relevel"
]

EDIT_CLASSES: tuple[EditClass, ...] = (
    "add",
    "merge",
    "split",
    "relate",
    "compose",
    "delete",
    "relevel",
)
CAUSES: tuple[Cause, ...] = ("search",) + EDIT_CLASSES
APPLICATION_COST: dict[EditClass, int] = {
    "add": 12,
    "merge": 10,
    "split": 14,
    "relate": 18,
    "compose": 20,
    "delete": 10,
    "relevel": 12,
}
CANDIDATE_GENERATION_COST = 3


@dataclass(frozen=True)
class EditCandidate:
    edit_class: EditClass
    target: str
    partition: tuple[int, ...] | None = None
    macro: tuple[str, ...] | None = None
    operator: str = ""


@dataclass(frozen=True)
class SearchTask:
    task_id: str
    cause: Cause
    adjacency: tuple[tuple[tuple[str, int], ...], ...]
    start: int
    goal: int
    base_partition: tuple[int, ...]
    fixed_partition: tuple[int, ...]
    signatures: tuple[int, ...]
    candidates: tuple[EditCandidate, ...]
    correct_target: str | None
    accepted_targets: frozenset[str]
    size_scale: int


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    expansions: int
    generated: int
    duplicates: int
    alias_conflicts: int
    max_depth: int
    frontier: int
    expanded_states: tuple[int, ...]
    encountered_states: tuple[int, ...]
    conflict_depths: tuple[int, ...]
    branching_counts: tuple[int, ...]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    mode: Literal[
        "raw",
        "fixed",
        "random",
        "trace",
        "trace_random_target",
        "score_only",
        "score_round_robin",
        "oracle_class",
        "oracle_edit",
    ]
    capabilities: frozenset[EditClass]


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec("Raw BFS", "raw", frozenset()),
    AgentSpec("Fixed coarse", "fixed", frozenset()),
    AgentSpec("Add-only", "trace", frozenset({"add"})),
    AgentSpec(
        "State revision",
        "trace",
        frozenset({"add", "merge", "split", "delete", "relevel"}),
    ),
    AgentSpec(
        "Random retry",
        "random",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Trace + random target",
        "trace_random_target",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Candidate score-only",
        "score_only",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Score round-robin",
        "score_round_robin",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Trace-guided",
        "trace",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Generator class",
        "oracle_class",
        frozenset(EDIT_CLASSES),
    ),
    AgentSpec(
        "Oracle edit",
        "oracle_edit",
        frozenset(EDIT_CLASSES),
    ),
)


def bfs(
    task: SearchTask,
    *,
    limit: int,
    candidate: EditCandidate | None = None,
    fixed: bool = False,
) -> SearchResult:
    """Run a concrete breadth-first planner with representation-based duplicate detection."""
    if limit <= 0:
        return SearchResult(False, 0, 0, 0, 0, 0, 0, (), (), (), ())
    partition = (
        task.fixed_partition
        if fixed
        else candidate.partition
        if candidate is not None and candidate.partition is not None
        else task.base_partition
    )
    macro = candidate.macro if candidate is not None else None
    queue: deque[tuple[int, int]] = deque([(task.start, 0)])
    representative: dict[int, int] = {partition[task.start]: task.start}
    expanded: list[int] = []
    encountered: list[int] = [task.start]
    generated = 0
    duplicates = 0
    conflicts = 0
    conflict_depths: list[int] = []
    branching_counts: list[int] = []
    max_depth = 0

    while queue and len(expanded) < limit:
        state, depth = queue.popleft()
        expanded.append(state)
        max_depth = max(max_depth, depth)
        if state == task.goal:
            return SearchResult(
                True,
                len(expanded),
                generated,
                duplicates,
                conflicts,
                max_depth,
                len(queue),
                tuple(expanded),
                tuple(encountered),
                tuple(conflict_depths),
                tuple(branching_counts),
            )

        successors: list[int] = []
        if macro is not None:
            macro_state = _apply_macro(task, state, macro)
            if macro_state is not None:
                successors.append(macro_state)
        successors.extend(destination for _, destination in task.adjacency[state])
        branching_counts.append(len(successors))

        for successor in successors:
            generated += 1
            encountered.append(successor)
            key = partition[successor]
            if key in representative:
                duplicates += 1
                if task.signatures[representative[key]] != task.signatures[successor]:
                    conflicts += 1
                    conflict_depths.append(depth + 1)
                continue
            representative[key] = successor
            queue.append((successor, depth + 1))

    return SearchResult(
        False,
        len(expanded),
        generated,
        duplicates,
        conflicts,
        max_depth,
        len(queue),
        tuple(expanded),
        tuple(encountered),
        tuple(conflict_depths),
        tuple(branching_counts),
    )


def _apply_macro(
    task: SearchTask, state: int, actions: tuple[str, ...]
) -> int | None:
    current = state
    for action in actions:
        transition = next(
            (destination for label, destination in task.adjacency[current] if label == action),
            None,
        )
        if transition is None:
            return None
        current = transition
    return current


def astar(
    task: SearchTask,
    *,
    limit: int,
    candidate: EditCandidate | None = None,
    fixed: bool = False,
) -> SearchResult:
    """Run A* with an admissible reverse-distance heuristic and the same representation keys."""
    if limit <= 0:
        return SearchResult(False, 0, 0, 0, 0, 0, 0, (), (), (), ())
    partition = (
        task.fixed_partition
        if fixed
        else candidate.partition
        if candidate is not None and candidate.partition is not None
        else task.base_partition
    )
    macro = candidate.macro if candidate is not None else None
    reverse: list[list[int]] = [[] for _ in task.adjacency]
    for source, edges in enumerate(task.adjacency):
        for _, destination in edges:
            reverse[destination].append(source)
    distance = [len(task.adjacency) + 1] * len(task.adjacency)
    distance[task.goal] = 0
    queue: deque[int] = deque([task.goal])
    while queue:
        state = queue.popleft()
        for predecessor in reverse[state]:
            if distance[predecessor] > distance[state] + 1:
                distance[predecessor] = distance[state] + 1
                queue.append(predecessor)

    frontier: list[tuple[int, int, int, int]] = [(distance[task.start], 0, 0, task.start)]
    representative: dict[int, int] = {partition[task.start]: task.start}
    expanded: list[int] = []
    encountered: list[int] = [task.start]
    conflict_depths: list[int] = []
    branching_counts: list[int] = []
    generated = duplicates = conflicts = tie = 0
    max_depth = 0
    while frontier and len(expanded) < limit:
        _, depth, _, state = heapq.heappop(frontier)
        expanded.append(state)
        max_depth = max(max_depth, depth)
        if state == task.goal:
            return SearchResult(
                True,
                len(expanded),
                generated,
                duplicates,
                conflicts,
                max_depth,
                len(frontier),
                tuple(expanded),
                tuple(encountered),
                tuple(conflict_depths),
                tuple(branching_counts),
            )
        successors: list[int] = []
        if macro is not None:
            macro_state = _apply_macro(task, state, macro)
            if macro_state is not None:
                successors.append(macro_state)
        successors.extend(destination for _, destination in task.adjacency[state])
        branching_counts.append(len(successors))
        for successor in successors:
            generated += 1
            encountered.append(successor)
            key = partition[successor]
            if key in representative:
                duplicates += 1
                if task.signatures[representative[key]] != task.signatures[successor]:
                    conflicts += 1
                    conflict_depths.append(depth + 1)
                continue
            representative[key] = successor
            tie += 1
            next_depth = depth + 1
            heapq.heappush(
                frontier,
                (next_depth + distance[successor], next_depth, tie, successor),
            )
    return SearchResult(
        False,
        len(expanded),
        generated,
        duplicates,
        conflicts,
        max_depth,
        len(frontier),
        tuple(expanded),
        tuple(encountered),
        tuple(conflict_depths),
        tuple(branching_counts),
    )


def plan(
    task: SearchTask,
    *,
    limit: int,
    candidate: EditCandidate | None = None,
    fixed: bool = False,
    planner: Literal["bfs", "astar"] = "bfs",
) -> SearchResult:
    search = bfs if planner == "bfs" else astar
    return search(task, limit=limit, candidate=candidate, fixed=fixed)


def generate_sequence(seed: int, *, split: Literal["train", "test"]) -> list[SearchTask]:
    """Generate a label-permuted eight-task sequence in randomized order."""
    offset = 0 if split == "train" else 1_000_000
    rng = np.random.default_rng(np.random.SeedSequence([seed, offset, 20260812]))
    causes = list(CAUSES)
    rng.shuffle(causes)
    return [
        generate_task(seed=seed, phase_index=index, cause=cause, split=split)
        for index, cause in enumerate(causes)
    ]


def generate_task(
    *,
    seed: int,
    phase_index: int,
    cause: Cause,
    split: Literal["train", "test"],
) -> SearchTask:
    offset = 0 if split == "train" else 1_000_000
    cause_index = CAUSES.index(cause)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, phase_index, cause_index, offset, 7717])
    )
    if cause == "compose":
        depth = int(rng.integers(17, 22) if split == "train" else rng.integers(20, 26))
        branches = int(rng.integers(4, 6))
        decoy_length = int(rng.integers(5, 8))
    elif cause == "search":
        depth = int(rng.integers(10, 13) if split == "train" else rng.integers(12, 15))
        branches = int(rng.integers(2, 4))
        decoy_length = int(rng.integers(3, 5))
    elif cause == "relevel":
        depth = int(rng.integers(18, 22) if split == "train" else rng.integers(20, 24))
        branches = int(rng.integers(8, 11))
        decoy_length = int(rng.integers(8, 11))
    elif cause in {"merge", "delete"}:
        depth = int(rng.integers(13, 18) if split == "train" else rng.integers(15, 20))
        branches = int(rng.integers(6, 9))
        decoy_length = int(
            rng.integers(3, 6) if cause == "delete" else rng.integers(7, 10)
        )
    else:
        depth = int(rng.integers(14, 18) if split == "train" else rng.integers(16, 20))
        branches = int(rng.integers(4, 6))
        decoy_length = int(rng.integers(5, 8))

    graph = _build_graph(rng, depth=depth, branches=branches, decoy_length=decoy_length)
    adjacency, path, decoys, signatures, action_labels = graph
    adjacency = _augment_topology(
        adjacency, path, decoys, cause=cause, rng=rng
    )
    node_count = len(adjacency)
    identity = tuple(range(node_count))
    safe = _safe_partition(path, decoys, node_count)
    dangerous = _dangerous_partition(safe, path, decoys, rng, cause=cause)

    if cause in {"add", "split", "relate"}:
        base_partition = dangerous
        fixed_partition = dangerous
        correct_partition = safe
    elif cause in {"merge", "delete"}:
        base_partition = identity
        fixed_partition = safe
        correct_partition = safe
    elif cause == "relevel":
        base_partition = dangerous
        fixed_partition = dangerous
        correct_partition = safe
    elif cause == "compose":
        base_partition = identity
        fixed_partition = identity
        correct_partition = None
    else:
        base_partition = identity
        fixed_partition = safe
        correct_partition = None

    candidates: list[EditCandidate] = []
    correct_target: str | None = None
    for edit_class in EDIT_CLASSES:
        class_candidates: list[EditCandidate] = []
        for target_index in range(4):
            target = _opaque_label(rng, prefix="x")
            if edit_class == "compose":
                macro = _decoy_macro(action_labels, target_index)
                candidate = EditCandidate(
                    edit_class,
                    target,
                    macro=macro,
                    operator="compose_action_sequence",
                )
            else:
                partition = _decoy_partition(
                    base_partition, path, decoys, node_count, target_index, rng
                )
                candidate = EditCandidate(
                    edit_class,
                    target,
                    partition=partition,
                    operator=_operator_name(edit_class),
                )
            class_candidates.append(candidate)

        if cause == edit_class:
            correct_indices = rng.choice(len(class_candidates), size=2, replace=False)
            correct_index = int(correct_indices[0])
            target = class_candidates[correct_index].target
            correct_target = target
            if edit_class == "compose":
                stride = depth
                correct = EditCandidate(
                    edit_class,
                    target,
                    macro=tuple([action_labels[0]] * stride),
                    operator="compose_action_sequence",
                )
            else:
                assert correct_partition is not None
                correct = EditCandidate(
                    edit_class,
                    target,
                    partition=correct_partition,
                    operator=_operator_name(edit_class),
                )
            class_candidates[correct_index] = correct
            alternate_index = int(correct_indices[1])
            alternate_target = class_candidates[alternate_index].target
            if edit_class == "compose":
                alternate = EditCandidate(
                    edit_class,
                    alternate_target,
                    macro=tuple([action_labels[0]] * max(1, depth - 1)),
                    operator="compose_action_sequence",
                )
            else:
                assert correct_partition is not None
                alternate_partition = list(correct_partition)
                if decoys:
                    for node in next(iter(decoys.values())):
                        alternate_partition[node] = node
                alternate = EditCandidate(
                    edit_class,
                    alternate_target,
                    partition=tuple(alternate_partition),
                    operator=_operator_name(edit_class),
                )
            class_candidates[alternate_index] = alternate
        candidates.extend(class_candidates)

    # At least one counterfactual edit class receives a functionally equivalent
    # but structurally distinct repair. The benchmark therefore accepts
    # executions, not generator labels or duplicated partition vectors.
    cross_class: dict[EditClass, EditClass] = {
        "add": "split",
        "merge": "delete",
        "split": "add",
        "relate": "split",
        "delete": "merge",
        "relevel": "split" if base_partition == dangerous else "merge",
        "compose": "compose",
    }
    if cause != "search":
        assert cause in EDIT_CLASSES
        alternate_class = cross_class[cause]
        if alternate_class != cause and correct_partition is not None:
            replace_index = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.edit_class == alternate_class
            )
            old = candidates[replace_index]
            cross_partition = _refine_safe_partition(
                correct_partition,
                decoys,
                variant=1,
            )
            assert cross_partition != correct_partition
            candidates[replace_index] = EditCandidate(
                alternate_class,
                old.target,
                partition=cross_partition,
                operator=_operator_name(alternate_class),
            )

    rng.shuffle(candidates)
    task = SearchTask(
        task_id=f"{split}-{seed}-{phase_index}-{cause}",
        cause=cause,
        adjacency=adjacency,
        start=path[0],
        goal=path[-1],
        base_partition=base_partition,
        fixed_partition=fixed_partition,
        signatures=signatures,
        candidates=tuple(candidates),
        correct_target=correct_target,
        accepted_targets=frozenset(),
        size_scale=node_count,
    )
    acceptance_limit = max(5_000, node_count * 2)
    accepted = frozenset(
        candidate.target
        for candidate in task.candidates
        if bfs(task, limit=acceptance_limit, candidate=candidate).solved
    )
    return replace(task, accepted_targets=accepted)


def _refine_safe_partition(
    partition: tuple[int, ...],
    decoys: dict[tuple[int, int], tuple[int, ...]],
    *,
    variant: int,
) -> tuple[int, ...]:
    """Return a strict refinement that preserves every concrete path state."""
    refined = list(partition)
    keys = sorted(decoys)
    nodes = decoys[keys[variant % len(keys)]]
    fresh = max(refined) + 1
    for offset, node in enumerate(nodes):
        refined[node] = fresh + offset
    return tuple(refined)


def _build_graph(
    rng: np.random.Generator,
    *,
    depth: int,
    branches: int,
    decoy_length: int,
) -> tuple[
    tuple[tuple[tuple[str, int], ...], ...],
    tuple[int, ...],
    dict[tuple[int, int], tuple[int, ...]],
    tuple[int, ...],
    tuple[str, ...],
]:
    labels = [_opaque_label(rng, prefix="a") for _ in range(branches + 2)]
    forward = labels[0]
    path = list(range(depth + 1))
    adjacency: list[list[tuple[str, int]]] = [[] for _ in path]
    signatures: list[int] = [10_000 + index for index in path]
    decoys: dict[tuple[int, int], tuple[int, ...]] = {}

    for position in range(depth):
        chains: list[tuple[int, ...]] = []
        for branch in range(branches):
            chain: list[int] = []
            previous: int | None = None
            for step in range(decoy_length):
                node = len(adjacency)
                adjacency.append([])
                signatures.append(100 + step)
                chain.append(node)
                if previous is not None:
                    adjacency[previous].append((labels[branch + 1], node))
                previous = node
            chains.append(tuple(chain))
            adjacency[path[position]].append((labels[branch + 1], chain[0]))
        adjacency[path[position]].append((forward, path[position + 1]))
        for step in range(decoy_length):
            decoys[(position, step)] = tuple(chain[step] for chain in chains)
    return (
        tuple(tuple(edges) for edges in adjacency),
        tuple(path),
        decoys,
        tuple(signatures),
        tuple(labels),
    )


def _safe_partition(
    path: tuple[int, ...],
    decoys: dict[tuple[int, int], tuple[int, ...]],
    node_count: int,
) -> tuple[int, ...]:
    partition = list(range(node_count))
    next_cell = node_count
    for nodes in decoys.values():
        for node in nodes:
            partition[node] = next_cell
        next_cell += 1
    for node in path:
        partition[node] = next_cell
        next_cell += 1
    return tuple(partition)


def _augment_topology(
    adjacency: tuple[tuple[tuple[str, int], ...], ...],
    path: tuple[int, ...],
    decoys: dict[tuple[int, int], tuple[int, ...]],
    *,
    cause: Cause,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Add family-specific, label-invariant topology without changing the solution backbone."""
    if cause not in {"delete", "relate"}:
        return adjacency
    mutable = [list(edges) for edges in adjacency]
    link_label = _opaque_label(rng, prefix="a")
    max_position = len(path) - 1
    max_step = max(step for _, step in decoys)
    for position in range(max_position):
        terminal_nodes = decoys[(position, max_step)]
        for branch, node in enumerate(terminal_nodes):
            if cause == "delete":
                mutable[node].append((link_label, path[position]))
            elif position + 1 < max_position:
                next_nodes = decoys[(position + 1, 0)]
                mutable[node].append((link_label, next_nodes[branch % len(next_nodes)]))
    return tuple(tuple(edges) for edges in mutable)


def _dangerous_partition(
    source: tuple[int, ...],
    path: tuple[int, ...],
    decoys: dict[tuple[int, int], tuple[int, ...]],
    rng: np.random.Generator,
    *,
    cause: Cause,
) -> tuple[int, ...]:
    partition = list(source)
    if cause == "add":
        trap_position = int(rng.integers(1, max(2, len(path) // 3)))
    elif cause == "split":
        trap_position = int(rng.integers(max(2, len(path) // 3), max(3, 2 * len(path) // 3)))
    elif cause == "relate":
        trap_position = int(rng.integers(max(2, 2 * len(path) // 3), len(path)))
    else:
        trap_position = int(rng.integers(1, max(2, len(path) // 2)))
    decoy_nodes = decoys[(trap_position - 1, 0)]
    trap_decoy = decoy_nodes[0]
    partition[path[trap_position]] = partition[trap_decoy]
    return tuple(partition)


def _decoy_partition(
    base: tuple[int, ...],
    path: tuple[int, ...],
    decoys: dict[tuple[int, int], tuple[int, ...]],
    node_count: int,
    target_index: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if target_index == 0:
        return base
    partition = list(base)
    if target_index == 1:
        position = int(rng.integers(1, len(path) - 1))
        nodes = decoys[(position - 1, 0)]
        partition[path[position]] = partition[nodes[-1]]
    elif target_index == 2:
        modulus = max(3, int(np.sqrt(node_count)))
        partition = [value % modulus for value in partition]
    else:
        position = int(rng.integers(1, len(path) - 1))
        nodes = decoys[(position - 1, 0)]
        partition[path[position]] = partition[nodes[0]]
    return tuple(partition)


def _decoy_macro(action_labels: tuple[str, ...], target_index: int) -> tuple[str, ...]:
    if target_index == 0:
        return (action_labels[-1], action_labels[0], action_labels[0])
    if target_index == 1:
        return tuple([action_labels[0]] * 2 + [action_labels[-1]])
    if target_index == 2:
        return (action_labels[1], action_labels[1])
    return (action_labels[-1],) * 4


def _opaque_label(rng: np.random.Generator, *, prefix: str) -> str:
    return f"{prefix}_{int(rng.integers(0, 16**6)):06x}"


def _operator_name(edit_class: EditClass) -> str:
    return {
        "add": "append_unary_predicate",
        "merge": "quotient_state_symbols",
        "split": "refine_concept_cell",
        "relate": "append_binary_relation",
        "compose": "compose_action_sequence",
        "delete": "remove_state_feature",
        "relevel": "select_abstraction_level",
    }[edit_class]


def repair_signature(candidate: EditCandidate) -> tuple[str, object]:
    """Canonicalize a repair modulo arbitrary partition-cell identifiers."""
    if candidate.macro is not None:
        return ("macro", candidate.macro)
    assert candidate.partition is not None
    relabeling: dict[int, int] = {}
    canonical: list[int] = []
    for cell in candidate.partition:
        if cell not in relabeling:
            relabeling[cell] = len(relabeling)
        canonical.append(relabeling[cell])
    return ("partition", tuple(canonical))


def trace_features(trace: SearchResult, *, size_scale: int) -> np.ndarray:
    """Extract features from a trace and graph-size scalar; no task is accepted."""
    branching = np.asarray(trace.branching_counts or (0,), dtype=float)
    conflict_depths = np.asarray(trace.conflict_depths or (0,), dtype=float)
    encountered_unique = len(set(trace.encountered_states))
    features = [
        float(trace.solved),
        trace.generated / max(1, trace.expansions),
        trace.duplicates / max(1, trace.generated),
        trace.alias_conflicts / max(1, trace.duplicates),
        float(trace.max_depth),
        trace.frontier / max(1, trace.expansions),
        np.log1p(size_scale),
        float(branching.mean()),
        float(branching.std()),
        float(branching.max()),
        float(conflict_depths.min()),
        float(conflict_depths.mean()),
        float(conflict_depths.max()),
        encountered_unique / max(1, len(trace.encountered_states)),
        trace.expansions / max(1, size_scale),
    ]
    return np.asarray(features, dtype=float)


@dataclass(frozen=True)
class CentroidDiagnostic:
    mean: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, labels: list[Cause]) -> "CentroidDiagnostic":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (features - mean) / scale
        centroids = np.vstack(
            [normalized[np.asarray(labels) == cause].mean(axis=0) for cause in CAUSES]
        )
        return cls(mean=mean, scale=scale, centroids=centroids)

    def ranked_causes(self, features: np.ndarray) -> list[Cause]:
        normalized = (features - self.mean) / self.scale
        distances = np.square(self.centroids - normalized).mean(axis=1)
        return [CAUSES[index] for index in np.argsort(distances)]


@dataclass(frozen=True)
class LinearDiagnostic:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: list[Cause],
        *,
        iterations: int = 2_500,
        learning_rate: float = 0.2,
        regularization: float = 1e-3,
    ) -> "LinearDiagnostic":
        """Fit deterministic multinomial logistic regression with full-batch GD."""
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        normalized = (features - mean) / scale
        design = np.column_stack([normalized, np.ones(len(normalized))])
        targets = np.eye(len(CAUSES))[
            np.asarray([CAUSES.index(label) for label in labels])
        ]
        weights = np.zeros((design.shape[1], len(CAUSES)), dtype=float)
        for _ in range(iterations):
            logits = design @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = design.T @ (probabilities - targets) / len(design)
            gradient[:-1] += regularization * weights[:-1]
            weights -= learning_rate * gradient
        return cls(mean=mean, scale=scale, weights=weights)

    def ranked_causes(self, features: np.ndarray) -> list[Cause]:
        normalized = (features - self.mean) / self.scale
        design = np.append(normalized, 1.0)
        scores = design @ self.weights
        return [CAUSES[index] for index in np.argsort(-scores)]


def train_diagnostic(
    *,
    seeds: int = 80,
    probe_limit: int = 60,
    planner: Literal["bfs", "astar"] = "bfs",
) -> tuple[CentroidDiagnostic, dict[str, int]]:
    rows: list[np.ndarray] = []
    labels: list[Cause] = []
    expansions = 0
    for seed in range(seeds):
        for task in generate_sequence(seed, split="train"):
            trace = plan(task, limit=probe_limit, planner=planner)
            expansions += trace.expansions
            rows.append(trace_features(trace, size_scale=task.size_scale))
            labels.append(task.cause)
    model = CentroidDiagnostic.fit(np.vstack(rows), labels)
    return model, {
        "training_seeds": seeds,
        "training_tasks": len(rows),
        "training_probe_expansions": expansions,
    }


def train_linear_diagnostic(
    *,
    seeds: int = 80,
    probe_limit: int = 60,
    planner: Literal["bfs", "astar"] = "bfs",
) -> LinearDiagnostic:
    rows: list[np.ndarray] = []
    labels: list[Cause] = []
    for seed in range(seeds):
        for task in generate_sequence(seed, split="train"):
            trace = plan(task, limit=probe_limit, planner=planner)
            rows.append(trace_features(trace, size_scale=task.size_scale))
            labels.append(task.cause)
    return LinearDiagnostic.fit(np.vstack(rows), labels)


def run_task(
    *,
    task: SearchTask,
    agent: AgentSpec,
    model: CentroidDiagnostic | LinearDiagnostic,
    seed: int,
    budget: int,
    phase_index: int,
    probe_limit: int = 60,
    trial_limit: int | None = None,
    revision_cost_scale: float = 1.0,
    planner: Literal["bfs", "astar"] = "bfs",
) -> dict[str, object]:
    if agent.mode == "raw":
        result = plan(task, limit=budget, planner=planner)
        return _record(
            task=task,
            agent=agent,
            seed=seed,
            budget=budget,
            phase_index=phase_index,
            success=result.solved,
            total_cost=result.expansions,
            probe_cost=0,
            verification_cost=0,
            generation_cost=0,
            application_cost=0,
            attempts=0,
            diagnosis="none",
            first_target="none",
            final_target="none",
            planner=planner,
        )
    if agent.mode == "fixed":
        result = plan(task, limit=budget, fixed=True, planner=planner)
        return _record(
            task=task,
            agent=agent,
            seed=seed,
            budget=budget,
            phase_index=phase_index,
            success=result.solved,
            total_cost=result.expansions,
            probe_cost=0,
            verification_cost=0,
            generation_cost=0,
            application_cost=0,
            attempts=0,
            diagnosis="fixed",
            first_target="none",
            final_target="none",
            planner=planner,
        )

    probe_cap = min(probe_limit, budget)
    trace = plan(task, limit=probe_cap, planner=planner)
    spent = trace.expansions
    if trace.solved:
        return _record(
            task=task,
            agent=agent,
            seed=seed,
            budget=budget,
            phase_index=phase_index,
            success=True,
            total_cost=spent,
            probe_cost=spent,
            verification_cost=0,
            generation_cost=0,
            application_cost=0,
            attempts=0,
            diagnosis="not_needed",
            first_target="none",
            final_target="none",
            planner=planner,
        )

    feature_vector = trace_features(trace, size_scale=task.size_scale)
    ranked = model.ranked_causes(feature_vector)
    diagnosis: Cause
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, budget, phase_index, AGENTS.index(agent), 419])
    )
    if agent.mode == "oracle_edit":
        diagnosis = task.cause
        ordered_actions: list[Cause | EditCandidate] = []
        if task.cause == "search":
            ordered_actions.append("search")
        else:
            ordered_actions.append(
                next(c for c in task.candidates if c.target == task.correct_target)
            )
    elif agent.mode == "oracle_class":
        diagnosis = task.cause
        ordered_actions = _candidate_actions(
            task, [task.cause], agent, rng, trace=trace, rank_targets=True
        )
    elif agent.mode == "random":
        diagnosis = "not_applicable"  # type: ignore[assignment]
        ordered_actions = _candidate_actions(
            task, list(CAUSES), agent, rng, trace=trace, rank_targets=False
        )
        rng.shuffle(ordered_actions)
    elif agent.mode == "score_only":
        diagnosis = "not_applicable"  # type: ignore[assignment]
        scored = sorted(
            task.candidates,
            key=lambda candidate: _candidate_score(task, candidate, trace),
            reverse=True,
        )
        ordered_actions = list(scored)
        # Search has no candidate score on the common scale. Put it after every
        # scored edit instead of using the former alias-conflict shortcut, which
        # could consume the whole remaining budget before any candidate ran.
        ordered_actions.append("search")
    elif agent.mode == "score_round_robin":
        diagnosis = "not_applicable"  # type: ignore[assignment]
        by_class: dict[EditClass, list[EditCandidate]] = {}
        for edit_class in EDIT_CLASSES:
            by_class[edit_class] = sorted(
                (
                    candidate
                    for candidate in task.candidates
                    if candidate.edit_class == edit_class
                ),
                key=lambda candidate: _candidate_score(task, candidate, trace),
                reverse=True,
            )
        class_order = list(EDIT_CLASSES)
        rng.shuffle(class_order)
        ordered_actions = []
        for rank in range(max(len(items) for items in by_class.values())):
            for edit_class in class_order:
                if rank < len(by_class[edit_class]):
                    ordered_actions.append(by_class[edit_class][rank])
        ordered_actions.append("search")
    else:
        diagnosis = ranked[0]
        ordered_actions = _candidate_actions(
            task,
            ranked,
            agent,
            rng,
            trace=trace,
            rank_targets=agent.mode == "trace",
        )

    attempts = 0
    generation_cost = 0
    application_cost = 0
    verification_cost = 0
    first_target = "none"
    final_target = "none"
    success = False
    for action in ordered_actions:
        if spent >= budget:
            break
        if action == "search":
            candidate = None
            target = "continue_search"
            marginal = 0
        else:
            candidate = action
            target = candidate.target
            generated_now = int(round(revision_cost_scale * CANDIDATE_GENERATION_COST))
            applied_now = int(
                round(revision_cost_scale * APPLICATION_COST[candidate.edit_class])
            )
            marginal = generated_now + applied_now
        if spent + marginal > budget:
            continue
        attempts += 1
        if attempts == 1:
            first_target = target
        if candidate is not None:
            generation_cost += generated_now
            application_cost += applied_now
        spent += marginal
        remaining = budget - spent
        cap = remaining if trial_limit is None else min(trial_limit, remaining)
        result = plan(task, limit=cap, candidate=candidate, planner=planner)
        verification_cost += result.expansions
        spent += result.expansions
        if result.solved:
            success = True
            final_target = target
            break

    return _record(
        task=task,
        agent=agent,
        seed=seed,
        budget=budget,
        phase_index=phase_index,
        success=success,
        total_cost=spent,
        probe_cost=trace.expansions,
        verification_cost=verification_cost,
        generation_cost=generation_cost,
        application_cost=application_cost,
        attempts=attempts,
        diagnosis=diagnosis,
        first_target=first_target,
        final_target=final_target,
        planner=planner,
    )


def _candidate_actions(
    task: SearchTask,
    ranked_causes: list[Cause],
    agent: AgentSpec,
    rng: np.random.Generator,
    *,
    trace: SearchResult,
    rank_targets: bool,
) -> list[Cause | EditCandidate]:
    actions: list[Cause | EditCandidate] = []
    for cause in ranked_causes:
        if cause == "search":
            actions.append("search")
            continue
        if cause not in agent.capabilities:
            continue
        candidates = [c for c in task.candidates if c.edit_class == cause]
        if rank_targets:
            candidates.sort(
                key=lambda candidate: _candidate_score(task, candidate, trace),
                reverse=True,
            )
        else:
            rng.shuffle(candidates)
        actions.extend(candidates)
    return actions


def _candidate_score(
    task: SearchTask, candidate: EditCandidate, trace: SearchResult
) -> float:
    """Rank targets using only states and transitions observed in the failed probe."""
    observed = tuple(dict.fromkeys(trace.encountered_states))
    if candidate.macro is not None:
        applicable = sum(
            _apply_macro(task, state, candidate.macro) is not None
            for state in trace.expanded_states
        )
        return len(candidate.macro) * applicable / max(1, len(trace.expanded_states))
    if candidate.partition is None:
        return 0.0

    base_unique = max(1, len({task.base_partition[state] for state in observed}))
    candidate_keys = [candidate.partition[state] for state in observed]
    unique = max(1, len(set(candidate_keys)))
    collapse = max(0.0, (base_unique - unique) / base_unique)
    representatives: dict[int, int] = {}
    conflicts = 0
    duplicates = 0
    for state, key in zip(observed, candidate_keys, strict=True):
        if key in representatives:
            duplicates += 1
            conflicts += int(
                task.signatures[representatives[key]] != task.signatures[state]
            )
        else:
            representatives[key] = state
    conflict_rate = conflicts / max(1, duplicates)
    base_conflict = trace.alias_conflicts / max(1, trace.duplicates)
    conflict_reduction = max(0.0, base_conflict - conflict_rate)
    return collapse * (1.0 - conflict_rate) + conflict_reduction


def _record(
    *,
    task: SearchTask,
    agent: AgentSpec,
    seed: int,
    budget: int,
    phase_index: int,
    success: bool,
    total_cost: int,
    probe_cost: int,
    verification_cost: int,
    generation_cost: int,
    application_cost: int,
    attempts: int,
    diagnosis: str,
    first_target: str,
    final_target: str,
    planner: str,
) -> dict[str, object]:
    edit_needed = task.cause != "search"
    target_correct = (
        final_target in task.accepted_targets
        if edit_needed
        else final_target == "continue_search" or diagnosis == "not_needed"
    )
    return {
        "seed": seed,
        "split": "test",
        "planner": planner,
        "budget": budget,
        "agent": agent.name,
        "phase": task.cause,
        "phase_index": phase_index,
        "success": bool(success),
        "total_cost": int(total_cost),
        "probe_cost": int(probe_cost),
        "verification_cost": int(verification_cost),
        "candidate_generation_cost": int(generation_cost),
        "edit_application_cost": int(application_cost),
        "attempts": int(attempts),
        "diagnosis": diagnosis,
        "diagnosis_correct": bool(diagnosis == task.cause)
        if diagnosis not in {"none", "fixed", "not_needed", "not_applicable"}
        else np.nan,
        "first_target": first_target,
        "first_target_correct": bool(first_target in task.accepted_targets)
        if edit_needed and first_target not in {"none", "continue_search"}
        else np.nan,
        "first_candidate_success": bool(
            success
            and attempts == 1
            and edit_needed
            and first_target not in {"none", "continue_search"}
        )
        if edit_needed and first_target not in {"none", "continue_search"}
        else np.nan,
        "final_target": final_target,
        "target_correct": bool(target_correct),
        "generator_target_match": bool(final_target == task.correct_target)
        if edit_needed and final_target not in {"none", "continue_search"}
        else np.nan,
        "final_candidate_accepted": bool(final_target in task.accepted_targets)
        if edit_needed and final_target not in {"none", "continue_search"}
        else np.nan,
        "accepted_candidate_count": len(task.accepted_targets),
        "accepted_class_count": len(
            {
                candidate.edit_class
                for candidate in task.candidates
                if candidate.target in task.accepted_targets
            }
        ),
        "accepted_unique_repair_count": len(
            {
                repair_signature(candidate)
                for candidate in task.candidates
                if candidate.target in task.accepted_targets
            }
        ),
        "unnecessary_edit": bool(task.cause == "search" and application_cost > 0),
        "task_size": task.size_scale,
    }


def diagnostic_accuracy(
    model: CentroidDiagnostic | LinearDiagnostic,
    *,
    seeds: int = 200,
    probe_limit: int = 60,
    planner: Literal["bfs", "astar"] = "bfs",
) -> tuple[float, np.ndarray]:
    matrix = np.zeros((len(CAUSES), len(CAUSES)), dtype=int)
    for seed in range(1_000, 1_000 + seeds):
        for task in generate_sequence(seed, split="test"):
            result = plan(task, limit=probe_limit, planner=planner)
            predicted = model.ranked_causes(
                trace_features(result, size_scale=task.size_scale)
            )[0]
            matrix[CAUSES.index(task.cause), CAUSES.index(predicted)] += 1
    return float(np.trace(matrix) / matrix.sum()), matrix


def action_histogram(task: SearchTask) -> Counter[str]:
    return Counter(action for edges in task.adjacency for action, _ in edges)
