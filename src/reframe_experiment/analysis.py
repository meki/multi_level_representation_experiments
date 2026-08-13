from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .interactive import AGENTS, CAUSES


AGENT_ORDER = [agent.name for agent in AGENTS]
PHASE_ORDER = list(CAUSES)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    key: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([20260812, key]))
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    raw: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    *,
    main_budget: int,
    bootstrap_samples: int,
    confusion: np.ndarray,
) -> dict[str, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    main = raw.loc[raw["budget"] == main_budget].copy()

    phase_summary = (
        main.groupby(["agent", "phase"], sort=False)
        .agg(
            success_rate=("success", "mean"),
            mean_total_cost=("total_cost", "mean"),
            diagnosis_accuracy=("diagnosis_correct", "mean"),
            first_target_accuracy=("first_target_correct", "mean"),
            first_candidate_success=("first_candidate_success", "mean"),
            unnecessary_edit_rate=("unnecessary_edit", "mean"),
            mean_attempts=("attempts", "mean"),
            mean_accepted_candidates=("accepted_candidate_count", "mean"),
            mean_accepted_classes=("accepted_class_count", "mean"),
            mean_accepted_unique_repairs=("accepted_unique_repair_count", "mean"),
            n=("success", "size"),
        )
        .reset_index()
    )
    phase_summary["ci_low"] = np.nan
    phase_summary["ci_high"] = np.nan
    for row_index, row in phase_summary.iterrows():
        values = main.loc[
            (main["agent"] == row["agent"]) & (main["phase"] == row["phase"]),
            "success",
        ].to_numpy()
        low, high = bootstrap_mean_ci(
            values,
            key=10_000 + PHASE_ORDER.index(row["phase"]) * 100 + AGENT_ORDER.index(row["agent"]),
            samples=bootstrap_samples,
        )
        phase_summary.loc[row_index, ["ci_low", "ci_high"]] = [low, high]
    seed_agent = (
        main.groupby(["seed", "agent"], sort=False)
        .agg(success_rate=("success", "mean"), mean_total_cost=("total_cost", "mean"))
        .reset_index()
    )

    overall_rows: list[dict[str, object]] = []
    for agent_index, agent in enumerate(AGENT_ORDER):
        values = seed_agent.loc[seed_agent["agent"] == agent, "success_rate"].to_numpy()
        low, high = bootstrap_mean_ci(
            values,
            key=30_000 + main_budget * 100 + agent_index,
            samples=bootstrap_samples,
        )
        subset = main.loc[main["agent"] == agent]
        overall_rows.append(
            {
                "agent": agent,
                "success_rate": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "mean_total_cost": float(subset["total_cost"].mean()),
                "mean_success_cost": float(
                    subset.loc[subset["success"], "total_cost"].mean()
                ),
                "mean_failure_cost": float(
                    subset.loc[~subset["success"], "total_cost"].mean()
                ),
                "mean_probe_cost": float(subset["probe_cost"].mean()),
                "mean_verification_cost": float(subset["verification_cost"].mean()),
                "mean_candidate_generation_cost": float(
                    subset["candidate_generation_cost"].mean()
                ),
                "mean_edit_application_cost": float(
                    subset["edit_application_cost"].mean()
                ),
                "diagnosis_accuracy": float(subset["diagnosis_correct"].mean()),
                "first_target_accuracy": float(subset["first_target_correct"].mean()),
                "first_candidate_success": float(
                    subset["first_candidate_success"].mean()
                ),
                "unnecessary_edit_rate": float(subset["unnecessary_edit"].mean()),
                "mean_attempts": float(subset["attempts"].mean()),
                "mean_accepted_candidates": float(
                    subset["accepted_candidate_count"].mean()
                ),
                "mean_accepted_classes": float(subset["accepted_class_count"].mean()),
                "mean_accepted_unique_repairs": float(
                    subset["accepted_unique_repair_count"].mean()
                ),
            }
        )
    overall = pd.DataFrame(overall_rows)

    pivot = seed_agent.pivot(index="seed", columns="agent", values="success_rate")
    comparisons: list[dict[str, object]] = []
    full = pivot["Trace-guided"].to_numpy()
    for baseline_index, baseline in enumerate(AGENT_ORDER):
        if baseline == "Trace-guided":
            continue
        differences = full - pivot[baseline].to_numpy()
        low, high = bootstrap_mean_ci(
            differences,
            key=2000 + baseline_index,
            samples=bootstrap_samples,
        )
        comparisons.append(
            {
                "comparison": f"Trace-guided - {baseline}",
                "mean_paired_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "n_pairs": len(differences),
            }
        )
    comparisons_df = pd.DataFrame(comparisons)

    curve_seed = (
        raw.groupby(["budget", "seed", "agent"], sort=False)["success"]
        .mean()
        .reset_index(name="success_rate")
    )
    curve_rows: list[dict[str, object]] = []
    for (budget, agent), group in (
        curve_seed.groupby(["budget", "agent"], sort=False)
    ):
        values = group["success_rate"].to_numpy()
        low, high = bootstrap_mean_ci(
            values,
            key=30_000 + int(budget) * 100 + AGENT_ORDER.index(agent),
            samples=bootstrap_samples,
        )
        curve_rows.append(
            {
                "budget": int(budget),
                "agent": agent,
                "success_rate": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    curve = pd.DataFrame(curve_rows)
    confusion_df = pd.DataFrame(confusion, index=CAUSES, columns=CAUSES)

    phase_summary.to_csv(results_dir / "summary_main_budget.csv", index=False, lineterminator="\n")
    overall.to_csv(results_dir / "overall_main_budget.csv", index=False, lineterminator="\n")
    comparisons_df.to_csv(results_dir / "paired_comparisons.csv", index=False, lineterminator="\n")
    curve.to_csv(results_dir / "budget_curve.csv", index=False, lineterminator="\n")
    confusion_df.to_csv(results_dir / "diagnostic_confusion.csv", lineterminator="\n")
    _write_latex_table(overall, results_dir / "table_main.tex", main_budget, int(main["seed"].nunique()))
    _plot_phase_success(phase_summary, figures_dir)
    _plot_budget_curve(curve, figures_dir)
    _plot_cost_breakdown(overall, figures_dir)
    _plot_confusion(confusion, figures_dir)
    return {
        "phase": phase_summary,
        "overall": overall,
        "comparisons": comparisons_df,
        "curve": curve,
        "confusion": confusion_df,
    }


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_")


def _format_optional(value: float) -> str:
    return "--" if np.isnan(value) else f"{value:.3f}"


def _format_cost(value: float) -> str:
    return "--" if np.isnan(value) else f"{value:.0f}"


def _write_latex_table(overall: pd.DataFrame, path: Path, budget: int, seed_count: int) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Held-out results at budget {budget:,}. Intervals are sequence-level percentile bootstrap 95\% intervals over {seed_count} paired test seeds. Diagnosis is measured when issued; first-candidate success is conditional on proposing an edit candidate.}}",
        r"\label{tab:main-results}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Agent & Success (95\% CI) & Cost S/F & Diagnosis & First cand. \\",
        r"\midrule",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"{_latex_escape(str(row['agent']))} & "
            f"{row['success_rate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] & "
            f"{row['mean_success_cost']:.0f}/{_format_cost(row['mean_failure_cost'])} & "
            f"{_format_optional(row['diagnosis_accuracy'])} & "
            f"{_format_optional(row['first_candidate_success'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _plot_phase_success(summary: pd.DataFrame, figures_dir: Path) -> None:
    selected = [
        "Raw BFS",
        "Random retry",
        "Candidate score-only",
        "Score round-robin",
        "Trace-guided",
        "Generator class",
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    x = np.arange(len(PHASE_ORDER))
    width = 0.135
    for index, agent in enumerate(selected):
        data = summary.loc[summary["agent"] == agent].set_index("phase").reindex(PHASE_ORDER)
        position = x + (index - 2.5) * width
        ax.bar(position, data["success_rate"], width, label=agent)
        ax.errorbar(
            position,
            data["success_rate"],
            yerr=np.vstack(
                [data["success_rate"] - data["ci_low"], data["ci_high"] - data["success_rate"]]
            ),
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=1.5,
        )
    ax.set_xticks(x, [phase.capitalize() for phase in PHASE_ORDER], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Held-out task family")
    ax.legend(
        ncol=3,
        fontsize=7.5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_success.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "phase_success.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_budget_curve(curve: pd.DataFrame, figures_dir: Path) -> None:
    selected = [
        "Raw BFS",
        "Random retry",
        "Candidate score-only",
        "Score round-robin",
        "Trace + random target",
        "Trace-guided",
        "Generator class",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for agent in selected:
        data = curve.loc[curve["agent"] == agent].sort_values("budget")
        ax.plot(data["budget"], data["success_rate"], marker="o", label=agent)
        ax.fill_between(data["budget"], data["ci_low"], data["ci_high"], alpha=0.08)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Total budget (search-equivalent units, log scale)")
    ax.set_ylabel("Mean success rate")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "budget_curve.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "budget_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_breakdown(overall: pd.DataFrame, figures_dir: Path) -> None:
    selected = [
        "Random retry",
        "Candidate score-only",
        "Score round-robin",
        "Trace + random target",
        "Trace-guided",
        "Generator class",
    ]
    data = overall.set_index("agent").loc[selected]
    components = [
        ("mean_probe_cost", "Initial probe"),
        ("mean_verification_cost", "Candidate verification"),
        ("mean_candidate_generation_cost", "Candidate generation"),
        ("mean_edit_application_cost", "Edit application"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bottom = np.zeros(len(data))
    for column, label in components:
        values = data[column].to_numpy()
        ax.bar(selected, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Mean charged cost")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "cost_breakdown.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "cost_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(confusion: np.ndarray, figures_dir: Path) -> None:
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(CAUSES)), CAUSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CAUSES)), CAUSES)
    ax.set_xlabel("Predicted diagnosis")
    ax.set_ylabel("Generator diagnosis")
    for row in range(len(CAUSES)):
        for column in range(len(CAUSES)):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=6.5,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figures_dir / "diagnostic_confusion.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "diagnostic_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
