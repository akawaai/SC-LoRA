from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _to_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _sanitize_array(values: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _save_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, path: str) -> None:
    matrix = _sanitize_array(matrix)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_scatter(x: np.ndarray, y: np.ndarray, labels: list[str], title: str, xlab: str, ylab: str, path: str) -> None:
    x = _sanitize_array(x).reshape(-1)
    y = _sanitize_array(y).reshape(-1)
    n = min(x.shape[0], y.shape[0], len(labels))

    fig, ax = plt.subplots(figsize=(6, 5))
    if n > 0:
        ax.scatter(x[:n], y[:n])
    for i, label in enumerate(labels[:n]):
        ax.annotate(label, (x[i], y[i]))
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_boxplot(values_by_group: dict[str, list[float]], title: str, ylab: str, path: str) -> None:
    labels = list(values_by_group.keys())
    values: list[list[float]] = []
    for key in labels:
        finite_values = [_to_finite_float(v, default=float("nan")) for v in values_by_group.get(key, [])]
        finite_values = [v for v in finite_values if np.isfinite(v)]
        values.append(finite_values if finite_values else [0.0])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(values, labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_poster_figures(
    logs_training: dict[str, Any],
    logs_eval: dict[str, Any],
    logs_mechanistic: dict[str, Any],
    save_path: str,
) -> str:
    ensure_dir(save_path)

    contextual = logs_eval.get("contextual", {})
    forgetting = logs_eval.get("forgetting", {})

    domains = list(contextual.keys())
    if domains:
        gate_matrix = np.array([[_to_finite_float(contextual[d].get("gate_usage", 0.0))] for d in domains], dtype=float)
        _save_heatmap(
            gate_matrix,
            row_labels=domains,
            col_labels=["avg_gate_usage"],
            title="Adapter Activation Heatmap",
            path=f"{save_path}/gate_heatmap.png",
        )

        rank_matrix = np.array([[_to_finite_float(contextual[d].get("rank_usage", 0.0)) for d in domains]], dtype=float)
        _save_heatmap(
            rank_matrix,
            row_labels=["rank"],
            col_labels=domains,
            title="Rank-by-Domain Heatmap",
            path=f"{save_path}/rank_heatmap.png",
        )

    score_dict = forgetting.get("score", {})
    forget_dict = forgetting.get("forgetting", {})
    if score_dict and forget_dict:
        pareto_labels = list(score_dict.keys())
        y = np.array([_to_finite_float(np.mean(score_dict[d])) for d in pareto_labels], dtype=float)
        x = np.array([_to_finite_float(np.mean(forget_dict[d])) for d in pareto_labels], dtype=float)
        _save_scatter(
            x=x,
            y=y,
            labels=pareto_labels,
            title="Target vs Forgetting Pareto",
            xlab="Forgetting",
            ylab="Target Score",
            path=f"{save_path}/pareto_target_forgetting.png",
        )

        phases = max((len(v) for v in forget_dict.values()), default=0)
        if phases > 0:
            domains_f = list(forget_dict.keys())
            mat = np.zeros((phases, len(domains_f)), dtype=float)
            for j, dom in enumerate(domains_f):
                series = forget_dict[dom]
                for i in range(min(phases, len(series))):
                    mat[i, j] = _to_finite_float(series[i])

            _save_heatmap(
                mat,
                row_labels=[f"phase_{i+1}" for i in range(phases)],
                col_labels=domains_f,
                title="Forgetting Matrix",
                path=f"{save_path}/forgetting_matrix.png",
            )

    drift_general: dict[str, list[float]] = {}
    drift_domain: dict[str, list[float]] = {}
    for domain_id, layer_stats in logs_mechanistic.items():
        drift_general[domain_id] = []
        drift_domain[domain_id] = []
        for values in layer_stats.values():
            drift_general[domain_id].append(_to_finite_float(values.get("drift_general", 0.0)))
            drift_domain[domain_id].append(_to_finite_float(values.get("drift_domain", 0.0)))

    if drift_general:
        _save_boxplot(
            values_by_group=drift_general,
            title="Mechanistic Drift (General Features)",
            ylab="Drift",
            path=f"{save_path}/feature_drift_general.png",
        )

    if drift_domain:
        _save_boxplot(
            values_by_group=drift_domain,
            title="Mechanistic Drift (Domain Features)",
            ylab="Drift",
            path=f"{save_path}/feature_drift_domain.png",
        )

    return save_path
