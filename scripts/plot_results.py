#!/usr/bin/env python3
"""Export figures from a completed CL4KT five-fold run; never invent missing data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score  # noqa: E402


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

# CL4KT, Table 2 in the supplied WWW'22 paper. These are reported reference
# means, never inferred from reproduced results or substituted for missing folds.
PAPER_TARGETS = {
    "algebra05": {"auc": 0.7891, "rmse": 0.3815},
    "algebra06": {"auc": 0.7733, "rmse": 0.3791},
    "assistments09": {"auc": 0.7624, "rmse": 0.4333},
    "slepemapy": {"auc": 0.7218, "rmse": 0.3926},
    "spanish": {"auc": 0.8289, "rmse": 0.4049},
    "statics": {"auc": 0.7943, "rmse": 0.3945},
}
DATASET_ALIASES = {
    "assist09": "assistments09",
    "bridge_algebra06": "algebra06",
    "sampled_slepemapy": "slepemapy",
}


def canonical_dataset_name(data_name: str) -> str:
    name = data_name.lower()
    return DATASET_ALIASES.get(name, name)


def paper_targets(data_name: str, recorded: dict | None = None) -> dict:
    """Resolve the paper reference even when generic training records paper=None."""
    canonical = canonical_dataset_name(data_name)
    if canonical not in PAPER_TARGETS:
        raise ValueError(f"No paper Table 2 target is defined for dataset {data_name!r}.")
    expected = PAPER_TARGETS[canonical]
    if recorded is not None:
        for name, target in expected.items():
            if name not in recorded or not np.isclose(recorded[name], target, rtol=0, atol=1e-12):
                raise ValueError(f"Recorded paper {name} for {data_name} disagrees with Table 2.")
    return dict(expected)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def export(figure, directory: Path, stem: str) -> list[str]:
    outputs = []
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def load_run(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    if config.get("num_folds") != 5:
        raise ValueError("The requested figures require a configured five-fold run.")
    if summary.get("complete") is not True or summary.get("completed_folds") != 5:
        raise ValueError("The run is incomplete. Wait for all five folds before exporting comparison figures.")
    # Resolve into an in-memory copy. Historical experiment JSON stays intact.
    summary = dict(summary)
    summary["paper"] = paper_targets(config["data_name"], summary.get("paper"))
    folds = sorted(summary["folds"], key=lambda result: result["fold"])
    if [fold["fold"] for fold in folds] != list(range(5)):
        raise ValueError("Summary must contain exactly folds 0 through 4.")
    loaded = []
    for fold in folds:
        directory = run_dir / f"fold_{fold['fold']}"
        metrics = read_json(directory / "metrics.json")
        if metrics != fold:
            raise ValueError(f"Summary differs from fold {fold['fold']} metrics.")
        with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        epochs = np.array([int(row["epoch"]) for row in history])
        valid_auc = np.array([float(row["valid_auc"]) for row in history])
        if not history or not np.isfinite(valid_auc).all():
            raise ValueError(f"Fold {fold['fold']} has empty or non-finite validation history.")
        best_index = int(np.argmax(valid_auc))
        if epochs[best_index] != metrics["best_epoch"]:
            raise ValueError(f"Fold {fold['fold']} best epoch differs from its validation history.")
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            prediction = np.asarray(archive["prediction"], dtype=float)
            target = np.asarray(archive["target"], dtype=float)
        if prediction.ndim != 1 or prediction.shape != target.shape or prediction.size == 0:
            raise ValueError(f"Fold {fold['fold']} prediction and target arrays must be nonempty matching vectors.")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f"Fold {fold['fold']} predictions contain non-finite values.")
        if not ((prediction >= 0) & (prediction <= 1)).all() or not np.isin(target, [0, 1]).all():
            raise ValueError(f"Fold {fold['fold']} has invalid probabilities or targets.")
        if len(prediction) != metrics["count"]:
            raise ValueError(f"Fold {fold['fold']} prediction count differs from its metrics.")
        recomputed = dict(auc=float(roc_auc_score(target, prediction)),
                          rmse=float(np.sqrt(mean_squared_error(target, prediction))),
                          acc=float(accuracy_score(target, prediction >= 0.5)))
        for name, value in recomputed.items():
            if not np.isfinite(value) or not np.isclose(value, metrics[name], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Fold {fold['fold']} recorded {name} differs from its saved predictions.")
        loaded.append(dict(metrics=metrics, epochs=epochs, valid_auc=valid_auc,
                           prediction=prediction, target=target))
    for name in ("auc", "rmse"):
        values = np.array([fold["metrics"][name] for fold in loaded])
        if not np.isfinite(values).all() or not np.isclose(values.mean(), summary["mean"][name]):
            raise ValueError(f"Summary mean {name} differs from per-fold results.")
        if not np.isclose(values.std(ddof=0), summary["std_population"][name]):
            raise ValueError(f"Summary standard deviation {name} differs from per-fold results.")
    return config, summary, loaded


def validation_figure(folds: list[dict], dataset: str):
    figure, ax = plt.subplots(figsize=(8.6, 4.8))
    for color, fold in zip(COLORS, folds):
        metrics = fold["metrics"]
        best_index = int(np.argmax(fold["valid_auc"]))
        epoch = int(fold["epochs"][best_index])
        auc = float(fold["valid_auc"][best_index])
        ax.plot(fold["epochs"], fold["valid_auc"], color=color, linewidth=1.6,
                label=f"Fold {metrics['fold'] + 1}; best epoch {epoch}")
        ax.scatter([epoch], [auc], color=color, s=45, marker="o", edgecolors="white", zorder=4)
        ax.annotate(str(epoch), (epoch, auc), xytext=(4, 7), textcoords="offset points",
                    color=color, fontsize=9)
    ax.set(title=f"{dataset}: validation performance", xlabel="Training epoch",
           ylabel="Validation ROC AUC (unitless)")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", frameon=False, fontsize=9)
    figure.text(0.5, 0.01, "Markers identify the checkpoint selected by maximum validation AUC.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def comparison_figure(summary: dict, folds: list[dict], dataset: str):
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    for ax, name, label, direction in zip(
        axes, ("auc", "rmse"),
        ("ROC AUC (unitless)", "RMSE (probability scale)"),
        ("Higher is better", "Lower is better"),
    ):
        paper = float(summary["paper"][name])
        mean = float(summary["mean"][name])
        std = float(summary["std_population"][name])
        ax.scatter([0], [paper], marker="D", color="#595959", s=65, zorder=4)
        ax.errorbar([1], [mean], yerr=[std], fmt="o", markersize=7, color=COLORS[0],
                    capsize=6, linewidth=1.5, zorder=4)
        # Show the actual folds as well as their unweighted mean and population SD.
        ax.scatter(1 + np.linspace(-0.09, 0.09, len(folds)),
                   [fold["metrics"][name] for fold in folds], color=COLORS[0],
                   alpha=0.35, s=25, zorder=3)
        ax.annotate(f"{paper:.4f}", (0, paper), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10)
        ax.annotate(f"{mean:.4f} ± {std:.4f}", (1, mean + std), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=10)
        ax.set_xticks([0, 1], ["Paper Table 2", "Reproduced"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(label)
        ax.set_title(direction, fontsize=11)
        ax.grid(axis="y", alpha=0.22)
        ax.margins(y=0.35)
        # AUC and RMSE are measured on [0, 1]; the point plot may use a narrow
        # axis to show the observed values, but never extends beyond this range.
        lower, upper = ax.get_ylim()
        ax.set_ylim(max(0, lower), min(1, upper))
    figure.suptitle(f"{dataset}: paper and reproduced five-fold test results", fontsize=13)
    figure.text(0.5, 0.01,
                "Reproduced error bars: population SD across folds; faint points: individual folds.\n"
                "The paper reports a mean only; its uncertainty is unavailable here.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    return figure


def calibration_bins(prediction: np.ndarray, target: np.ndarray, n_bins: int) -> list[dict]:
    # Uniform probability intervals, with p=1 included in the last interval.
    indices = np.minimum((prediction * n_bins).astype(int), n_bins - 1)
    result = []
    for index in range(n_bins):
        mask = indices == index
        if mask.any():
            result.append(dict(bin=index, lower=index / n_bins, upper=(index + 1) / n_bins,
                               count=int(mask.sum()), mean_prediction=float(prediction[mask].mean()),
                               observed_fraction_correct=float(target[mask].mean())))
    return result


def calibration_figure(folds: list[dict], dataset: str, n_bins: int):
    figure, ax = plt.subplots(figsize=(6.7, 6.0))
    rows = []
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1.2, label="Perfect calibration")
    for color, fold in zip(COLORS, folds):
        bins = calibration_bins(fold["prediction"], fold["target"], n_bins)
        ax.plot([item["mean_prediction"] for item in bins],
                [item["observed_fraction_correct"] for item in bins], color=color,
                marker="o", markersize=3.5, alpha=0.7, linewidth=1.1,
                label=f"Fold {fold['metrics']['fold'] + 1}")
        rows.extend(dict(series=f"fold_{fold['metrics']['fold']}", **item) for item in bins)
    prediction = np.concatenate([fold["prediction"] for fold in folds])
    target = np.concatenate([fold["target"] for fold in folds])
    pooled = calibration_bins(prediction, target, n_bins)
    ax.plot([item["mean_prediction"] for item in pooled],
            [item["observed_fraction_correct"] for item in pooled], color="#111111",
            marker="s", markersize=4.5, linewidth=2.4, label="Pooled test predictions", zorder=5)
    rows.extend(dict(series="pooled", **item) for item in pooled)
    ax.set(title=f"{dataset}: test prediction calibration", xlim=(0, 1), ylim=(0, 1),
           xlabel="Mean predicted probability", ylabel="Observed fraction correct")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    figure.text(0.5, 0.015,
                f"{n_bins} equal-width probability bins; empty bins omitted.\n"
                "Supplementary calibration analysis, not an exact reproduction of paper Figure 7.\n"
                "CL4KT only; no AKT baseline was trained for this plot.",
                ha="center", fontsize=8.5)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    return figure, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run directory containing summary.json")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of equal-width probability bins (default: 10)")
    args = parser.parse_args()
    if args.calibration_bins < 2:
        parser.error("--calibration-bins must be at least 2")
    run_dir = args.run_dir.expanduser().resolve()
    try:
        config, summary, folds = load_run(run_dir)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot plot run: {error}\n")
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    canonical = canonical_dataset_name(config["data_name"])
    dataset = {"assistments09": "ASSISTments09"}.get(canonical, canonical)
    outputs = export(validation_figure(folds, dataset), output_dir, "validation_auc")
    outputs.extend(export(comparison_figure(summary, folds, dataset), output_dir, "paper_comparison"))
    calibration, rows = calibration_figure(folds, dataset, args.calibration_bins)
    outputs.extend(export(calibration, output_dir, "calibration"))
    bin_path = output_dir / "calibration_bins.csv"
    with bin_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(str(bin_path))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
