#!/usr/bin/env python3
"""Summarize six completed CL4KT five-fold experiments using recorded artifacts.

No output is written until all six experiments are complete and their metrics,
dataset hashes, frozen source hashes and data manifests are consistent.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER_REFERENCE = {
    "title": "Contrastive Learning for Knowledge Tracing",
    "authors": "Wonsung Lee, Jaeyoon Chun, Youngmin Lee, Kyoungsoo Park, Sungrae Park",
    "venue": "The ACM Web Conference 2022",
    "doi": "10.1145/3485447.3512105",
    "url": "https://doi.org/10.1145/3485447.3512105",
    "local_notes": "docs/PAPER.md",
    "dataset_statistics_table": 1,
    "model_metrics_table": 2,
    "reported_uncertainty": "Table 2 provides fold means without numerical standard deviations.",
}
TARGETS = {
    "algebra05": dict(label="Algebra05", paper_name="algebra05", auc=0.7891, rmse=0.3815,
                      students=571, questions=173113, skills=112, interactions=607014),
    "bridge_algebra06": dict(label="Bridge Algebra06", paper_name="algebra06", auc=0.7733, rmse=0.3791,
                             students=1138, questions=129263, skills=493, interactions=1817450),
    "assistments09": dict(label="ASSIST09", paper_name="assist09", auc=0.7624, rmse=0.4333,
                         students=3695, questions=17728, skills=112, interactions=282071),
    "sampled_slepemapy": dict(label="Slepemapy*", paper_name="slepemapy", auc=0.7218, rmse=0.3926,
                             students=5000, questions=2723, skills=1391, interactions=625523),
    "spanish": dict(label="Spanish", paper_name="spanish", auc=0.8289, rmse=0.4049,
                    students=182, questions=409, skills=221, interactions=578726),
    "statics": dict(label="Statics†", paper_name="statics", auc=0.7943, rmse=0.3945,
                    students=333, questions=None, skills=1223, interactions=189297),
}


class ArtifactError(Exception):
    """A recorded experiment is incomplete or inconsistent."""


def require(condition, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, label: str) -> None:
    require(np.isfinite(actual) and np.isfinite(expected)
            and np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
            f"{label}: recorded {actual!r}, recomputed {expected!r}")


def completion_status(name: str, run_dir: Path) -> str | None:
    count = sum((run_dir / f"fold_{fold}" / "metrics.json").is_file() for fold in range(5))
    path = run_dir / "summary.json"
    if not path.is_file():
        return f"{name}: no summary.json; completed fold metrics {count}/5 ({run_dir})"
    try:
        summary = read_json(path)
    except (OSError, ValueError) as error:
        return f"{name}: cannot read summary.json ({error}); completed fold metrics {count}/5"
    if summary.get("complete") is not True or summary.get("completed_folds") != 5 or count != 5:
        return (f"{name}: incomplete; summary complete={summary.get('complete')}, "
                f"summary completed_folds={summary.get('completed_folds')}, fold metrics={count}/5")
    return None


def gpu_identity(environment: dict) -> dict:
    """Resolve recorded logical CUDA index through recorded visibility mapping."""
    logical = environment.get("device")
    visible = environment.get("cuda_visible_devices")
    token = None
    if isinstance(logical, str) and logical.startswith("cuda:") and isinstance(visible, str):
        index = int(logical.split(":", 1)[1])
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if index < len(devices):
            token = devices[index]
    return {
        "logical_device": logical,
        "cuda_visible_devices": visible,
        "physical_device_token": token,
        "physical_gpu_ordinal": int(token) if token is not None and token.isdigit() else None,
        "gpu_name": environment.get("gpu"),
        "mapping_source": "environment.json; no GPU is queried by this script",
        "note": "Unrecorded visibility mappings are left unknown rather than inferred from the current machine.",
    }


def data_description(name: str, data_path: Path, environment: dict) -> dict:
    manifest_path = data_path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("dataset") == name, f"{name}: dataset manifest name differs")
    digest = sha256(data_path)
    require(digest == environment["data_sha256"], f"{name}: prepared dataset differs from the training data hash")
    outputs = manifest.get("outputs", manifest.get("output_files", {}))
    require(outputs.get(data_path.name, {}).get("sha256") == digest,
            f"{name}: manifest output hash differs from the prepared dataset")
    frame = pd.read_csv(data_path, sep="\t", usecols=["user_id", "item_id", "skill_id", "correct"])
    require(not frame.isna().any().any() and set(frame.correct.unique()) <= {0, 1},
            f"{name}: invalid dataset IDs or responses")
    actual = dict(students=int(frame.user_id.nunique()), questions=int(frame.item_id.nunique()),
                  interactions=int(len(frame)), model_skill_tokens=int(frame.skill_id.nunique()))
    if name == "assistments09":
        table_skills = int(manifest["stages"]["deduplicate_order_id"]["atomic_skills"])
        manifest_final = manifest["final"]
        require(manifest_final["unique_skill_combinations"] == actual["model_skill_tokens"],
                f"{name}: token count differs from manifest")
        notes = list(manifest.get("interpretation", []))
        status = "expanded corrected.csv; Table 1 counts match; paper sample identity not independently established"
    elif name in ("algebra05", "bridge_algebra06"):
        manifest_final = manifest["final"]
        table_skills = int(manifest_final["atomic_skills"])
        require(manifest_final["unique_skill_combinations"] == actual["model_skill_tokens"],
                f"{name}: token count differs from manifest")
        notes = list(manifest.get("notes", []))
        status = "official original KDD training file; Table 1 counts match; authors' fold identities unavailable"
    elif name in ("spanish", "statics"):
        manifest_final = manifest["stages"][-1]
        table_skills = int(manifest_final["skills"])
        require(table_skills == actual["model_skill_tokens"], f"{name}: token count differs from manifest")
        notes = list(manifest.get("limitations", []))
        status = ("complete public AKT 333-student sequence version; counts match but source/fold identity is not proven"
                  if name == "statics" else "author-published WCRP responses and expert labels; Table 1 counts match")
    else:
        manifest_final = manifest["final"]
        table_skills = int(manifest_final["skills"])
        require(table_skills == actual["model_skill_tokens"], f"{name}: token count differs from manifest")
        notes = list(manifest.get("protocol_notes", []))
        status = "new fixed-seed 5000-student sample; not the paper's exact sample"
    for key in ("students", "questions"):
        require(manifest_final[key] == actual[key], f"{name}: actual {key} differs from manifest")
    manifest_rows = manifest_final.get("interactions", manifest_final.get("rows"))
    require(manifest_rows == actual["interactions"], f"{name}: actual row count differs from manifest")
    actual["skills_table1_convention"] = table_skills
    target = TARGETS[name]
    comparison = {
        key: {"actual": actual["skills_table1_convention"] if key == "skills" else actual[key],
              "paper": target[key],
              "matches": None if target[key] is None else
              (actual["skills_table1_convention"] if key == "skills" else actual[key]) == target[key]}
        for key in ("students", "questions", "skills", "interactions")
    }
    matches = all(item["matches"] is not False for item in comparison.values())
    if name == "sampled_slepemapy":
        require(manifest.get("exact_paper_statistics") == matches,
                "Slepemapy Table 1 comparison differs from its manifest")
    raw_sources = manifest.get("source", manifest.get("source_files"))
    if raw_sources is None:
        raw_sources = {key: manifest[key] for key in
                       ("source_path", "source_url", "raw_sha256", "sampling_seed") if key in manifest}
    return {
        "data_path": str(data_path), "data_sha256": digest,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "actual_statistics_before_last100_truncation": actual,
        "paper_table1_comparison": comparison,
        "matches_paper_table1_statistics": matches,
        "exact_paper_sample_established": False,
        "sample_comparability": status,
        "new_fixed_sample": name == "sampled_slepemapy",
        "sampling_seed": manifest.get("sampling_seed"),
        "public_sequence_version": name == "statics",
        "raw_sources": raw_sources,
        "limitations_and_protocol_notes": notes,
    }


def execution_description(name: str, run_dir: Path, config: dict, environment: dict) -> dict | None:
    path = run_dir / "execution.json"
    if name != "assistments09":
        require(path.is_file(), f"{name}: required GPU7 execution.json is missing")
    if not path.is_file():
        return None  # The historical ASSIST09 run predates the GPU7 launcher.
    execution = read_json(path)
    snapshot = Path(execution["launcher_snapshot"])
    require(not snapshot.is_absolute() and ".." not in snapshot.parts,
            f"{name}: unsafe launcher snapshot path")
    require(sha256(run_dir / snapshot) == execution["launcher_sha256"],
            f"{name}: launcher snapshot hash differs")
    recorded_gpu = gpu_identity(environment)["physical_gpu_ordinal"]
    if recorded_gpu is not None:
        require(execution["physical_gpu"] == recorded_gpu,
                f"{name}: physical GPU differs between environment.json and execution.json")
    require(execution["training_batch_size"] == config["train"]["batch_size"],
            f"{name}: execution batch size differs from config")
    if name != "assistments09":
        require(execution.get("physical_gpu") == 7 and execution.get("cuda_visible_devices") == "7"
                and environment.get("cuda_visible_devices") == "7",
                f"{name}: new experiments must record physical GPU7 and CUDA_VISIBLE_DEVICES=7 in both metadata files")
        require(environment.get("device") == "cuda:0" and execution.get("logical_device") == "cuda:0",
                f"{name}: physical GPU7 must be exposed as logical cuda:0")
        require(execution.get("exit_status") == "success" and execution.get("failure") is None,
                f"{name}: GPU7 launcher has not recorded a successful exit")
        require(execution.get("precision") == "float32" and execution.get("autocast_enabled") is False,
                f"{name}: GPU7 experiment must preserve FP32 with autocast disabled")
        require(execution.get("training_batch_size") == 512 and config["train"]["batch_size"] == 512,
                f"{name}: GPU7 experiment must preserve the paper's training batch size of 512")
        checkpointing = execution.get("activation_checkpointing", {})
        require(checkpointing.get("enabled") is True
                and checkpointing.get("use_reentrant") is False
                and checkpointing.get("preserve_rng_state") is True
                and checkpointing.get("target") == "models.modules.CL4KTTransformerLayer.forward"
                and checkpointing.get("condition") == "module.training and torch.is_grad_enabled()",
                f"{name}: expected enabled nonreentrant transformer activation checkpointing with RNG preservation")
    return execution


def verify_complete_run(name: str, run_dir: Path) -> list[dict]:
    # Import only after completion checks; --help and incomplete-run diagnostics
    # do not import torch. The verifier loads checkpoints with map_location=cpu
    # and checks split isolation/coverage, target order, metrics and snapshots.
    sys.path.insert(0, str(ROOT))
    from scripts.verify_run import VerificationError, verify_run
    try:
        return verify_run(run_dir)
    except (VerificationError, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise ArtifactError(f"{name}: complete read-only verification failed: {error}") from error


def collect_run(name: str, run_dir: Path) -> dict:
    config = read_json(run_dir / "config.json")
    environment = read_json(run_dir / "environment.json")
    summary = read_json(run_dir / "summary.json")
    sources = read_json(run_dir / "source_sha256.json")
    require(config["data_name"] == name and config["num_folds"] == 5,
            f"{name}: incorrect dataset name or fold count in config")
    execution = execution_description(name, run_dir, config, environment)
    folds = verify_complete_run(name, run_dir)
    for fold in folds:
        require(np.isfinite(fold["elapsed_seconds"]) and fold["elapsed_seconds"] >= 0,
                f"{name}: invalid fold elapsed time")
    metrics = {}
    for key in ("auc", "rmse", "acc"):
        values = np.array([fold[key] for fold in folds], dtype=float)
        require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                f"{name}: invalid fold values for {key}")
        mean, std = float(values.mean()), float(values.std(ddof=0))
        close(summary["mean"][key], mean, f"{name}: summary {key} mean")
        close(summary["std_population"][key], std, f"{name}: summary {key} population SD")
        paper_value = TARGETS[name].get(key)
        if summary.get("paper") and key in summary["paper"]:
            close(summary["paper"][key], paper_value, f"{name}: paper target {key}")
        metrics[key] = {
            "mean": mean, "std_population": std, "fold_values": values.tolist(),
            "paper_mean": paper_value, "paper_std": None,
            "delta_reproduced_minus_paper": mean - paper_value if paper_value is not None else None,
        }
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    description = data_description(name, data_path.resolve(), environment)
    execution_path = run_dir / "execution.json"
    return {
        "dataset": name, "paper_dataset_name": TARGETS[name]["paper_name"],
        "display_label": TARGETS[name]["label"], "run_dir": str(run_dir),
        "complete": True, "completed_folds": 5,
        "metrics": metrics, "folds": folds,
        "fold_training_seconds_sum": float(sum(fold["elapsed_seconds"] for fold in folds)),
        "base_seed": config["seed"], "config": config,
        "gpu": gpu_identity(environment), "recorded_environment": environment,
        "recorded_execution": execution,
        "execution_metadata_path": str(execution_path) if execution is not None else None,
        "execution_metadata_sha256": sha256(execution_path) if execution is not None else None,
        "source_sha256": sources, "source_sha256_file": str(run_dir / "source_sha256.json"),
        "full_verification_passed": True,
        "verifier": "scripts.verify_run.verify_run (CPU checkpoint loading; read-only)",
        "data": description,
        "paper_reference": PAPER_REFERENCE,
        "comparison_note": "Positive AUC delta and negative RMSE delta indicate higher performance. No significance claim is inferred.",
    }


def flat_row(run: dict) -> dict:
    data = run["data"]
    actual = data["actual_statistics_before_last100_truncation"]
    row = {"dataset": run["dataset"], "paper_dataset": run["paper_dataset_name"],
           "completed_folds": 5, "run_dir": run["run_dir"]}
    for key, metric in run["metrics"].items():
        row.update({f"{key}_mean": metric["mean"], f"{key}_std_population": metric["std_population"],
                    f"{key}_paper_mean": metric["paper_mean"],
                    f"{key}_delta_reproduced_minus_paper": metric["delta_reproduced_minus_paper"]})
    row.update({
        "base_seed": run["base_seed"],
        "fold_seeds": json.dumps([fold["seed"] for fold in run["folds"]]),
        "fold_best_epochs": json.dumps([fold["best_epoch"] for fold in run["folds"]]),
        "fold_epochs_run": json.dumps([fold["epochs_run"] for fold in run["folds"]]),
        "fold_elapsed_seconds": json.dumps([fold["elapsed_seconds"] for fold in run["folds"]]),
        "fold_training_seconds_sum": run["fold_training_seconds_sum"],
        "batch_size": run["config"]["train"]["batch_size"],
        "sequence_length": run["config"]["train"]["seq_len"],
        "physical_gpu_from_recorded_environment": run["gpu"]["physical_device_token"],
        "cuda_visible_devices": run["gpu"]["cuda_visible_devices"],
        "logical_device": run["gpu"]["logical_device"],
        "data_sha256": data["data_sha256"], "data_path": data["data_path"],
        "data_manifest_sha256": data["manifest_sha256"],
        "data_manifest_path": data["manifest_path"],
        "source_sha256": json.dumps(run["source_sha256"], sort_keys=True),
        "execution_metadata_path": run["execution_metadata_path"],
        "execution_metadata_sha256": run["execution_metadata_sha256"],
        "launcher_sha256": (run["recorded_execution"] or {}).get("launcher_sha256"),
        "activation_checkpointing": json.dumps((run["recorded_execution"] or {}).get("activation_checkpointing")),
        "raw_sources": json.dumps(data["raw_sources"], sort_keys=True),
        "students": actual["students"], "questions": actual["questions"],
        "skills_table1_convention": actual["skills_table1_convention"],
        "model_skill_tokens": actual["model_skill_tokens"], "interactions": actual["interactions"],
        "matches_paper_table1_statistics": data["matches_paper_table1_statistics"],
        "exact_paper_sample_established": data["exact_paper_sample_established"],
        "sample_comparability": data["sample_comparability"],
        "new_fixed_sample": data["new_fixed_sample"], "sampling_seed": data["sampling_seed"],
        "public_sequence_version": data["public_sequence_version"],
        "paper_doi": PAPER_REFERENCE["doi"], "paper_url": PAPER_REFERENCE["url"],
        "paper_metric_table": 2, "paper_statistics_table": 1,
    })
    return row


def make_figure(runs: list[dict], output_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    positions = np.arange(len(runs))
    labels = [run["display_label"] for run in runs]
    for ax, key, ylabel, title in zip(
        axes, ("auc", "rmse"), ("ROC AUC (unitless)", "RMSE (probability scale)"),
        ("AUC: higher is better", "RMSE: lower is better"),
    ):
        paper = [run["metrics"][key]["paper_mean"] for run in runs]
        actual = [run["metrics"][key]["mean"] for run in runs]
        std = [run["metrics"][key]["std_population"] for run in runs]
        ax.scatter(positions - 0.10, paper, color="#555555", marker="D", s=42,
                   label="Paper Table 2 mean", zorder=4)
        ax.errorbar(positions + 0.10, actual, yerr=std, fmt="o", color="#0072B2",
                    markersize=5, capsize=4, linewidth=1.4,
                    label="Reproduced mean ± fold population SD", zorder=4)
        for x, original, reproduced in zip(positions, paper, actual):
            ax.plot([x - 0.10, x + 0.10], [original, reproduced], color="#bbbbbb", linewidth=0.8, zorder=2)
        slep_index = next(index for index, run in enumerate(runs) if run["dataset"] == "sampled_slepemapy")
        ax.axvspan(slep_index - 0.42, slep_index + 0.42, color="#cccccc", alpha=0.18, zorder=0)
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.23)
        ax.margins(x=0.06, y=0.20)
        lower, upper = ax.get_ylim()
        ax.set_ylim(max(0, lower), min(1, upper))
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    figure.suptitle("CL4KT: six datasets, student-level five-fold cross-validation", fontsize=14)
    figure.text(0.5, 0.02,
                "Error bars are population SD across the five reproduced folds; paper SD is unavailable.\n"
                "* Slepemapy uses a new fixed-seed sample, not the paper's exact sample.\n"
                "† Statics uses the complete public AKT sequence version; matching counts do not establish identical author data/folds.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.15, 1, 0.95))
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / f"all_datasets_comparison.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assistments-run", type=Path, default=ROOT / "results/assistments09-fivefold")
    parser.add_argument("--gpu7-root", type=Path, default=ROOT / "results/gpu7")
    parser.add_argument("--output-dir", type=Path, help="Defaults to --gpu7-root")
    parser.add_argument("--no-figures", action="store_true", help="Write CSV/JSON only")
    args = parser.parse_args()
    gpu7_root = args.gpu7_root.expanduser().resolve()
    paths = {name: (args.assistments_run.expanduser().resolve() if name == "assistments09"
                    else gpu7_root / f"{name}-fivefold") for name in TARGETS}
    incomplete = [status for name, path in paths.items() if (status := completion_status(name, path))]
    if incomplete:
        print("Six-dataset summary not written: all six runs must finish five folds.", file=sys.stderr)
        print("\n".join(incomplete), file=sys.stderr)
        return 2
    try:
        runs = [collect_run(name, path) for name, path in paths.items()]
    except (OSError, ValueError, KeyError, ArtifactError) as error:
        print(f"Six-dataset summary not written: {error}", file=sys.stderr)
        return 1
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else gpu7_root
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": True, "datasets": 6, "folds_per_dataset": 5,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "paper_reference": PAPER_REFERENCE,
        "aggregation": "Unweighted mean and population SD (ddof=0) of five per-fold metrics; pooled AUC is not substituted.",
        "gpu_note": "GPU identity is taken separately from each run's recorded environment. ASSIST09 may predate the GPU7-only request.",
        "scope": "Six fixed-configuration CL4KT experiments; this does not reproduce all baselines, ablations or significance tests.",
        "runs": runs,
    }
    csv_path, json_path = output_dir / "all_datasets.csv", output_dir / "all_datasets.json"
    rows = [flat_row(run) for run in runs]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs = [str(csv_path), str(json_path)]
    if not args.no_figures:
        outputs.extend(make_figure(runs, output_dir))
    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
