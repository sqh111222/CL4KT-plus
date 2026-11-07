#!/usr/bin/env python3
"""Independently verify a completed five-fold run without evaluating a model.

Usage: .venv/bin/python scripts/verify_run.py --run-dir results/assistments09-fivefold
The command is read-only. Missing or inconsistent artifacts produce a nonzero
exit status and a diagnostic; partially written runs are never reported as valid.
Frozen source snapshots must match each run's recorded hashes. Current workspace
source differences are reported and can optionally be rejected with
--strict-current-source; they do not invalidate an intact older experiment.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "main.py", "data_loaders.py", "utils/augment_seq.py",
    "models/cl4kt.py", "models/modules.py",
}


class VerificationError(Exception):
    """An artifact is missing or contradicts another recorded artifact."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, label):
    require(np.isfinite(actual) and np.isfinite(expected),
            f"{label}: non-finite value ({actual!r}, {expected!r})")
    require(np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
            f"{label}: recorded={actual!r}, recomputed={expected!r}")


def verify_sources(run_dir, recorded_sources, project_root=ROOT, strict_current_source=False):
    """Always verify frozen source; optionally require the current source too."""
    require(set(recorded_sources) == SOURCE_FILES, "Unexpected/missing recorded core source files")
    differences = []
    for name, recorded in recorded_sources.items():
        snapshot = Path(run_dir) / "source" / name
        require(snapshot.is_file(), f"Frozen source snapshot is missing: {snapshot}")
        require(sha256(snapshot) == recorded, f"Frozen source hash differs: {name}")
        current = Path(project_root) / name
        if not current.is_file() or sha256(current) != recorded:
            differences.append(name)
    require(not (strict_current_source and differences),
            f"Current source differs from this run's snapshot: {', '.join(sorted(differences))}")
    if differences:
        print("NOTE current workspace sources differ from this run: "
              + ", ".join(sorted(differences))
              + ". Verification uses the run's hash-verified frozen snapshots.")
    return differences


def verify_execution(run_dir, config, environment, project_root=ROOT, strict_current_source=False):
    """Check a GPU7 runtime patch and its frozen provenance, when present."""
    run_dir = Path(run_dir)
    path = run_dir / "execution.json"
    command = environment.get("command", [])
    launched_with_gpu7 = bool(command and Path(str(command[0])).name == "train_gpu7.py")
    if not path.is_file():
        require(not launched_with_gpu7 and not (run_dir / "launcher" / "train_gpu7.py").exists(),
                "GPU7 launcher run is missing execution.json")
        return None  # Historical runs executed main.py directly.
    execution = read_json(path)
    require(execution.get("schema_version") == 1, "Unsupported execution metadata schema")
    require(execution.get("launcher_snapshot") == "launcher/train_gpu7.py",
            "Unexpected GPU7 launcher snapshot path")
    snapshot = run_dir / "launcher" / "train_gpu7.py"
    require(snapshot.is_file(), "Frozen GPU7 launcher snapshot is missing")
    require(sha256(snapshot) == execution.get("launcher_sha256"),
            "Frozen GPU7 launcher hash differs from execution metadata")
    current = Path(project_root) / "scripts" / "train_gpu7.py"
    current_differs = not current.is_file() or sha256(current) != execution["launcher_sha256"]
    require(not (strict_current_source and current_differs),
            "Current GPU7 launcher differs from this run's snapshot")
    if current_differs:
        print("NOTE current GPU7 launcher differs from this run. Verification uses its hash-verified frozen snapshot.")
    require(execution.get("physical_gpu") == 7 and execution.get("cuda_visible_devices") == "7"
            and environment.get("cuda_visible_devices") == "7",
            "GPU7 execution/environment must record physical_gpu=7 and CUDA_VISIBLE_DEVICES=7")
    require(execution.get("cuda_device_order") == "PCI_BUS_ID",
            "GPU7 execution must record CUDA_DEVICE_ORDER=PCI_BUS_ID")
    require(execution.get("logical_device") == "cuda:0" and environment.get("device") == "cuda:0",
            "GPU7 execution/environment must use logical device cuda:0")
    require(execution.get("training_batch_size") == config["train"]["batch_size"] == 512,
            "GPU7 execution must preserve training batch_size=512")
    require(execution.get("precision") == "float32" and execution.get("autocast_enabled") is False,
            "GPU7 execution must preserve FP32 with autocast disabled")
    checkpointing = execution.get("activation_checkpointing", {})
    require(checkpointing.get("enabled") is True and checkpointing.get("use_reentrant") is False
            and checkpointing.get("preserve_rng_state") is True
            and checkpointing.get("target") == "models.modules.CL4KTTransformerLayer.forward"
            and checkpointing.get("condition") == "module.training and torch.is_grad_enabled()",
            "GPU7 activation checkpointing must be non-reentrant, preserve RNG, and wrap training forwards only")
    require(execution.get("exit_status") == "success" and execution.get("failure") is None,
            "GPU7 execution has not finished successfully")
    require(isinstance(execution.get("recorded_at"), str) and isinstance(execution.get("finished_at"), str),
            "GPU7 execution must record start and finish timestamps")
    started = datetime.fromisoformat(execution["recorded_at"])
    finished = datetime.fromisoformat(execution["finished_at"])
    require(started.tzinfo is not None and finished.tzinfo is not None and finished >= started,
            "GPU7 execution timestamps must have timezones and finish after start")
    for key in ("torch", "cuda"):
        require(execution.get(key) is not None and execution[key] == environment.get(key),
                f"GPU7 execution/environment {key} versions differ")
    require(execution.get("gpu_name") == environment.get("gpu"),
            "GPU7 execution/environment GPU names differ")
    require(execution.get("seed") == config["seed"], "GPU7 execution seed differs from config")
    driver = execution.get("physical_gpu_details", {})
    require(driver.get("index") == 7, "GPU7 driver metadata must identify physical index 7")
    require(isinstance(driver.get("pci_bus_id"), str) and
            re.fullmatch(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", driver["pci_bus_id"]),
            "GPU7 driver metadata must contain a valid PCI bus ID")

    def normalized_uuid(value, label):
        require(isinstance(value, str), f"Missing GPU7 {label} UUID")
        try:
            return str(uuid.UUID(value.removeprefix("GPU-")))
        except ValueError as error:
            raise VerificationError(f"Invalid GPU7 {label} UUID: {value!r}") from error

    require(normalized_uuid(driver.get("uuid"), "driver") ==
            normalized_uuid(execution.get("gpu_uuid"), "torch"),
            "GPU7 driver and torch UUIDs differ")
    allocated = execution.get("peak_cuda_memory_allocated")
    reserved = execution.get("peak_cuda_memory_reserved")
    require(type(allocated) is int and type(reserved) is int and 0 < allocated <= reserved,
            "GPU7 peak CUDA memory byte counts must satisfy 0 < allocated <= reserved")
    print(f"PASS GPU7 runtime: frozen launcher, PCI/UUID binding, FP32/batch512, RNG-preserving checkpointing; "
          f"peak allocated={allocated} bytes, reserved={reserved} bytes")
    return execution


def verify_run(run_dir, strict_current_source=False):
    run_dir = Path(run_dir).resolve()
    require(run_dir.is_dir(), f"Run directory does not exist: {run_dir}")
    top_files = ("config.json", "environment.json", "source_sha256.json",
                 "splits.json", "summary.json")
    fold_files = ("metrics.json", "split.json", "data_stats.json", "history.csv",
                  "best.pt", "predictions.npz")
    missing = [name for name in top_files if not (run_dir / name).is_file()]
    complete = []
    for fold in range(5):
        absent = [name for name in fold_files if not (run_dir / f"fold_{fold}" / name).is_file()]
        missing.extend(f"fold_{fold}/{name}" for name in absent)
        if not absent:
            complete.append(fold)
    require(not missing,
            f"Incomplete run: completed folds {complete}/[0, 1, 2, 3, 4]; "
            f"missing artifacts: {', '.join(missing)}. Wait for training to finish.")

    config = read_json(run_dir / "config.json")
    environment = read_json(run_dir / "environment.json")
    sources = read_json(run_dir / "source_sha256.json")
    splits = read_json(run_dir / "splits.json")
    summary = read_json(run_dir / "summary.json")
    require(config["num_folds"] == 5, "Paper verification requires num_folds=5")
    verify_sources(run_dir, sources, strict_current_source=strict_current_source)
    verify_execution(run_dir, config, environment, strict_current_source=strict_current_source)
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    require(data_path.is_file(), f"Recorded dataset does not exist: {data_path}")
    require(sha256(data_path) == environment["data_sha256"], "Dataset SHA256 differs")
    df = pd.read_csv(data_path, sep="\t")
    require({"user_id", "item_id", "skill_id", "correct"} <= set(df.columns),
            "Dataset is missing required columns")
    require(not df[["user_id", "item_id", "skill_id", "correct"]].isna().any().any(),
            "Dataset contains missing IDs or responses")
    require(set(df["correct"].unique()) <= {0, 1}, "Dataset contains non-binary responses")
    for name in ("item_id", "skill_id"):
        values = df[name].to_numpy()
        require(np.isfinite(values).all() and (values >= 0).all() and
                np.equal(values, np.floor(values)).all(),
                f"Dataset {name} must contain finite non-negative integer IDs")
    users = set(df["user_id"].unique().tolist())
    require(len(splits) == 5 and [split["fold"] for split in splits] == list(range(5)),
            "Top-level splits must contain fold IDs 0..4 exactly once, in order")
    test_coverage = Counter()
    for split in splits:
        groups = []
        for part in ("train", "valid", "test"):
            ids = split[part]
            require(ids and len(ids) == len(set(ids)),
                    f"fold {split['fold']} {part}: empty or duplicate student IDs")
            groups.append(set(ids))
        train, valid, test = groups
        require(not (train & valid or train & test or valid & test),
                f"fold {split['fold']}: student leakage between partitions")
        require(train | valid | test == users,
                f"fold {split['fold']}: partitions do not cover dataset students exactly")
        test_coverage.update(test)
    require(set(test_coverage) == users and all(count == 1 for count in test_coverage.values()),
            "Each dataset student must occur in a test fold exactly once")

    results = []
    seq_len = config["train"]["seq_len"]
    require(isinstance(seq_len, int) and seq_len >= 2, "seq_len must be an integer >=2")
    expected_skills = int(df["skill_id"].max()) + 2
    expected_questions = int(df["item_id"].max()) + 2
    for fold, split in enumerate(splits):
        directory = run_dir / f"fold_{fold}"
        require(read_json(directory / "split.json") == split,
                f"fold {fold}: local split differs from top-level split")
        metrics = read_json(directory / "metrics.json")
        stats = read_json(directory / "data_stats.json")
        require(metrics["fold"] == fold, f"fold {fold}: metrics has incorrect fold ID")
        require(metrics["seed"] == config["seed"] + fold,
                f"fold {fold}: incorrect recorded seed")
        for part in ("train", "valid", "test"):
            lengths = df.loc[df.user_id.isin(split[part])].groupby("user_id").size()
            retained = lengths.clip(upper=seq_len)
            # Official left padding excludes sequence position zero. A short
            # student's first valid response remains scored, exactly as trained.
            scored = retained - (retained == seq_len).astype(int)
            require(stats[part]["students"] == len(lengths),
                    f"fold {fold} {part}: student count mismatch")
            require(stats[part]["retained_interactions"] == int(retained.sum()),
                    f"fold {fold} {part}: retained interaction count mismatch")
            require(stats[part]["scored_interactions"] == int(scored.sum()),
                    f"fold {fold} {part}: scored interaction count mismatch")
        with np.load(directory / "predictions.npz", allow_pickle=False) as values:
            require(set(values.files) == {"prediction", "target"},
                    f"fold {fold}: unexpected prediction archive keys")
            pred, target = values["prediction"], values["target"]
        require(pred.ndim == target.ndim == 1 and pred.shape == target.shape and pred.size > 0,
                f"fold {fold}: prediction/target arrays must be matching nonempty vectors")
        require(np.isfinite(pred).all() and np.isfinite(target).all(),
                f"fold {fold}: non-finite predictions or targets")
        require(np.all((pred >= 0) & (pred <= 1)) and set(np.unique(target)) == {0, 1},
                f"fold {fold}: invalid probabilities or non-binary/single-class targets")
        require(pred.size == metrics["count"] == stats["test"]["scored_interactions"],
                f"fold {fold}: prediction count disagrees with recorded test count")
        # Independently reconstruct target order from data, without loading a
        # model or using labels for any selection or parameter tuning.
        expected_target = []
        for _, student in df.loc[df.user_id.isin(split["test"])].groupby("user_id"):
            answers = student["correct"].to_numpy()[-seq_len:]
            expected_target.extend(answers[1:] if len(answers) == seq_len else answers)
        require(np.array_equal(target, np.asarray(expected_target)),
                f"fold {fold}: saved targets disagree with dataset order")
        recomputed = {
            "auc": float(roc_auc_score(target, pred)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred))),
            "acc": float(accuracy_score(target, pred >= 0.5)),
        }
        for name, value in recomputed.items():
            close(metrics[name], value, f"fold {fold} {name}")
        with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        require(history, f"fold {fold}: empty training history")
        epochs = [int(row["epoch"]) for row in history]
        require(epochs == list(range(1, len(history) + 1)),
                f"fold {fold}: history epochs are not consecutive starting at 1")
        require(metrics["epochs_run"] == len(history) <= config["train"]["num_epochs"],
                f"fold {fold}: epochs_run differs from history or exceeds configured cap")
        valid_aucs = np.asarray([float(row["valid_auc"]) for row in history])
        require(np.isfinite(valid_aucs).all(), f"fold {fold}: non-finite validation AUC")
        best = int(np.argmax(valid_aucs)) + 1
        require(metrics["best_epoch"] == best,
                f"fold {fold}: best_epoch is not the first maximum validation AUC epoch")
        close(metrics["best_valid_auc"], valid_aucs[best - 1], f"fold {fold} best_valid_auc")
        require(sha256(directory / "best.pt") == metrics["checkpoint_sha256"],
                f"fold {fold}: checkpoint SHA256 mismatch")
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        require(checkpoint["epoch"] == best and checkpoint["fold"] == fold,
                f"fold {fold}: checkpoint epoch/fold does not match model selection")
        require(checkpoint["config"] == config and checkpoint["seed"] == metrics["seed"],
                f"fold {fold}: checkpoint config/seed mismatch")
        require(checkpoint["num_skills"] == expected_skills and
                checkpoint["num_questions"] == expected_questions,
                f"fold {fold}: checkpoint vocabulary sizes disagree with dataset")
        close(checkpoint["valid_auc"], metrics["best_valid_auc"], f"fold {fold} checkpoint valid_auc")
        require(bool(checkpoint["model_state_dict"]), f"fold {fold}: empty checkpoint state")
        results.append(metrics)
        print(f"PASS fold {fold}: students={len(split['test'])}, predictions={pred.size}, "
              f"best_epoch={best}, auc={recomputed['auc']:.6f}, rmse={recomputed['rmse']:.6f}")

    require(summary["complete"] is True and summary["completed_folds"] == 5,
            "Summary does not mark all five folds complete")
    require(summary["folds"] == results, "Summary fold records differ from per-fold metrics")
    for name in ("auc", "rmse", "acc"):
        values = [row[name] for row in results]
        close(summary["mean"][name], float(np.mean(values)), f"summary mean {name}")
        close(summary["std_population"][name], float(np.std(values, ddof=0)),
              f"summary population std {name}")
    print("PASS five-fold coverage, split isolation, prediction metrics, validation selection, "
          "checkpoint hashes, summary statistics, dataset and source hashes")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strict-current-source", action="store_true",
                        help="Also reject current source differences; frozen snapshots are always verified")
    args = parser.parse_args()
    try:
        verify_run(args.run_dir, strict_current_source=args.strict_current_source)
    except (VerificationError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
