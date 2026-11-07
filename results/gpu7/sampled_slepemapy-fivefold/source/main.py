"""Reproducible single-device CL4KT cross-validation with auditable artifacts."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data_loaders import MostRecentQuestionSkillDataset, SimCLRDatasetWrapper
from models.cl4kt import CL4KT

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # These kernels are deterministic on the recorded environment. Warn for any
    # operation without a deterministic implementation rather than silently vary.
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return tuple(move_batch(value, device) for value in batch)
    return batch


def make_splits(df, config):
    """Match upstream's initial user shuffle and student-level KFold structure."""
    users = df.user_id.unique().copy()
    rng = np.random.RandomState(config["seed"])
    rng.shuffle(users)
    result = []
    kfold = KFold(n_splits=config["num_folds"], shuffle=True, random_state=config["seed"])
    for fold, (train_idx, test_idx) in enumerate(kfold.split(users)):
        train_users = users[train_idx].copy()
        rng.shuffle(train_users)
        offset = int(len(train_users) * (1 - config["validation_fraction"]))
        split = dict(fold=fold, train=train_users[:offset].tolist(),
                     valid=train_users[offset:].tolist(), test=users[test_idx].tolist())
        a, b, c = (set(split[key]) for key in ("train", "valid", "test"))
        assert not (a & b or a & c or b & c)
        assert a | b | c == set(users)
        result.append(split)
    assert sorted(user for split in result for user in split["test"]) == sorted(users)
    return result


def make_loaders(df, split, config, num_skills, num_questions):
    train_cfg, model_cfg = config["train"], config["model"]
    loaders, stats = {}, {}
    for part in ("train", "valid", "test"):
        subset = df[df.user_id.isin(split[part])]
        dataset = MostRecentQuestionSkillDataset(
            subset, train_cfg["seq_len"], num_skills, num_questions)
        probs = [model_cfg[key] if part == "train" else 0.0 for key in
                 ("mask_prob", "crop_prob", "permute_prob", "replace_prob", "negative_prob")]
        wrapped = SimCLRDatasetWrapper(dataset, train_cfg["seq_len"], *probs,
                                       eval_mode=part != "train")
        # Upstream leaves shuffle=False; keep the same in-batch negative policy.
        loaders[part] = DataLoader(wrapped, batch_size=train_cfg[
            "batch_size" if part == "train" else "eval_batch_size"],
            shuffle=False, num_workers=train_cfg["num_workers"])
        stats[part] = dict(students=len(dataset), retained_interactions=dataset.num_interactions,
                           scored_interactions=int((dataset.padded_r[:, 1:] >= 0).sum()))
    return loaders, stats


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    for batch in loader:
        output = model(move_batch(batch, device))
        true, pred = output["true"].flatten(), output["pred"].flatten()
        mask = true >= 0
        predictions.append(pred[mask].cpu().numpy())
        labels.append(true[mask].cpu().numpy())
    pred, true = np.concatenate(predictions), np.concatenate(labels)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Non-finite prediction")
    return dict(auc=float(roc_auc_score(true, pred)),
                rmse=float(np.sqrt(mean_squared_error(true, pred))),
                acc=float(accuracy_score(true, pred >= 0.5)),
                count=len(true)), pred, true


def run_fold(df, split, config, output_dir, device, num_skills, num_questions):
    fold = split["fold"]
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)
    if (fold_dir / "metrics.json").exists():
        raise FileExistsError(f"Completed fold exists: {fold_dir}; use --resume or a new output directory")
    seed = config["seed"] + fold
    seed_everything(seed)
    loaders, stats = make_loaders(df, split, config, num_skills, num_questions)
    write_json(fold_dir / "split.json", split)
    write_json(fold_dir / "data_stats.json", stats)
    model = CL4KT(num_skills, num_questions, config["train"]["seq_len"], **config["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"],
                           weight_decay=config["model"]["l2"])
    best_auc, best_epoch = -float("inf"), 0
    started = time.monotonic()
    with (fold_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "valid_auc", "valid_rmse", "elapsed_seconds"])
        writer.writeheader()
        for epoch in range(1, config["train"]["num_epochs"] + 1):
            model.train()
            losses = []
            for batch in loaders["train"]:
                batch = move_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                output = model(batch)
                loss, _, _ = model.loss(batch, output)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in fold {fold}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["max_grad_norm"], error_if_nonfinite=True)
                opt.step()
                losses.append(loss.item())
            valid, _, _ = evaluate(model, loaders["valid"], device)
            elapsed = time.monotonic() - started
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), valid_auc=valid["auc"],
                       valid_rmse=valid["rmse"], elapsed_seconds=round(elapsed, 3))
            writer.writerow(row)
            handle.flush()
            if valid["auc"] > best_auc:
                best_auc, best_epoch = valid["auc"], epoch
                torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, valid_auc=best_auc,
                                config=config, fold=fold, seed=seed, num_skills=num_skills,
                                num_questions=num_questions), fold_dir / "best.pt")
            print(f"fold={fold} epoch={epoch:03d} loss={row['train_loss']:.5f} "
                  f"val_auc={valid['auc']:.5f} best={best_auc:.5f}@{best_epoch} elapsed={elapsed:.1f}s", flush=True)
            if epoch - best_epoch >= config["train"]["patience"]:
                break
    checkpoint = torch.load(fold_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Test labels are accessed for scoring only after model selection is over.
    metrics, pred, true = evaluate(model, loaders["test"], device)
    np.savez_compressed(fold_dir / "predictions.npz", prediction=pred, target=true)
    metrics.update(fold=fold, seed=seed, best_epoch=best_epoch, best_valid_auc=best_auc,
                   epochs_run=epoch, elapsed_seconds=round(time.monotonic() - started, 3),
                   checkpoint_sha256=sha256(fold_dir / "best.pt"))
    write_json(fold_dir / "metrics.json", metrics)
    print(f"TEST fold={fold}: {json.dumps(metrics)}", flush=True)
    del model, opt, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assistments09.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0 (select physical GPU with CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--folds", type=int, nargs="+", help="Zero-based fold IDs; default all five")
    parser.add_argument("--epochs", type=int, help="Override epoch cap, intended for smoke runs")
    parser.add_argument("--resume", action="store_true", help="Skip completed folds, restart any interrupted fold")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.epochs is not None:
        config["train"]["num_epochs"] = args.epochs
    for name in ("num_epochs", "patience", "seq_len", "batch_size", "eval_batch_size", "cpu_threads"):
        if config["train"][name] < 1:
            parser.error(f"{name} must be positive")
    torch.set_num_threads(config["train"]["cpu_threads"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        parser.error("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=<GPU ID> to avoid using other jobs' devices")
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        parser.error(f"Missing {data_path}; run scripts/prepare_assistments09.py first")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or ROOT / "results" / f"{config['data_name']}-{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        if not args.resume or json.loads((output_dir / "config.json").read_text()) != config:
            parser.error("Existing run directory; --resume requires identical resolved config")
    data_digest = sha256(data_path)
    source_files = ["main.py", "data_loaders.py", "utils/augment_seq.py", "models/cl4kt.py", "models/modules.py"]
    source_digests = {name: sha256(ROOT / name) for name in source_files}
    if args.resume and (output_dir / "environment.json").exists():
        previous_environment = json.loads((output_dir / "environment.json").read_text())
        previous_sources = json.loads((output_dir / "source_sha256.json").read_text())
        if previous_environment["data_sha256"] != data_digest or previous_sources != source_digests:
            parser.error("Cannot resume after dataset or source changes; use a new output directory")
    write_json(output_dir / "config.json", config)
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                       pandas=pd.__version__, sklearn=sklearn.__version__, device=str(device),
                       cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                       cublas_workspace_config=os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                       deterministic_algorithms="enabled, warn_only=True",
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       data_sha256=data_digest, upstream_commit="f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2",
                       recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), command=sys.argv)
    if not (output_dir / "environment.json").exists():
        write_json(output_dir / "environment.json", environment)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output_dir / "requirements.freeze.txt").write_text(freeze.stdout)
        write_json(output_dir / "source_sha256.json", source_digests)
        for name in source_files:
            destination = output_dir / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
    df = pd.read_csv(data_path, sep="\t")
    if df.isna().any().any() or not set(df.correct.unique()) <= {0, 1}:
        raise ValueError("Invalid preprocessed dataset")
    splits = make_splits(df, config)
    write_json(output_dir / "splits.json", splits)
    df["skill_id"] += 1
    df["item_id"] += 1
    num_skills, num_questions = int(df.skill_id.max()) + 1, int(df.item_id.max()) + 1
    folds = args.folds if args.folds is not None else list(range(config["num_folds"]))
    if len(set(folds)) != len(folds) or any(fold not in range(config["num_folds"]) for fold in folds):
        parser.error("Fold IDs must be unique and in range")
    print(f"Run directory: {output_dir}\nDevice: {device}; folds: {folds}", flush=True)
    for fold in folds:
        if args.resume and (output_dir / f"fold_{fold}" / "metrics.json").exists():
            continue
        run_fold(df, splits[fold], config, output_dir, device, num_skills, num_questions)
    results = [json.loads(path.read_text()) for path in sorted(output_dir.glob("fold_*/metrics.json"))]
    summary = dict(complete=len(results) == config["num_folds"], completed_folds=len(results), folds=results,
                   paper=dict(auc=0.7624, rmse=0.4333) if config["data_name"] == "assistments09" else None)
    if results:
        summary["mean"] = {key: float(np.mean([row[key] for row in results])) for key in ("auc", "rmse", "acc")}
        summary["std_population"] = {key: float(np.std([row[key] for row in results])) for key in ("auc", "rmse", "acc")}
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
