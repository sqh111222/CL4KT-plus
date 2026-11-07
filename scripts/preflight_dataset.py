#!/usr/bin/env python3
"""Check all student partitions against CL4KT loader and AUC requirements."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import make_splits


def preflight(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    data_path = ROOT / config["data_path"]
    frame = pd.read_csv(data_path, sep="\t")
    required = {"user_id", "item_id", "skill_id", "correct"}
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Missing or null required columns")
    for name in ("user_id", "item_id", "skill_id"):
        values = frame[name].to_numpy()
        if not np.isfinite(values).all() or np.any(values < 0) or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Invalid nonnegative integer IDs in {name}")
    if set(frame.correct.unique()) != {0, 1}:
        raise ValueError("Need binary labels with both classes")
    seq_len = config["train"]["seq_len"]
    profiles = {}
    for user, group in frame.groupby("user_id"):
        if len(group) < 1:
            raise ValueError("Empty student sequence")
        if "timestamp" in group and not group.timestamp.is_monotonic_increasing:
            raise ValueError(f"Student {user} has unsorted timestamps")
        recent = group.iloc[-seq_len:]
        scored = recent.iloc[1:] if len(recent) == seq_len else recent
        profiles[user] = dict(skills=set(recent.skill_id), count=len(scored), positive=int(scored.correct.sum()))
    checks = []
    for split in make_splits(frame, config):
        for part in ("train", "valid", "test"):
            users = split[part]
            if not users:
                raise ValueError(f"Empty fold {split['fold']} {part}")
            skills = set().union(*(profiles[user]["skills"] for user in users))
            count = sum(profiles[user]["count"] for user in users)
            positive = sum(profiles[user]["positive"] for user in users)
            if len(skills) < 2:
                raise ValueError(f"Loader needs at least two skills: fold {split['fold']} {part}")
            if not 0 < positive < count:
                raise ValueError(f"AUC needs both scored label classes: fold {split['fold']} {part}")
            checks.append(dict(fold=split["fold"], partition=part, students=len(users), skills=len(skills),
                               predictions=count, positives=positive))
    return dict(data_name=config["data_name"], data_path=str(data_path),
                students=len(profiles), interactions=len(frame), checked_partitions=checks, passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = preflight(args.config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checked_partitions"}, indent=2))
