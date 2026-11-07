#!/usr/bin/env python3
"""Prepare the paper's two KDD datasets with unmodified CL4KT preprocessing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocess_data  # noqa: E402

UPSTREAM_COMMIT = "f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2"
SETTINGS = {
    "algebra05": {
        "archive": "algebra_2005_2006.zip",
        "member": "algebra_2005_2006_train.txt",
        "kc_column": "KC(Default)",
        "paper": {"students": 571, "questions": 173113, "atomic_skills": 112, "interactions": 607014},
    },
    "bridge_algebra06": {
        "archive": "bridge_to_algebra_2006_2007.zip",
        "member": "bridge_to_algebra_2006_2007_train.txt",
        "kc_column": "KC(SubSkills)",
        "paper": {"students": 1138, "questions": 129263, "atomic_skills": 493, "interactions": 1817450},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CL4KT-reproduction/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                shutil.copyfileobj(response, handle)
            handle.flush()
            if not zipfile.is_zipfile(temporary):
                raise ValueError(f"Downloaded response is not a ZIP archive: {url}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def install_raw(source: Path, destination: Path, member: str) -> str | None:
    if not source.is_file():
        raise FileNotFoundError(source)
    chosen_member = None
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".raw", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            if zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as archive:
                    if member not in archive.namelist():
                        raise ValueError(f"Expected ZIP member {member!r}; found {archive.namelist()}")
                    with archive.open(member) as original:
                        shutil.copyfileobj(original, handle)
                chosen_member = member
            else:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, handle)
            handle.flush()
            if destination.exists():
                if sha256(destination) != sha256(temporary):
                    raise FileExistsError(f"Refusing to overwrite a different raw file: {destination}")
            else:
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return chosen_member


def stage_stats(frame: pd.DataFrame, kc_column: str) -> dict:
    skills = {skill for combo in frame[kc_column].dropna().unique() for skill in combo.split("~~")}
    return {
        "interactions": int(len(frame)),
        "students": int(frame["user_id"].nunique()),
        "questions": int(frame["item_id"].nunique()),
        "atomic_skills": len(skills),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", choices=SETTINGS, required=True)
    parser.add_argument("--raw", type=Path, help="Original training TXT or official ZIP; otherwise fetch/reuse the official ZIP")
    parser.add_argument("--output-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--min-user-inter-num", type=int, default=5)
    args = parser.parse_args()
    if args.min_user_inter_num < 1:
        parser.error("--min-user-inter-num must be positive")
    settings = SETTINGS[args.data_name]
    directory = args.output_root.expanduser().resolve() / args.data_name
    directory.mkdir(parents=True, exist_ok=True)
    url = "https://kdd.org/cupfiles/" + settings["archive"]
    source = args.raw.expanduser().resolve() if args.raw else directory / settings["archive"]
    downloaded = False
    if not args.raw and not source.exists():
        print(f"Downloading official archive: {url}", flush=True)
        fetch(url, source)
        downloaded = True
    member = install_raw(source, directory / "data.txt", settings["member"])
    kc_column = settings["kc_column"]
    columns = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time", "Correct First Attempt", kc_column]
    frame = pd.read_csv(directory / "data.txt", sep="\t", usecols=columns)
    frame.rename(columns={"Anon Student Id": "user_id", "Correct First Attempt": "correct"}, inplace=True)
    frame["item_id"] = frame["Problem Name"] + ":" + frame["Step Name"]
    stages = {"raw_training_file": stage_stats(frame, kc_column)}
    print(f"Raw training file: {stages['raw_training_file']}", flush=True)
    # Use the same timestamp conversion and ordering of filters as the source.
    timestamp = pd.to_datetime(frame["First Transaction Time"])
    timestamp = timestamp - timestamp.min()
    frame["timestamp"] = timestamp.apply(lambda value: value.total_seconds()).astype(np.int64)
    frame = frame[frame["correct"].isin([0, 1])]
    stages["binary_correctness"] = stage_stats(frame, kc_column)
    frame = frame[frame[kc_column].notna()]
    stages["remove_missing_skills"] = stage_stats(frame, kc_column)
    frame = frame.drop_duplicates(["user_id", "item_id", "timestamp"])
    stages["deduplicate_user_item_timestamp"] = stage_stats(frame, kc_column)
    frame = frame.groupby("user_id").filter(lambda group: len(group) >= args.min_user_inter_num)
    stages["minimum_interactions"] = stage_stats(frame, kc_column)
    print(f"After upstream filters: {stages['minimum_interactions']}", flush=True)
    del frame, timestamp
    gc.collect()

    print(f"Running unmodified upstream preprocessing for {args.data_name}", flush=True)
    preprocess_data.BASE_PATH = str(directory.parent)
    preprocess_data.prepare_kddcup10(
        data_name=args.data_name,
        min_user_inter_num=args.min_user_inter_num,
        kc_col_name=kc_column,
        remove_nan_skills=True,
    )
    processed = pd.read_csv(directory / "preprocessed_df.csv", sep="\t")
    final = {
        "students": int(processed.user_id.nunique()),
        "questions": int(processed.item_id.nunique()),
        "atomic_skills": stages["minimum_interactions"]["atomic_skills"],
        "interactions": int(len(processed)),
        "unique_skill_combinations": int(processed.skill_id.nunique()),
        "minimum_interactions_per_student": int(processed.groupby("user_id").size().min()),
    }
    paper = settings["paper"]
    matches = all(final[key] == value for key, value in paper.items())
    outputs = {}
    for name in ("preprocessed_df.csv", "original_df.csv", "q_mat.npz", "question_skill_rel.pkl"):
        path = directory / name
        outputs[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "dataset": args.data_name,
        "paper_dataset_name": "algebra06" if args.data_name == "bridge_algebra06" else args.data_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_path": str(source), "input_sha256": sha256(source),
            "official_archive_url": url,
            "input_origin": "official SIGKDD archive at default archive path" if not args.raw else "caller-supplied local file; source URL is a retrieval option",
            "downloaded_by_script_in_this_invocation": downloaded,
            "zip_member": member,
            "raw_training_file_sha256": sha256(directory / "data.txt"),
            "dataset_page": "https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data",
            "selected_partition": "original KDD development dataset training TXT only; master/test not concatenated",
        },
        "upstream_commit": UPSTREAM_COMMIT,
        "preprocessing_module_sha256": sha256(Path(preprocess_data.__file__)),
        "protocol": {"min_user_inter_num": args.min_user_inter_num, "remove_nan_skills": True,
                     "kc_column": kc_column, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                     "implementation": "unmodified prepare_kddcup10; dense original Q matrix and np.unique"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "stages": stages,
        "final": final,
        "paper_table1": paper,
        "matches_paper_table1": matches,
        "notes": [
            "Paper skill counts are atomic KCs before merging; the model uses unique item-to-KC combinations.",
            "PYTHONHASHSEED=0 fixes the upstream set iteration order when assigning KC columns and merged skill IDs.",
            "Raw and processed interaction counts precede the training-time truncation to the latest 100 interactions per student.",
            "The original dataset is retained; no synthetic data, merged test set, or pre-split sequence files are substituted.",
        ],
        "outputs": outputs,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data_name": args.data_name, "final": final, "matches_paper_table1": matches}, indent=2), flush=True)
    if not matches:
        print("WARNING: Table 1 statistics differ; inspect manifest.json before claiming an exact dataset reproduction.", file=sys.stderr)


if __name__ == "__main__":
    # The original code enumerates a Python set. Fix its order before Python
    # starts so repeated preprocessing uses identical categorical IDs.
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = dict(os.environ, PYTHONHASHSEED="0")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    main()
