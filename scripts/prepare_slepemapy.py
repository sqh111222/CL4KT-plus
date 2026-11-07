#!/usr/bin/env python3
"""Prepare the upstream 5,000-student Slepemapy protocol with a recorded seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://www.fi.muni.cz/adaptivelearning/data/slepemapy/2015-05-21.zip"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12405)
    parser.add_argument("--students", type=int, default=5000)
    parser.add_argument("--min-user-inter-num", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/sampled_slepemapy")
    args = parser.parse_args()
    if args.students < 5 or args.min_user_inter_num < 1:
        parser.error("Need at least five students and a positive minimum length")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    source = args.raw.resolve()
    raw_hash = sha256(source)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous["raw_sha256"] != raw_hash or previous["sampling_seed"] != args.seed
                or previous["sample_students"] != args.students
                or previous["min_user_inter_num"] != args.min_user_inter_num):
            raise FileExistsError("Existing output has a different input/protocol; use a new output directory")
        if all(sha256(output / name) == record["sha256"] for name, record in previous["outputs"].items()):
            print("Verified existing Slepemapy artifacts:", output)
            return
        raise FileExistsError("Existing artifacts differ from their manifest; preserve them and use a new output directory")
    for name in ("preprocessed_df.csv", "q_mat.npz", "student_mapping.csv"):
        if (output / name).exists():
            raise FileExistsError(f"Untracked existing artifact: {output / name}")
    raw_copy = output / "answer.csv"
    if raw_copy.exists():
        if sha256(raw_copy) != raw_hash:
            raise FileExistsError(f"Different raw input already exists: {raw_copy}")
    elif source != raw_copy:
        shutil.copyfile(source, raw_copy)
    columns = ["id", "user", "place_asked", "place_answered", "type", "inserted", "response_time"]
    frame = pd.read_csv(raw_copy, sep=";", usecols=columns)
    raw_rows = len(frame)
    raw_students = frame.user.nunique()
    frame = frame.loc[frame.place_answered.notna()].copy()
    nonmissing_rows = len(frame)
    counts = frame.groupby("user", sort=True).size()
    eligible = counts.index[counts >= args.min_user_inter_num].to_numpy().copy()
    if len(eligible) < args.students:
        raise ValueError(f"Only {len(eligible)} eligible students, requested {args.students}")
    # Upstream shuffles a list of user groups in sorted groupby order. Shuffling
    # these IDs uses the same RandomState permutation without materializing all
    # ~90,000 DataFrames or parsing timestamps for unselected users.
    np.random.RandomState(args.seed).shuffle(eligible)
    selected = eligible[:args.students]
    chosen = frame.loc[frame.user.isin(selected)].copy()
    del frame
    chosen["timestamp"] = pd.to_datetime(chosen.inserted, format="%Y-%m-%d %H:%M:%S").astype("int64") // 10**9
    chosen["correct"] = (chosen.place_asked.astype(float) == chosen.place_answered.astype(float)).astype(float)
    chosen["dwell_time"] = chosen.response_time / 1000.0
    chosen["skill_id"] = chosen.place_asked - 1
    chosen["item_id"] = chosen.skill_id * 2 + chosen["type"] - 1
    groups = chosen.groupby("user", sort=True)
    pieces = []
    mapping = []
    for user_id, original_user_id in enumerate(selected, start=1):
        group = groups.get_group(original_user_id).sort_values("timestamp").copy()
        group["user_id"] = user_id
        pieces.append(group)
        mapping.append(dict(user_id=user_id, original_user_id=int(original_user_id), interactions=len(group)))
    processed = pd.concat(pieces, ignore_index=True)
    processed["item_id"] = np.unique(processed.item_id, return_inverse=True)[1]
    processed["skill_id"] = np.unique(processed.skill_id, return_inverse=True)[1]
    processed["timestamp"] = processed.timestamp.astype(float)
    pd.DataFrame(mapping).to_csv(output / "student_mapping.csv", index=False)
    processed[["user_id", "id"]].rename(columns={"id": "original_answer_id"}).to_csv(output / "answer_mapping.csv", index=False)
    pairs = processed[["item_id", "skill_id"]].drop_duplicates()
    q_mat = sparse.coo_matrix((np.ones(len(pairs)), (pairs.item_id, pairs.skill_id)),
                              shape=(processed.item_id.nunique(), processed.skill_id.nunique())).tocsr()
    sparse.save_npz(output / "q_mat.npz", q_mat)
    processed = processed[["user_id", "skill_id", "item_id", "dwell_time", "timestamp", "correct"]]
    processed.to_csv(output / "preprocessed_df.csv", sep="\t", index=False)
    observed = dict(students=int(processed.user_id.nunique()), questions=int(processed.item_id.nunique()),
                    skills=int(processed.skill_id.nunique()), interactions=len(processed))
    expected = dict(students=5000, questions=2723, skills=1391, interactions=625523)
    names = ["preprocessed_df.csv", "q_mat.npz", "student_mapping.csv", "answer_mapping.csv"]
    manifest = dict(dataset="sampled_slepemapy", created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    source_path=str(source), raw_copy=str(raw_copy), source_url=DATA_URL,
                    raw_sha256=raw_hash, raw_rows=raw_rows, raw_students=int(raw_students),
                    rows_after_removing_missing_answers=nonmissing_rows, eligible_students=len(eligible),
                    sampling_seed=args.seed, sample_students=args.students,
                    min_user_inter_num=args.min_user_inter_num, final=observed, paper_statistics=expected,
                    exact_paper_statistics=observed == expected,
                    upstream_commit="f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2",
                    preparation_script_sha256=sha256(__file__),
                    protocol_notes=["Same sorted-user shuffle and first-5000 selection as upstream, now with a recorded seed.",
                                    "The paper's student IDs and sampling seed are unpublished; this is a new reproducible sample, not its exact sample.",
                                    "Timestamp strings interpreted as UTC rather than host-local mktime; a common offset does not change sequence order.",
                                    "Per-user timestamp quicksort, question/skill definitions, ID reindexing and missing-answer filtering follow upstream.",
                                    "No repeated sampling to force the paper's counts. Training uses the fixed sample for all five folds."],
                    license="ODbL 1.0; database contents under DbCL 1.0, see raw dataset README",
                    outputs={name:dict(sha256=sha256(output / name), bytes=(output / name).stat().st_size) for name in names})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), statistics=observed, paper_statistics=expected,
                          exact_match=observed == expected), indent=2), flush=True)


if __name__ == "__main__":
    main()
