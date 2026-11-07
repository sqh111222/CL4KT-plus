#!/usr/bin/env python3
"""Prepare author-published Spanish and complete AKT Statics student sequences."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocess_data  # noqa: E402

UPSTREAM_COMMIT = "f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2"
SOURCES = {
    "spanish": [
        {
            "filename": "spanish_dataset.txt",
            "repository": "robert-lindsey/WCRP",
            "commit": "7e46c02d430b5fdc7a13dda42cc965b87f640346",
            "repository_path": "datasets/spanish_dataset.txt",
            "sha256": "47aec7f0d52d11e87d9e8e1415a7d41e02e2219075fb3b0e4d710bc011af1769",
        },
        {
            "filename": "spanish_expert_labels.txt",
            "repository": "robert-lindsey/WCRP",
            "commit": "7e46c02d430b5fdc7a13dda42cc965b87f640346",
            "repository_path": "datasets/spanish_expert_labels.txt",
            "sha256": "85aaf66ad12bd315a81ce96a4d824d0bfcf7dd943d1be6bffd3fda070cb60d35",
        },
    ],
    "statics": [
        {
            "filename": "statics_sequences.csv",
            "repository": "arghosh/AKT",
            "commit": "01b7e8d7d7c862cabc9467108e3e7f2555653a88",
            "repository_path": "data/statics/statics.csv",
            "sha256": "cd8b6ed640bb2e3dfd63ec681e17ec2262fb7284cea8735d1cfc2f317d17bab1",
        }
    ],
}
TARGETS = {
    "spanish": dict(rows=578726, students=182, questions=409, skills=221),
    "statics": dict(rows=189297, students=333, questions=1223, skills=1223),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fetch_sources(name: str, directory: Path, download: bool) -> list[dict]:
    records = []
    for source in SOURCES[name]:
        record = dict(source)
        record["url"] = (
            f"https://raw.githubusercontent.com/{source['repository']}/"
            f"{source['commit']}/{source['repository_path']}"
        )
        path = directory / source["filename"]
        downloaded = False
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; supply the pinned file or use --download")
            with urllib.request.urlopen(record["url"], timeout=90) as response:
                body = response.read()
            if hashlib.sha256(body).hexdigest() != source["sha256"]:
                raise ValueError(f"Unexpected downloaded content: {record['url']}")
            path.write_bytes(body)
            downloaded = True
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Raw source hash differs from pinned author version: {path}")
        record.update(local_path=str(path.resolve()), size_bytes=path.stat().st_size,
                      downloaded_by_this_invocation=downloaded)
        records.append(record)
    return records


def stats(frame: pd.DataFrame) -> dict:
    lengths = frame.groupby("user_id", sort=False).size()
    return dict(rows=int(len(frame)), students=int(frame.user_id.nunique()),
                questions=int(frame.item_id.nunique()),
                skills=int(frame.skill_id.nunique()) if "skill_id" in frame else None,
                shortest_student_history=int(lengths.min()),
                longest_student_history=int(lengths.max()),
                students_with_fewer_than_five_interactions=int((lengths < 5).sum()))


def read_statics_sequences(path: Path) -> pd.DataFrame:
    """Each three-line block is one complete student, not a training window."""
    lines = path.read_text().splitlines()
    if len(lines) % 3:
        raise ValueError("Statics file must contain complete three-line student blocks")
    records = []
    for student, offset in enumerate(range(0, len(lines), 3)):
        length = int(lines[offset])
        questions = [int(value) for value in lines[offset + 1].split(",") if value]
        responses = [int(value) for value in lines[offset + 2].split(",") if value]
        if length <= 0 or len(questions) != length or len(responses) != length:
            raise ValueError(f"Invalid student block {student}: recorded and actual lengths differ")
        if not set(responses) <= {0, 1} or min(questions) < 1:
            raise ValueError(f"Invalid question or response in student block {student}")
        records.extend((student, token, 0, response, token)
                       for token, response in zip(questions, responses))
    return pd.DataFrame(records, columns=["user_id", "item_id", "timestamp", "correct", "skill_id"])


def prepare(name: str, data_root: Path, download: bool) -> None:
    directory = data_root / name
    directory.mkdir(parents=True, exist_ok=True)
    sources = fetch_sources(name, directory, download)
    stages = []
    preprocess_data.BASE_PATH = str(data_root)
    if name == "spanish":
        data = np.loadtxt(directory / "spanish_dataset.txt", dtype=np.int64)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Spanish source must have exactly student/item/response columns")
        raw = pd.DataFrame(data, columns=["user_id", "item_id", "correct"])
        if not set(raw.correct.unique()) <= {0, 1} or raw.item_id.min() < 0:
            raise ValueError("Invalid Spanish item IDs or responses")
        stages.append(dict(stage="raw_student_responses", **stats(raw)))
        labels = np.loadtxt(directory / "spanish_expert_labels.txt", dtype=np.int64)
        if raw.item_id.max() >= len(labels):
            raise ValueError("Spanish item ID has no corresponding expert label")
        raw["skill_id"] = labels[raw.item_id.to_numpy()]
        stages.append(dict(stage="attach_author_expert_labels", **stats(raw)))
        preprocess_data.prepare_spanish()
        limitations = [
            "Original file order defines temporal order; source has no real timestamps.",
            "Official preprocessing uses zero timestamps and preserves within-student input order.",
            "Author repository licenses code under MIT; no separate dataset license is stated in README.",
        ]
        version = "WCRP author-published raw student responses and expert labels"
    else:
        raw = read_statics_sequences(directory / "statics_sequences.csv")
        stages.append(dict(stage="complete_AKT_student_sequences", **stats(raw)))
        # CL4KT's official function expects a one-based table and subtracts one.
        # The published Statics sequence token acts as both item and skill here;
        # no problem-to-KC metadata or original anonymous IDs can be recovered.
        target = directory / "interaction_df.csv"
        table = raw.to_csv(sep="\t", index=False).encode()
        if target.exists() and target.read_bytes() != table:
            raise FileExistsError(f"Refusing to replace a different Statics table: {target}")
        target.write_bytes(table)
        stages.append(dict(stage="adapt_to_CL4KT_one_based_interaction_table", **stats(raw)))
        preprocess_data.prepare_statics()
        limitations = [
            "This is the complete public AKT sequence version, not a DataShop raw export.",
            "Student ID is the zero-based sequence-block index; original anonymous IDs are unavailable.",
            "Within-student interaction order and boundaries are preserved exactly; original timestamps are unavailable.",
            "Timestamp is zero and is a placeholder, not an observed event time.",
            "item_id and skill_id both reference the published exercise token; a separate original question ID is unavailable.",
            "Matching paper counts establishes statistical compatibility, not proof of identical CL4KT author data or folds.",
        ]
        version = "AKT author-published complete 333-student Statics sequences"
    output = directory / "preprocessed_df.csv"
    frame = pd.read_csv(output, sep="\t")
    if frame.isna().any().any() or not set(frame.correct.unique()) <= {0, 1}:
        raise ValueError("Preprocessed table contains invalid responses or missing values")
    for column in ("user_id", "item_id", "skill_id"):
        values = np.sort(frame[column].unique())
        if not np.array_equal(values, np.arange(len(values))):
            raise ValueError(f"Expected contiguous zero-based {column}")
    if name == "spanish":
        # Official grouping must not change any student's history.
        expected = pd.concat([part for _, part in raw.groupby("user_id")])
        columns = ["user_id", "item_id", "correct", "skill_id"]
        pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True),
                                      expected[columns].reset_index(drop=True), check_dtype=False)
    else:
        expected = raw.copy()
        expected[["item_id", "skill_id"]] -= 1
        pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    actual = stats(frame)
    target_stats = TARGETS[name]
    if any(actual[key] != value for key, value in target_stats.items()):
        raise ValueError(f"Published-source statistics changed: {actual} vs {target_stats}")
    stages.append(dict(stage="official_CL4KT_output", **actual))
    matrix = sparse.load_npz(directory / "q_mat.npz")
    if matrix.shape != (actual["questions"], actual["skills"]):
        raise ValueError("Question-skill matrix shape does not match output IDs")
    files = [p for p in directory.iterdir()
             if p.is_file() and p.name in {"preprocessed_df.csv", "q_mat.npz", "question_skill_rel.pkl", "interaction_df.csv"}]
    manifest = dict(
        dataset=name, version=version,
        recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        source_files=sources, stages=stages,
        paper_table1={**target_stats, "questions": None if name == "statics" else target_stats["questions"]},
        published_source_target=target_stats,
        statistics_match_paper=True, limitations=limitations,
        preprocessing=dict(upstream_commit=UPSTREAM_COMMIT,
                           upstream_function=f"prepare_{name}",
                           source_path="preprocess_data.py",
                           source_sha256=sha256(ROOT / "preprocess_data.py"),
                           adapter_path="scripts/prepare_spanish_statics.py",
                           adapter_sha256=sha256(Path(__file__))),
        output_files={p.name: dict(sha256=sha256(p), size_bytes=p.stat().st_size) for p in files},
        sequence_policy="Official training loader retains the last 100 interactions per student; no truncation is applied here.",
    )
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(dict(dataset=name, output=str(output), stats=actual,
                          sha256=sha256(output), manifest=str(directory / "manifest.json")), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "spanish", "statics"], default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--download", action="store_true", help="Download pinned public source files when absent")
    args = parser.parse_args()
    for name in (["spanish", "statics"] if args.dataset == "all" else [args.dataset]):
        prepare(name, args.data_root.resolve(), args.download)


if __name__ == "__main__":
    main()
