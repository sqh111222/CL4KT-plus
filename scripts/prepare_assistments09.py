#!/usr/bin/env python3
"""Prepare ASSISTments09 with the unmodified upstream preprocessing protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocess_data  # noqa: E402


KNOWN_RAW_SHA256 = "1aa296e00b6c88c4d6fad4ca2ae4866484d9fe5484f38f5c8c94dfc49f045e08"
OFFICIAL_PAGE = (
    "https://sites.google.com/site/assistmentsdata/home/"
    "2009-2010-assistment-data/skill-builder-data-2009-2010"
)
MIRROR_URL = "https://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(frame: pd.DataFrame) -> dict:
    """Atomic skill statistics before upstream replaces skills with Q-row IDs."""
    result = {
        "rows": int(len(frame)),
        "students": int(frame["user_id"].nunique()),
        "questions": int(frame["problem_id"].nunique()),
        "atomic_skills": int(frame["skill_id"].nunique()),
        "unique_order_ids": int(frame["order_id"].nunique()),
        "duplicate_order_rows": int(frame["order_id"].duplicated().sum()),
    }
    return result


def install_raw(source: Path, destination: Path, member: str | None) -> dict:
    """Read exactly one CSV without extracting arbitrary archive paths."""
    if not source.is_file():
        raise FileNotFoundError(f"Raw file does not exist: {source}")
    source_hash = sha256(source)
    is_zip = zipfile.is_zipfile(source)
    if member is not None and not is_zip:
        raise ValueError("--zip-member requires a ZIP archive")
    chosen = None
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".csv", delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            if is_zip:
                with zipfile.ZipFile(source) as archive:
                    candidates = [entry for entry in archive.namelist() if entry.lower().endswith(".csv")]
                    if member is not None:
                        if member not in candidates:
                            raise ValueError(f"ZIP CSV member not found: {member}; available: {candidates}")
                        chosen = member
                    elif len(candidates) == 1:
                        chosen = candidates[0]
                    else:
                        raise ValueError(f"ZIP has {len(candidates)} CSV members; select one with --zip-member: {candidates}")
                    with archive.open(chosen) as raw_stream:
                        shutil.copyfileobj(raw_stream, tmp)
            else:
                with source.open("rb") as raw_stream:
                    shutil.copyfileobj(raw_stream, tmp)
            tmp.flush()
            csv_hash = sha256(temporary)
            if destination.exists():
                if sha256(destination) != csv_hash:
                    raise FileExistsError(
                        f"Existing {destination} has different content; use a different --output-dir."
                    )
            else:
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "input_path": str(source),
        "input_type": "zip" if is_zip else "csv",
        "input_sha256": source_hash,
        "zip_member": chosen,
        "raw_csv_path": str(destination),
        "raw_csv_sha256": csv_hash,
        "matches_verified_corrected_csv": csv_hash == KNOWN_RAW_SHA256,
        "dataset_homepage": OFFICIAL_PAGE,
        "public_corrected_zip_mirror": MIRROR_URL,
        "provenance_note": "Input path and hashes identify the file actually used; the mirror URL is a retrieval option, not a claim this local file was downloaded in this run.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Original corrected.csv or ZIP containing it")
    parser.add_argument("--zip-member", help="Exact CSV member name when a ZIP contains multiple CSV files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset" / "assistments09")
    parser.add_argument("--min-user-inter-num", type=int, default=5)
    args = parser.parse_args()
    if args.min_user_inter_num < 1:
        parser.error("--min-user-inter-num must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.name != "assistments09":
        parser.error("--output-dir must end with /assistments09 to preserve the upstream dataset-specific branch")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = install_raw(args.raw.expanduser().resolve(), output_dir / "data.csv", args.zip_member)

    raw = pd.read_csv(
        output_dir / "data.csv", encoding="ISO-8859-1", low_memory=False,
        usecols=["order_id", "user_id", "problem_id", "skill_id", "skill_name", "correct"],
    )
    stages = {"raw": stats(raw)}
    binary = raw[raw["correct"].isin([0, 1])]
    stages["binary_correctness"] = stats(binary)
    named = binary[binary["skill_id"].notna()]
    stages["remove_missing_skills"] = stats(named)
    min_length = named.groupby("user_id").filter(lambda frame: len(frame) >= args.min_user_inter_num)
    stages["minimum_length_before_deduplication"] = stats(min_length)
    deduplicated = min_length.drop_duplicates("order_id")
    stages["deduplicate_order_id"] = stats(deduplicated)

    # Preserve the upstream filtering, Q-matrix merging, sorting and ID rules.
    preprocess_data.BASE_PATH = str(output_dir.parent)
    preprocess_data.prepare_assistments(
        data_name=output_dir.name,
        min_user_inter_num=args.min_user_inter_num,
        remove_nan_skills=True,
    )
    processed = pd.read_csv(output_dir / "preprocessed_df.csv", sep="\t")
    lengths = processed.groupby("user_id").size()
    final = {
        "rows": int(len(processed)),
        "students": int(processed["user_id"].nunique()),
        "questions": int(processed["item_id"].nunique()),
        "unique_skill_combinations": int(processed["skill_id"].nunique()),
        "minimum_interactions_per_student": int(lengths.min()),
        "students_below_requested_minimum_after_deduplication": int((lengths < args.min_user_inter_num).sum()),
    }
    expected = {"students": 3695, "questions": 17728, "rows": 282071, "atomic_skills_after_order_deduplication": 112, "unique_skill_combinations": 149}
    actual = {
        **{key: final[key] for key in ["students", "questions", "rows", "unique_skill_combinations"]},
        "atomic_skills_after_order_deduplication": stages["deduplicate_order_id"]["atomic_skills"],
    }
    outputs = {}
    for name in ["preprocessed_df.csv", "original_df.csv", "q_mat.npz", "question_skill_rel.pkl", "skill_id_name"]:
        path = output_dir / name
        if path.is_file():
            outputs[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "dataset": "assistments09",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": "f08aa4b5adb8eabfa4306b64e43bf3ac92ff06a2",
        "preprocessing_module_sha256": sha256(Path(preprocess_data.__file__)),
        "source": provenance,
        "protocol": {"min_user_inter_num": args.min_user_inter_num, "remove_nan_skills": True, "minimum_filter_applied_before_order_id_deduplication": True, "skill_representation": "unique question-to-atomic-skill Q-matrix rows"},
        "stages": stages,
        "final": final,
        "reference_statistics": expected,
        "matches_reference_statistics": actual == expected,
        "interpretation": [
            "Paper Table 1 reports 112 skills; upstream prints this after keeping the first row of each order_id and before replacing skills by unique Q-matrix rows.",
            "The full question-to-skill matrix is built before order_id deduplication, so the actual model input contains 149 unique skill combinations for the verified corrected.csv.",
            "The minimum-length filter precedes deduplication in upstream. The verified file therefore retains 51 students with fewer than 5 distinct interactions (minimum 2). This run preserves that order.",
            "The later collapsed one-row-per-student-problem release differs from the expanded corrected.csv used here and should not be substituted silently.",
        ],
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "final": final, "matches_reference_statistics": actual == expected}, ensure_ascii=False, indent=2))
    if actual != expected:
        print("WARNING: statistics differ from the verified reference; inspect manifest.json before comparing with the paper.", file=sys.stderr)


if __name__ == "__main__":
    main()
