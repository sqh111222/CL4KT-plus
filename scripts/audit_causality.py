#!/usr/bin/env python3
"""Reproduce upstream's future-response leak and the minimal attention fix.

This CPU-only audit copies the pinned upstream model to a temporary directory.
The same initialized model is evaluated before and after zeroing disallowed
attention probabilities. No project source, upstream source, or model weight is
modified. Optional --output saves the measurements as a JSON artifact.
"""

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

# Set before importing torch so the audit never initializes a CUDA context.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit():
    upstream = ROOT / "upstream"
    source = upstream / "models" / "modules.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official source is missing: {source}")
    source_hashes = {str(path.relative_to(upstream)): file_digest(path)
                     for path in (source, upstream / "models" / "cl4kt.py")}
    torch.set_num_threads(2)
    torch.manual_seed(1)
    torch.use_deterministic_algorithms(True)
    config = dict(hidden_size=64, num_blocks=2, num_attn_heads=8,
                  kq_same=True, final_fc_dim=512, d_ff=1024, l2=0.0,
                  dropout=0.0, reg_cl=0.1, negative_prob=1.0,
                  hard_negative_weight=1.0, temp=0.05)
    package_name = "_cl4kt_upstream_causality_audit"
    try:
        with TemporaryDirectory(prefix="cl4kt-causality-") as temporary:
            models = Path(temporary) / "models"
            shutil.copytree(upstream / "models", models,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            module_path = models / "modules.py"
            # Device object selection is the only compatibility change needed
            # to evaluate this GPU-oriented upstream implementation on CPU.
            cpu_source = source.read_text().replace(".get_device()", ".device")
            module_path.write_text(cpu_source)
            spec = importlib.util.spec_from_file_location(
                package_name, models / "__init__.py",
                submodule_search_locations=[str(models)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
            model_module = importlib.import_module(f"{package_name}.cl4kt")
            attention_module = importlib.import_module(f"{package_name}.modules")
            model = model_module.CL4KT(20, 30, 10, **config).cpu().eval()
            skills = torch.tensor([
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            ], device="cpu")
            responses = torch.tensor([
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                [1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            ], device="cpu")
            changed = responses.clone()
            changed[:, 5:] = 1 - changed[:, 5:]
            attention_mask = torch.ones_like(skills)

            @torch.no_grad()
            def differences():
                before = model(dict(skills=skills, responses=responses,
                                    attention_mask=attention_mask))["pred"]
                after = model(dict(skills=skills, responses=changed,
                                   attention_mask=attention_mask))["pred"]
                # Output column zero scores original position one. Predictions
                # at positions 1..5 cannot depend on answers at positions >=5.
                delta = (after - before)[:, :5]
                return dict(differences=delta.tolist(),
                            max_abs_diff=float(delta.abs().max()),
                            mean_abs_diff=float(delta.abs().mean()))

            initialized_state = state_digest(model)
            original = differences()
            old = "    scores = F.softmax(scores, dim=-1)\n\n    attn_scores = scores"
            new = ("    scores = F.softmax(scores, dim=-1)\n"
                   "    scores = scores * mask.to(scores.dtype)\n\n"
                   "    attn_scores = scores")
            if cpu_source.count(old) != 1:
                raise AssertionError("Expected upstream individual_attention softmax was not found exactly once")
            # Replacing this function in its module globals changes attention
            # for the existing model; initialization and all weights are shared.
            exec(compile(cpu_source.replace(old, new), str(module_path), "exec"),
                 attention_module.__dict__)
            fixed = differences()
            assert state_digest(model) == initialized_state, "Audit unexpectedly changed model weights"
            assert original["max_abs_diff"] > 1e-6, "Expected upstream causality violation was not reproduced"
            assert fixed["max_abs_diff"] == 0.0, "Minimal attention fix did not restore causality"
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
    assert all(file_digest(upstream / name) == digest for name, digest in source_hashes.items()), \
        "Upstream source changed during the audit"
    commit = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return dict(
        passed=True, torch=str(torch.__version__), device="cpu", seed=1,
        upstream_commit=commit, upstream_source_sha256=source_hashes,
        model_state_sha256=initialized_state, model_config=config,
        change="Flip responses[:, 5:] while keeping all questions and earlier answers fixed",
        checked_targets="Original sequence positions 1 through 5 inclusive, zero-based",
        compatibility_change="get_device() -> device in the temporary copy only",
        minimal_fix="Multiply individual_attention's final softmax probabilities by its allowed-position mask",
        original=original, fixed=fixed, model_weights_unchanged=True,
        upstream_sources_unchanged=True,
        recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence artifact")
    args = parser.parse_args()
    result = audit()
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
