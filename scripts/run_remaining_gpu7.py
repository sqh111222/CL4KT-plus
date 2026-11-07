#!/usr/bin/env python3
"""Run the remaining five CL4KT experiments sequentially on physical GPU 7."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["spanish", "statics", "algebra05", "bridge_algebra06", "sampled_slepemapy"]


def check_gpu7_execution(run, name):
    record = json.loads((run / "execution.json").read_text())
    environment = json.loads((run / "environment.json").read_text())
    expected = dict(physical_gpu=7, cuda_visible_devices="7", cuda_device_order="PCI_BUS_ID",
                    precision="float32", training_batch_size=512)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed {name} is not a matching physical GPU7 execution")
    if environment.get("cuda_visible_devices") != "7" or record.get("exit_status") != "success":
        raise ValueError(f"Completed {name} has inconsistent device/status metadata")
    if record.get("activation_checkpointing", {}).get("enabled") is not True:
        raise ValueError(f"Completed {name} has no checkpointing provenance")
    snapshot = run / "launcher/train_gpu7.py"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != record["launcher_sha256"]:
        raise ValueError(f"Completed {name} has an inconsistent launcher snapshot")
    if hashlib.sha256((ROOT / f"configs/{name}.yaml").read_bytes()).hexdigest() != record["config_sha256"]:
        raise ValueError(f"Completed {name} uses a different configuration file")
    if hashlib.sha256((ROOT / "scripts/train_gpu7.py").read_bytes()).hexdigest() != record["launcher_sha256"]:
        raise ValueError(f"Completed {name} uses a different checkpointing launcher")
    config = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
    data_path = ROOT / config["data_path"]
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != environment["data_sha256"]:
        raise ValueError(f"Completed {name} uses a different dataset")
    for filename, digest in json.loads((run / "source_sha256.json").read_text()).items():
        if (hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest
                or hashlib.sha256((run / "source" / filename).read_bytes()).hexdigest() != digest):
            raise ValueError(f"Completed {name} has different source: {filename}")


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=NAMES, nargs="+", default=NAMES)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/gpu7")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="7", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    state_path = output_root / "suite.json"
    if state_path.exists() and not args.resume:
        parser.error("Existing suite; use --resume or a new output root")
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        created_at=timestamp(), physical_gpu=7, activation_checkpointing=True,
        datasets=args.datasets, runs={}, scope="Five CL4KT student-level five-fold experiments")
    if state["datasets"] != args.datasets:
        parser.error("Resume requires the same ordered dataset list")

    def save_state():
        state["updated_at"] = timestamp()
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(state_path)

    for name in args.datasets:
        run = output_root / f"{name}-fivefold"
        log = output_root / f"{name}-fivefold.log"
        if args.resume and (run / "summary.json").exists():
            summary = json.loads((run / "summary.json").read_text())
            if summary.get("complete"):
                check_gpu7_execution(run, name)
                state["runs"].setdefault(name, {})["status"] = "complete"
                save_state()
                print(f"SKIP complete {name}", flush=True)
                continue
        preflight = [sys.executable, "scripts/preflight_dataset.py", "--config", f"configs/{name}.yaml",
                     "--output", str(output_root / f"preflight-{name}.json")]
        subprocess.run(preflight, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=""), check=True)
        command = [sys.executable, "-u", "scripts/train_gpu7.py", "--config", f"configs/{name}.yaml",
                   "--device", "cuda:0", "--output-dir", str(run)]
        if args.resume and (run / "execution.json").exists():
            command.append("--resume")
        state["runs"][name] = dict(status="running", started_at=timestamp(),
                                  command=command, run_directory=str(run), log=str(log), physical_gpu=7)
        save_state()
        print(f"START {name} on physical GPU7; log={log}", flush=True)
        with log.open("a" if args.resume else "w") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        state["runs"][name].update(return_code=result.returncode, finished_at=timestamp())
        if result.returncode:
            state["runs"][name]["status"] = "failed"
            save_state()
            raise SystemExit(f"Training failed for {name}: code {result.returncode}; inspect {log}")
        summary = json.loads((run / "summary.json").read_text())
        if summary.get("complete") is not True:
            raise RuntimeError(f"Incomplete summary for {name}")
        check_gpu7_execution(run, name)
        state["runs"][name].update(status="complete", metrics=summary["mean"],
                                  std_population=summary["std_population"])
        save_state()
        print(f"COMPLETE {name}: {json.dumps(summary['mean'])}", flush=True)
    state["complete"] = True
    save_state()
    print(f"All requested GPU7 experiments complete: {state_path}", flush=True)


if __name__ == "__main__":
    main()
