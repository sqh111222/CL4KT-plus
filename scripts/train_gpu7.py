#!/usr/bin/env python3
"""Run the unchanged CL4KT trainer on physical GPU 7 with activation checkpointing.

All arguments are forwarded to main.py. An explicit --output-dir is required;
fresh runs require an empty directory, and --resume requires matching GPU7
execution metadata. This launcher preserves FP32 and a training batch of 512.
"""

import argparse
from datetime import datetime
from functools import wraps
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTING = {
    "enabled": True,
    "target": "models.modules.CL4KTTransformerLayer.forward",
    "use_reentrant": False,
    "preserve_rng_state": True,
    "condition": "module.training and torch.is_grad_enabled()",
}


def install_activation_checkpointing():
    """Install the runtime wrapper and return a function restoring the original.

    Imports are deliberately deferred: the CLI must bind the physical GPU before
    importing torch. Tests can invoke this helper directly in a CPU-only process.
    """
    import torch
    import torch.utils.checkpoint as checkpointing
    from models.modules import CL4KTTransformerLayer

    original = CL4KTTransformerLayer.forward
    if getattr(original, "_gpu7_activation_checkpointing", False):
        raise RuntimeError("Activation checkpointing is already installed")

    @wraps(original)
    def forward(self, mask, query, key, values, apply_pos=True):
        if self.training and torch.is_grad_enabled():
            # checkpoint infers CUDA devices from positional tensor arguments.
            # Models call this method with keyword tensors; passing them through
            # as kwargs would miss CUDA RNG preservation during recomputation.
            return checkpointing.checkpoint(
                original, self, mask, query, key, values, apply_pos,
                use_reentrant=False, preserve_rng_state=True)
        return original(self, mask, query, key, values, apply_pos)

    forward._gpu7_activation_checkpointing = True
    CL4KTTransformerLayer.forward = forward

    def restore():
        if CL4KTTransformerLayer.forward is forward:
            CL4KTTransformerLayer.forward = original
        elif CL4KTTransformerLayer.forward is not original:
            raise RuntimeError("The transformer forward changed after checkpointing was installed")

    return restore


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_launcher_arguments(arguments):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/assistments09.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    known, _ = parser.parse_known_args(arguments)
    if known.device not in ("auto", "cuda", "cuda:0"):
        parser.error("The GPU7 launcher requires --device auto, cuda, or cuda:0")
    output = known.output_dir.expanduser().resolve()
    if known.resume:
        if not (output / "execution.json").is_file():
            parser.error("--resume requires an existing GPU7 run with execution.json")
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("A fresh GPU7 run requires an empty --output-dir; existing results are preserved")
    return known, output


def physical_gpu_details():
    """Read driver identifiers without creating an additional CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=7", "--query-gpu=index,uuid,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
        index, uuid, pci_bus_id = [part.strip() for part in result.stdout.strip().split(",")]
        return dict(index=int(index), uuid=uuid, pci_bus_id=pci_bus_id)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(query_available=False, error=str(error))


def write_execution(path, metadata):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def record_execution(output, args, previous_visibility, previous_device_order, torch):
    import yaml

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config["train"]["batch_size"] != 512:
        raise ValueError("The GPU7 launcher preserves training batch_size=512; use an unchanged paper configuration")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("The GPU7 launcher requires the unchanged FP32 default dtype")
    launcher_digest = sha256(__file__)
    snapshot = output / "launcher" / "train_gpu7.py"
    required = dict(
        physical_gpu=7, cuda_visible_devices="7", cuda_device_order="PCI_BUS_ID", precision="float32",
        training_batch_size=512, activation_checkpointing=CHECKPOINTING,
        launcher_sha256=launcher_digest, config_sha256=sha256(config_path),
        torch=str(torch.__version__), cuda=torch.version.cuda,
    )
    metadata_path = output / "execution.json"
    if args.resume:
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in required.items()):
            raise ValueError("Existing execution metadata differs from this GPU7 launcher/configuration/environment; use a new directory")
        if not snapshot.is_file() or sha256(snapshot) != launcher_digest:
            raise ValueError("Existing GPU7 launcher snapshot is missing or differs; preserve the run and use a new directory")
        print(f"Verified matching GPU7 execution metadata: {metadata_path}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    metadata = dict(
        schema_version=1, **required, launcher_snapshot="launcher/train_gpu7.py",
        previous_cuda_visible_devices=previous_visibility,
        previous_cuda_device_order=previous_device_order,
        logical_device="cuda:0", gpu_name=torch.cuda.get_device_name(0),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        physical_gpu_details=physical_gpu_details(),
        autocast_enabled=False, python=sys.version, executable=sys.executable,
        platform=platform.platform(), config_path=str(config_path),
        seed=config["seed"], optimizer="Adam (unchanged main.py)",
        checkpointing_implementation="runtime wrapper; core source files unchanged",
        command=sys.argv,
        recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    )
    write_execution(metadata_path, metadata)
    print(f"Recorded GPU7 execution metadata: {metadata_path}", flush=True)


def main():
    if "torch" in sys.modules:
        raise RuntimeError("Invoke this launcher as a fresh Python process so GPU7 is bound before importing torch")
    previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    previous_device_order = os.environ.get("CUDA_DEVICE_ORDER")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    print(f"Physical GPU binding: CUDA_DEVICE_ORDER={previous_device_order!r} -> 'PCI_BUS_ID'; "
          f"CUDA_VISIBLE_DEVICES={previous_visibility!r} -> '7'; logical device cuda:0", flush=True)
    arguments = sys.argv[1:]
    if not any(argument in ("-h", "--help") for argument in arguments):
        args, output = parse_launcher_arguments(arguments)
    else:
        args, output = None, None
    sys.path.insert(0, str(ROOT))
    trainer = importlib.import_module("main")
    import torch
    if args is None:
        trainer.main()  # Preserve the unchanged trainer's --help behavior.
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Physical GPU7 must be available as the sole visible CUDA device; CPU fallback is disabled")
    record_execution(output, args, previous_visibility, previous_device_order, torch)
    torch.cuda.reset_peak_memory_stats(0)
    restore = install_activation_checkpointing()
    exit_status, failure = "failed", None
    try:
        # The trainer keeps its own argument parsing, optimizer, seeds, batches,
        # model source snapshots, validation selection, and test evaluation.
        trainer.main()
        exit_status = "success"
    except BaseException as error:
        exit_status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        restore()
        try:
            path = output / "execution.json"
            metadata = json.loads(path.read_text())
            metadata.update(
                exit_status=exit_status, failure=failure,
                finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                peak_cuda_memory_allocated=int(torch.cuda.max_memory_allocated(0)),
                peak_cuda_memory_reserved=int(torch.cuda.max_memory_reserved(0)),
            )
            write_execution(path, metadata)
        except Exception as reporting_error:
            if failure is None:
                raise
            # Preserve the training exception if writing diagnostics also fails.
            print(f"Could not finalize execution diagnostics: {reporting_error}", file=sys.stderr)


if __name__ == "__main__":
    main()
