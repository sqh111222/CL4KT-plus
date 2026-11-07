"""Checks for historical snapshot integrity and dataset-specific plot references."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.plot_results import paper_targets
from scripts.verify_run import SOURCE_FILES, VerificationError, sha256, verify_execution, verify_sources


class ArtifactTests(unittest.TestCase):
    def make_execution(self, root):
        run, current = root / "run", root / "current"
        snapshot = run / "launcher" / "train_gpu7.py"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text("# frozen runtime launcher unit-test fixture\n")
        launcher = current / "scripts" / "train_gpu7.py"
        launcher.parent.mkdir(parents=True)
        launcher.write_bytes(snapshot.read_bytes())
        environment = dict(command=["scripts/train_gpu7.py"], cuda_visible_devices="7", device="cuda:0",
                           torch="test-torch-version", cuda="test-cuda-version", gpu="unit-test GPU")
        config = dict(seed=12405, train=dict(batch_size=512))
        execution = dict(
            schema_version=1, launcher_snapshot="launcher/train_gpu7.py", launcher_sha256=sha256(snapshot),
            physical_gpu=7, cuda_visible_devices="7", cuda_device_order="PCI_BUS_ID", logical_device="cuda:0",
            precision="float32", autocast_enabled=False, training_batch_size=512,
            activation_checkpointing=dict(enabled=True, use_reentrant=False, preserve_rng_state=True,
                                          target="models.modules.CL4KTTransformerLayer.forward",
                                          condition="module.training and torch.is_grad_enabled()"),
            exit_status="success", failure=None, recorded_at="2026-11-05T10:00:00+08:00",
            finished_at="2026-11-05T10:01:00+08:00", torch=environment["torch"], cuda=environment["cuda"],
            gpu_name=environment["gpu"], seed=config["seed"],
            physical_gpu_details=dict(index=7, uuid="GPU-00000000-0000-0000-0000-000000000007",
                                      pci_bus_id="00000000:20:00.0"),
            gpu_uuid="00000000-0000-0000-0000-000000000007",
            peak_cuda_memory_allocated=1024, peak_cuda_memory_reserved=2048,
        )
        (run / "execution.json").write_text(json.dumps(execution))
        return run, current, config, environment, execution

    def make_sources(self, root):
        run, current = root / "run", root / "current"
        recorded = {}
        for name in SOURCE_FILES:
            snapshot = run / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text("# frozen unit-test source\n")
            workspace = current / name
            workspace.parent.mkdir(parents=True, exist_ok=True)
            workspace.write_bytes(snapshot.read_bytes())
            recorded[name] = sha256(snapshot)
        return run, current, recorded

    def test_current_source_drift_is_reported_without_invalidating_snapshot(self):
        with TemporaryDirectory() as temporary:
            run, current, recorded = self.make_sources(Path(temporary))
            (current / "main.py").write_text("# later unit-test source\n")
            output = io.StringIO()
            with redirect_stdout(output):
                differences = verify_sources(run, recorded, current)
            self.assertEqual(differences, ["main.py"])
            self.assertIn("hash-verified frozen snapshots", output.getvalue())
            with self.assertRaisesRegex(VerificationError, "Current source differs"):
                verify_sources(run, recorded, current, strict_current_source=True)

    def test_frozen_snapshot_corruption_is_always_rejected(self):
        with TemporaryDirectory() as temporary:
            run, current, recorded = self.make_sources(Path(temporary))
            (run / "source" / "models" / "cl4kt.py").write_text("# altered snapshot\n")
            with self.assertRaisesRegex(VerificationError, "Frozen source hash differs"):
                verify_sources(run, recorded, current)

    def test_table_two_dataset_aliases_and_missing_summary_reference(self):
        self.assertEqual(paper_targets("bridge_algebra06", None),
                         {"auc": 0.7733, "rmse": 0.3791})
        self.assertEqual(paper_targets("sampled_slepemapy", None),
                         {"auc": 0.7218, "rmse": 0.3926})
        self.assertEqual(paper_targets("assist09"), paper_targets("assistments09"))
        for name in ("algebra05", "algebra06", "spanish", "statics", "slepemapy", "assistments09"):
            self.assertEqual(set(paper_targets(name)), {"auc", "rmse"})

    def test_wrong_or_unknown_paper_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees with Table 2"):
            paper_targets("spanish", {"auc": 0.1, "rmse": 0.4049})
        with self.assertRaisesRegex(ValueError, "No paper Table 2 target"):
            paper_targets("unknown_dataset")

    def test_valid_gpu7_runtime_metadata_and_historical_compatibility(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, current, config, environment, expected = self.make_execution(root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(verify_execution(run, config, environment, current), expected)
            historical = root / "historical"
            historical.mkdir()
            self.assertIsNone(verify_execution(historical, {}, dict(command=["main.py"]), current))
            with self.assertRaisesRegex(VerificationError, "missing execution.json"):
                verify_execution(historical, config, environment, current)

    def test_invalid_gpu7_runtime_constraints_are_rejected(self):
        with TemporaryDirectory() as temporary:
            run, current, config, environment, expected = self.make_execution(Path(temporary))
            cases = [
                ("physical GPU", dict(physical_gpu=0)),
                ("CUDA visibility", dict(cuda_visible_devices="0")),
                ("device order", dict(cuda_device_order="FASTEST_FIRST")),
                ("batch", dict(training_batch_size=128)),
                ("precision", dict(precision="bfloat16")),
                ("checkpointing", dict(activation_checkpointing=dict(expected["activation_checkpointing"], preserve_rng_state=False))),
                ("non-reentrant", dict(activation_checkpointing=dict(expected["activation_checkpointing"], use_reentrant=True))),
                ("completion", dict(exit_status="failed")),
                ("environment", dict(torch="different-version")),
                ("UUID", dict(gpu_uuid="00000000-0000-0000-0000-000000000008")),
                ("peak bytes", dict(peak_cuda_memory_allocated=4096)),
                ("snapshot", dict(launcher_sha256="0" * 64)),
            ]
            for label, changes in cases:
                with self.subTest(constraint=label):
                    metadata = copy.deepcopy(expected)
                    metadata.update(changes)
                    (run / "execution.json").write_text(json.dumps(metadata))
                    with self.assertRaises(VerificationError), redirect_stdout(io.StringIO()):
                        verify_execution(run, config, environment, current)
            (run / "execution.json").write_text(json.dumps(expected))
            with self.assertRaisesRegex(VerificationError, "CUDA_VISIBLE_DEVICES=7"):
                verify_execution(run, config, dict(environment, cuda_visible_devices="1"), current)

    def test_runtime_launcher_drift_and_snapshot_corruption(self):
        with TemporaryDirectory() as temporary:
            run, current, config, environment, _ = self.make_execution(Path(temporary))
            (current / "scripts" / "train_gpu7.py").write_text("# later launcher\n")
            captured = io.StringIO()
            with redirect_stdout(captured):
                verify_execution(run, config, environment, current)
            self.assertIn("current GPU7 launcher differs", captured.getvalue())
            with self.assertRaisesRegex(VerificationError, "Current GPU7 launcher differs"):
                verify_execution(run, config, environment, current, strict_current_source=True)
            (run / "launcher" / "train_gpu7.py").write_text("# corrupted snapshot\n")
            with self.assertRaisesRegex(VerificationError, "Frozen GPU7 launcher hash differs"):
                verify_execution(run, config, environment, current)


if __name__ == "__main__":
    unittest.main()
