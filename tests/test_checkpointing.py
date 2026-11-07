"""Activation checkpointing preserves FP32/dropout training semantics on CPU."""

import copy
import json
import os
import unittest
from unittest.mock import patch

import torch

from models.cl4kt import CL4KT
from models.modules import CL4KTTransformerLayer
from scripts.train_gpu7 import install_activation_checkpointing, physical_gpu_details


class CheckpointingTests(unittest.TestCase):
    def test_forward_loss_all_gradients_and_rng_are_exactly_preserved(self):
        self.assert_equivalence(torch.device("cpu"))

    def test_cuda_dropout_and_device_rng_are_exactly_preserved(self):
        if os.environ.get("CL4KT_TEST_DEVICE") != "cuda:0":
            self.skipTest("Set CL4KT_TEST_DEVICE=cuda:0 to opt in on physical GPU7")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "7")
        self.assertEqual(os.environ.get("CUDA_DEVICE_ORDER"), "PCI_BUS_ID")
        self.assertTrue(torch.cuda.is_available(), "Requested physical GPU7 is unavailable")
        self.assert_equivalence(torch.device("cuda:0"))

    def assert_equivalence(self, device):
        old_threads = torch.get_num_threads()
        old_rng = torch.get_rng_state()
        old_cuda_rng = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_initialized() or device.type == "cuda" else None)
        torch.set_num_threads(2)
        try:
            torch.manual_seed(31)
            ordinary = CL4KT(
                12, 16, 8, hidden_size=16, num_blocks=2, num_attn_heads=2,
                kq_same=True, final_fc_dim=32, d_ff=32, l2=0.0,
                dropout=0.2, reg_cl=0.1, negative_prob=1.0,
                hard_negative_weight=1.0, temp=0.05,
            ).to(device)
            checkpointed = copy.deepcopy(ordinary)
            skills = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                                   [8, 7, 6, 5, 4, 3, 2, 1]], device=device)
            responses = torch.tensor([[0, 1, 0, 1, 1, 0, 0, 1],
                                      [1, 1, 0, 1, 0, 0, 1, 1]], device=device)
            mask = torch.ones_like(skills)
            batch = dict(skills=(skills, skills.flip(1), skills),
                         responses=(responses, responses.flip(1), responses, 1 - responses),
                         attention_mask=(mask, mask, mask))

            def forward_backward(model):
                torch.manual_seed(919)
                model.train()
                model.zero_grad(set_to_none=True)
                result = model(batch)
                loss, _, _ = model.loss(batch, result)
                loss.backward()
                gradients = {name: None if value.grad is None else value.grad.detach().clone()
                             for name, value in model.named_parameters()}
                return (result["pred"].detach(), result["cl_loss"].detach(),
                        loss.detach(), gradients, torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if device.type == "cuda" else None)

            expected = forward_backward(ordinary)
            original_forward = CL4KTTransformerLayer.forward
            restore = install_activation_checkpointing()
            try:
                actual = forward_backward(checkpointed)
                for name, left, right in zip(("prediction", "contrastive loss", "total loss"),
                                             expected[:3], actual[:3]):
                    with self.subTest(value=name):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(set(expected[3]), set(actual[3]))
                for name, gradient in expected[3].items():
                    with self.subTest(parameter=name):
                        other = actual[3][name]
                        if gradient is None:
                            self.assertIsNone(other)
                        else:
                            self.assertIsNotNone(other)
                            self.assertTrue(torch.isfinite(other).all().item())
                            torch.testing.assert_close(gradient, other, rtol=0, atol=0)
                self.assertTrue(torch.equal(expected[4], actual[4]),
                                "Checkpoint recomputation must preserve the post-backward RNG state")
                if device.type == "cuda":
                    self.assertTrue(torch.equal(expected[5], actual[5]),
                                    "Checkpoint recomputation must preserve the CUDA device RNG state")
                gradient_differences = [(gradient - actual[3][name]).abs().max().item()
                                        for name, gradient in expected[3].items() if gradient is not None]
                evidence = dict(
                    passed=True, device=str(device), model_seed=31, forward_seed=919,
                    batch_size=2, hidden_size=16, num_blocks=2, dropout=0.2,
                    torch=str(torch.__version__), precision="float32",
                    prediction_max_abs_diff=float((expected[0] - actual[0]).abs().max()),
                    contrastive_loss_abs_diff=float((expected[1] - actual[1]).abs()),
                    total_loss_abs_diff=float((expected[2] - actual[2]).abs()),
                    gradient_max_abs_diff=max(gradient_differences),
                    parameter_tensors_compared=len(expected[3]),
                    parameter_gradients_compared=len(gradient_differences),
                    cpu_rng_exact=torch.equal(expected[4], actual[4]),
                    cuda_rng_exact=torch.equal(expected[5], actual[5]) if device.type == "cuda" else None,
                )
                if device.type == "cuda":
                    evidence.update(
                        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                        cuda_device_order=os.environ["CUDA_DEVICE_ORDER"],
                        physical_gpu_details=physical_gpu_details(),
                        cuda_device_uuid=str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
                    )
                print(json.dumps(evidence, indent=2), flush=True)
                evaluation = dict(skills=skills, responses=responses, attention_mask=mask)
                with patch("torch.utils.checkpoint.checkpoint", side_effect=AssertionError("unexpected checkpoint")):
                    checkpointed.eval()
                    with torch.no_grad():
                        checkpointed(evaluation)
                    checkpointed.train()
                    with torch.no_grad():
                        checkpointed(batch)
            finally:
                restore()
            self.assertIs(CL4KTTransformerLayer.forward, original_forward)
        finally:
            torch.set_rng_state(old_rng)
            if old_cuda_rng is not None:
                torch.cuda.set_rng_state_all(old_cuda_rng)
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
