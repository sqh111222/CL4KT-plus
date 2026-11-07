"""Regression checks for correctness fixes to the official CL4KT model."""

import unittest

import torch

from models.cl4kt import CL4KT
from models.modules import individual_attention, monotonic_attention


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self):
        torch.manual_seed(1)
        return CL4KT(
            20, 30, 8,
            hidden_size=8, num_blocks=2, num_attn_heads=2,
            kq_same=True, final_fc_dim=16, d_ff=16, l2=0.0,
            dropout=0.0, reg_cl=0.1, negative_prob=1.0,
            hard_negative_weight=1.0, temp=0.05,
        ).cpu()

    def eval_batch(self):
        skills = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ], device="cpu")
        responses = torch.tensor([
            [0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
        ], device="cpu")
        return dict(skills=skills, responses=responses,
                    attention_mask=torch.ones_like(skills))

    def test_predictions_do_not_see_current_or_future_responses(self):
        model = self.make_model().eval()
        batch = self.eval_batch()
        with torch.no_grad():
            original = model(batch)["pred"]
            for target in (1, 3, 5):
                changed = dict(batch)
                changed["responses"] = batch["responses"].clone()
                changed["responses"][:, target:] = 1 - changed["responses"][:, target:]
                after = model(changed)["pred"]
                # pred[:, 0] is the response at original sequence position 1.
                torch.testing.assert_close(original[:, :target], after[:, :target],
                                           rtol=0, atol=0)

    def test_cpu_training_loss_and_backward_are_finite(self):
        model = self.make_model().train()
        base = self.eval_batch()
        q, r, mask = base["skills"], base["responses"], base["attention_mask"]
        batch = dict(skills=(q, q.flip(1), q),
                     responses=(r, r.flip(1), r, 1-r),
                     attention_mask=(mask, mask, mask))
        output = model(batch)
        loss, count, _ = model.loss(batch, output)
        self.assertEqual(count, 14)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))

    def test_single_student_evaluation_preserves_batch_dimension(self):
        model = self.make_model().eval()
        batch = {k: v[:1] for k, v in self.eval_batch().items()}
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(output["pred"].shape, (1, 7))
        self.assertTrue(torch.isfinite(output["pred"]).all().item())

    def test_mask_and_real_skill_response_embeddings_do_not_collide(self):
        model = self.make_model()
        vocabulary_size = model.num_skills + 2
        # Give each row a deterministic fingerprint, then inspect embeddings
        # returned by the public method rather than repeating its index formula.
        with torch.no_grad():
            model.interaction_embed.weight.zero_()
            model.interaction_embed.weight[:, 0] = torch.arange(
                2 * vocabulary_size, dtype=torch.float, device="cpu")
        skills = torch.arange(vocabulary_size, device="cpu").repeat(2)
        responses = torch.cat((torch.zeros(vocabulary_size, dtype=torch.long),
                               torch.ones(vocabulary_size, dtype=torch.long)))
        fingerprints = model.get_interaction_embed(skills, responses)[:, 0]
        self.assertEqual(fingerprints.unique().numel(), 2 * vocabulary_size)
        mask_id = model.num_skills + 1
        self.assertNotEqual(fingerprints[mask_id].item(),
                            fingerprints[vocabulary_size + 1].item())
        padding = model.get_interaction_embed(torch.tensor([0]), torch.tensor([-1]))
        self.assertTrue(torch.equal(padding, torch.zeros_like(padding)))

    def test_fully_masked_attention_rows_are_zero(self):
        torch.manual_seed(7)
        q = torch.randn(1, 2, 4, 3, device="cpu")
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool, device="cpu").tril(-1)
        gamma = torch.zeros(2, 1, 1, device="cpu")
        for attention in (individual_attention, monotonic_attention):
            with self.subTest(attention=attention.__name__):
                output, weights = attention(q, q, q, 3, mask, torch.nn.Dropout(0), gamma)
                self.assertTrue(torch.equal(output[:, :, 0], torch.zeros_like(output[:, :, 0])))
                self.assertTrue(torch.equal(weights[:, :, 0], torch.zeros_like(weights[:, :, 0])))
                self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
