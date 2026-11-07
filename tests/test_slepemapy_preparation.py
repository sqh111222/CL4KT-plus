"""Semantic comparison against upstream, using temporary synthetic fixtures only.

The fixture is not an experiment dataset. Both implementations receive exactly
the same raw CSV and seed; all outputs live in TemporaryDirectory, never dataset/.
"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

import preprocess_data


ROOT = Path(__file__).resolve().parents[1]


class SlepemapyPreparationTests(unittest.TestCase):
    @staticmethod
    def raw_fixture():
        rows = []
        for index in range(5003):
            user = 100000 + 17 * index
            # Users 0/1 fall below five valid answers after NaN removal. User 2
            # remains eligible despite one missing answer, leaving 5,001 eligible
            # students so sampling actually excludes one eligible student.
            missing_steps = {0, 1, 2} if index in (0, 1) else ({0} if index == 2 else set())
            offsets = [6, 0, 6, 3, 0, 9, 3, 9]
            for step in range(5 + index % 4):
                question = 1 + (index * 3 + step) % 11
                answer = question if (index + step) % 3 else 1 + question % 11
                if step in missing_steps:
                    answer = np.nan
                rows.append(dict(
                    id=100 * index + step, user=user, place_asked=question,
                    place_answered=answer, type=1 + (index + step) % 2,
                    inserted=f"2015-05-21 12:00:{offsets[step]:02d}",
                    # The integer part of dwell_time identifies the raw user,
                    # independently recovering upstream's sampled order after
                    # that implementation discards the original student IDs.
                    response_time=user * 1000 + step,
                ))
        frame = pd.DataFrame(rows)
        return frame.iloc[np.random.RandomState(403).permutation(len(frame))].reset_index(drop=True)

    def test_optimized_default_5000_matches_original_seeded_preparation(self):
        if not hasattr(time, "tzset"):
            self.skipTest("The host must support TZ/tzset to isolate upstream local mktime")
        previous_base = preprocess_data.BASE_PATH
        previous_rng = np.random.get_state()
        previous_tz = os.environ.get("TZ")
        seed = 12405
        try:
            os.environ["TZ"] = "UTC"
            time.tzset()
            with TemporaryDirectory(prefix="cl4kt-slepemapy-equivalence-") as temporary:
                root = Path(temporary)
                original = root / "original" / "sampled_slepemapy"
                optimized = root / "optimized" / "sampled_slepemapy"
                original.mkdir(parents=True)
                raw = root / "raw-answer.csv"
                fixture = self.raw_fixture()
                self.assertEqual(fixture.user.nunique(), 5003)
                counts = fixture.loc[fixture.place_answered.notna()].groupby("user").size()
                self.assertEqual(int((counts >= 5).sum()), 5001)
                self.assertTrue(fixture.duplicated(["user", "inserted"]).any())
                fixture.to_csv(raw, sep=";", index=False)
                shutil.copyfile(raw, original / "answer.csv")
                preprocess_data.BASE_PATH = str(original.parent)
                np.random.seed(seed)
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
                    preprocess_data.prepare_sampled_slepemapy(min_user_inter_num=5)
                command = [sys.executable, str(ROOT / "scripts" / "prepare_slepemapy.py"),
                           "--raw", str(raw), "--seed", str(seed),
                           "--output-dir", str(optimized)]
                # --students and --min-user-inter-num are intentionally omitted:
                # the optimized CLI defaults must preserve the original protocol.
                completed = subprocess.run(command, capture_output=True, text=True,
                                           env=dict(os.environ, TZ="UTC"), check=False)
                self.assertEqual(completed.returncode, 0,
                                 msg=f"Optimized CLI failed:\n{completed.stdout}\n{completed.stderr}")
                reference = pd.read_csv(original / "preprocessed_df.csv", sep="\t")
                result = pd.read_csv(optimized / "preprocessed_df.csv", sep="\t")
                pd.testing.assert_frame_equal(result, reference, check_exact=True)
                self.assertEqual(result.user_id.nunique(), 5000)

                # Recover original IDs from fixture payload; no duplicated
                # implementation of the sampler or sorting algorithm is used.
                raw_users = np.floor(reference.groupby("user_id", sort=True).dwell_time.first()).astype(int)
                mapping = pd.read_csv(optimized / "student_mapping.csv")
                self.assertEqual(mapping.user_id.tolist(), list(range(1, 5001)))
                self.assertEqual(mapping.original_user_id.tolist(), raw_users.tolist())
                excluded = set(fixture.user.unique()) - set(mapping.original_user_id)
                self.assertIn(100000, excluded)
                self.assertIn(100017, excluded)
                self.assertEqual(len(excluded), 3)
                self.assertTrue(result.groupby("user_id", sort=False).timestamp.apply(
                    lambda values: values.is_monotonic_increasing).all())

                reference_q = sparse.load_npz(original / "q_mat.npz")
                result_q = sparse.load_npz(optimized / "q_mat.npz")
                self.assertEqual(result_q.shape, reference_q.shape)
                self.assertEqual(result_q.dtype, reference_q.dtype)
                self.assertEqual((result_q != reference_q).nnz, 0)
                answer_mapping = pd.read_csv(optimized / "answer_mapping.csv")
                source_answers = fixture.set_index("id").loc[answer_mapping.original_answer_id]
                self.assertEqual(answer_mapping.user_id.tolist(), result.user_id.tolist())
                np.testing.assert_array_equal(source_answers.response_time.to_numpy() / 1000.0,
                                              result.dwell_time.to_numpy())
                manifest = json.loads((optimized / "manifest.json").read_text())
                self.assertEqual(manifest["sampling_seed"], seed)
                self.assertEqual(manifest["eligible_students"], 5001)
                self.assertEqual(manifest["sample_students"], 5000)
        finally:
            preprocess_data.BASE_PATH = previous_base
            np.random.set_state(previous_rng)
            if previous_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous_tz
            time.tzset()


if __name__ == "__main__":
    unittest.main()
