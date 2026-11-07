import unittest
import numpy as np
import pandas as pd
from data_loaders import MostRecentQuestionSkillDataset, SimCLRDatasetWrapper
from main import make_splits
from utils.augment_seq import augment_kt_seqs


class DataTests(unittest.TestCase):
    def augment(self, crop, permute):
        return augment_kt_seqs([0, 0, 1, 2], [0, 0, 1, 2], [-1, -1, 0, 1],
                               0, crop, permute, 0, 1, {}, {}, 5, 5, 4, seed=7)

    def test_augmentation_boundaries_and_padding(self):
        for crop in (0, 0.3, 1):
            result = self.augment(crop, 1)
            self.assertEqual(len(result[0]), 4)
            self.assertEqual(result[4], [int(s > 0) for s in result[1]])
            self.assertEqual(result[3], [-1, -1, 1, 0])

    def test_augmentation_does_not_reset_global_rng(self):
        np.random.seed(99)
        expected = np.random.random(2)
        np.random.seed(99)
        first = np.random.random()
        self.augment(0.3, 0.5)
        second = np.random.random()
        np.testing.assert_equal([first, second], expected)

    def test_recent_history_and_student_splits(self):
        rows = [dict(user_id=u, item_id=t % 3 + 1, skill_id=t % 2 + 1, correct=t % 2)
                for u in range(20) for t in range(8)]
        df = pd.DataFrame(rows)
        ds = MostRecentQuestionSkillDataset(df, 4, 3, 4)
        self.assertEqual(ds[0]["questions"].tolist(), [2, 3, 1, 2])
        config = dict(seed=12405, num_folds=5, validation_fraction=0.1)
        splits = make_splits(df, config)
        self.assertEqual(splits, make_splits(df, config))
        self.assertEqual(sorted(u for s in splits for u in s["test"]), list(range(20)))


if __name__ == "__main__":
    unittest.main()
