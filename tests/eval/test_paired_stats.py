import unittest

from eval.paired_stats import (
    cluster_bootstrap_gap,
    mcnemar_exact,
    pair_checks,
)


def _row(sid, lang, **metrics):
    row = {"scenario_id": sid, "language": lang}
    for name, passed in metrics.items():
        row[name + "_applicable"] = passed is not None
        row[name + "_passed"] = bool(passed)
    return row


FIXTURE = [
    _row("s01-a", "en", specialty=True, state=True),
    _row("s01-a", "ar", specialty=False, state=True),
    _row("s02-b", "en", specialty=True, state=None),   # state not applicable in en
    _row("s02-b", "ar", specialty=True, state=True),
    _row("s16-x", "en", specialty=True),               # outside core-15: excluded
    _row("s16-x", "ar", specialty=False),
]


class McNemarTests(unittest.TestCase):
    def test_known_value_ten_zero(self):
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 * (0.5 ** 10))

    def test_symmetric_discordance_is_insignificant(self):
        self.assertEqual(mcnemar_exact(1, 1), 1.0)  # raw 1.5 capped

    def test_no_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_symmetry(self):
        self.assertEqual(mcnemar_exact(7, 2), mcnemar_exact(2, 7))


class PairChecksTests(unittest.TestCase):
    def test_pairs_only_core15_and_both_applicable(self):
        pairs = pair_checks(FIXTURE, "ar")
        keys = {(sid, metric) for sid, metric, _, _ in pairs}
        self.assertEqual(keys, {("s01-a", "specialty"), ("s01-a", "state"), ("s02-b", "specialty")})

    def test_pass_values_line_up(self):
        pairs = {(sid, m): (e, a) for sid, m, e, a in pair_checks(FIXTURE, "ar")}
        self.assertEqual(pairs[("s01-a", "specialty")], (True, False))
        self.assertEqual(pairs[("s02-b", "specialty")], (True, True))


class ClusterBootstrapTests(unittest.TestCase):
    def test_deterministic_with_seed(self):
        a = cluster_bootstrap_gap(FIXTURE, "ar", n_boot=200, seed=42)
        b = cluster_bootstrap_gap(FIXTURE, "ar", n_boot=200, seed=42)
        self.assertEqual(a, b)

    def test_point_gap_matches_hand_count(self):
        gap, lo, hi = cluster_bootstrap_gap(FIXTURE, "ar", n_boot=200, seed=42)
        # ar passes 2/3 paired checks, en passes 3/3 -> gap = -1/3
        self.assertAlmostEqual(gap, -1 / 3)
        self.assertLessEqual(lo, gap)
        self.assertGreaterEqual(hi, gap)


if __name__ == "__main__":
    unittest.main()
