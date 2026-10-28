import unittest
from pathlib import Path

import monolithic_milp
from validate_solution import check_schedule

INSTANCE = Path(__file__).parent / (
    "synthetic_data_generator/singlecommodity_scenarios/F4_operational_conflict_L/seed_001/"
    "F4_operational_conflict_L_seed_001.json"
)


class ValidatorCatchesViolations(unittest.TestCase):
    """Each test starts from a hand-built schedule that passes every rule, breaks one
    thing, and checks that exactly that rule is reported."""

    @classmethod
    def setUpClass(cls):
        cls.data = data = monolithic_milp.load_scenario(INSTANCE)
        cls.asset_of = {m: a for a, ms in data["M_a"].items() for m in ms}
        # Greedy earliest-feasible F-mode schedule with strict no-overlap and fixed days respected.
        busy, schedule = set(), []
        for m in sorted(data["M"], key=lambda m: (data["l_m"][m], m)):
            a, dur = cls.asset_of[m], data["p"][(m, "F")]
            for s in range(data["e_m"][m], data["l_m"][m] + 1):
                days = range(s, s + dur)
                if all((a, t) not in busy and not data["h"].get((a, t), 0) for t in days):
                    busy.update((a, t) for t in days)
                    schedule.append((m, "F", s))
                    break
        cls.valid = schedule
        assert len(schedule) == len(data["M"]), "fixture: could not build a fully valid schedule"

    def violations(self, schedule):
        return {k for k, v in check_schedule(self.data, schedule).items() if v}

    def test_fixture_passes_every_rule(self):
        self.assertEqual(self.violations(self.valid), set())

    def test_start_outside_window(self):
        m, k, _ = self.valid[0]
        self.assertIn("start_outside_window", self.violations([(m, k, self.data["l_m"][m] + 1)] + self.valid[1:]))

    def test_missing_and_duplicate_jobs(self):
        self.assertEqual(self.violations(self.valid[1:]), {"not_scheduled"})
        self.assertIn("scheduled_more_than_once", self.violations(self.valid + [self.valid[0]]))

    def test_invalid_mode(self):
        m, _, s = self.valid[0]
        self.assertIn("invalid_mode", self.violations([(m, "P50", s)] + self.valid[1:]))

    def test_fixed_activity_day(self):
        blocked = {(a, t) for (a, t), v in self.data["h"].items() if v}
        self.assertTrue(blocked, "fixture instance should have blocking restrictions (F4)")
        for i, (m, k, _) in enumerate(self.valid):
            a = self.asset_of[m]
            for s in range(self.data["e_m"][m], self.data["l_m"][m] + 1):
                if any((a, t) in blocked for t in range(s, s + self.data["p"][(m, k)])):
                    broken = self.valid[:i] + [(m, k, s)] + self.valid[i + 1:]
                    self.assertIn("fixed_activity_day", self.violations(broken))
                    return
        self.skipTest("no job window touches a blocked day on this instance")

    def test_overlap_and_capacity_loss(self):
        by_asset = {}
        for m, k, s in self.valid:
            by_asset.setdefault(self.asset_of[m], []).append((m, k, s))
        for a, jobs in by_asset.items():
            for m2, k2, _ in jobs[1:]:
                m1, _, s1 = jobs[0]
                if self.data["e_m"][m2] <= s1 <= self.data["l_m"][m2]:
                    others = [j for j in self.valid if j[0] != m2]
                    found = self.violations(others + [(m2, k2, s1)])
                    # Two full shutdowns on one asset at once: both an overlap and 200% loss.
                    self.assertIn("same_asset_overlap", found)
                    self.assertIn("capacity_loss_over_100pct", found)
                    return
        self.skipTest("no two jobs on one asset share a feasible start day")


if __name__ == "__main__":
    unittest.main()
