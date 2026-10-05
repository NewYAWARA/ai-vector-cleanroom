import unittest

from execution_control import (
    CandidateBudgetExceeded,
    ConversionCancelled,
    ConversionTimedOut,
    ExecutionControl,
)


class _FakeClock:
    def __init__(self, value=100.0):
        self.value = float(value)

    def __call__(self):
        return self.value


class ExecutionControlTests(unittest.TestCase):
    def test_deadline_is_monotonic_and_fail_closed(self):
        clock = _FakeClock()
        control = ExecutionControl(
            budget_seconds=5.0, candidate_cap=2, clock=clock)
        control.checkpoint("prepare")
        clock.value = 104.999
        control.checkpoint("candidate_search")
        clock.value = 105.0
        with self.assertRaises(ConversionTimedOut):
            control.checkpoint("candidate_search")

    def test_cancel_before_candidate_does_not_consume_budget(self):
        cancelled = {"value": True}
        control = ExecutionControl(
            candidate_cap=2,
            cancel_requested=lambda: cancelled["value"])
        with self.assertRaises(ConversionCancelled):
            control.before_candidate(("base",))
        self.assertEqual(control.candidate_started, 0)
        self.assertEqual(control.candidate_evaluated, 0)

    def test_candidate_cap_counts_unique_attempt_boundaries(self):
        control = ExecutionControl(candidate_cap=2)
        control.set_candidate_plan(2)
        control.before_candidate(("base",))
        control.after_candidate()
        control.before_candidate(("gradients", "off"))
        control.after_candidate()
        with self.assertRaises(CandidateBudgetExceeded):
            control.before_candidate(("geometry", "off"))
        self.assertEqual(control.candidate_started, 2)
        self.assertEqual(control.candidate_evaluated, 2)
        self.assertEqual(control.snapshot()["candidate_planned"], 2)

    def test_plan_larger_than_cap_fails_before_any_candidate(self):
        control = ExecutionControl(candidate_cap=2)
        with self.assertRaises(CandidateBudgetExceeded):
            control.set_candidate_plan(3)
        self.assertEqual(control.candidate_started, 0)
        self.assertEqual(control.candidate_evaluated, 0)


if __name__ == "__main__":
    unittest.main()
